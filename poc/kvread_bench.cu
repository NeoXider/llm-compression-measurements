// PoC-B: can a fixed-rate KV format be read faster than F16 / Q8_0 on an RTX 3090?
// One decode-attention-shaped pass per format over T tokens of ONE layer
// (8 KV heads x 128 dims, 32 query heads, GQA 4):
//   scores[qh][t] = dot(q[qh], K[t][kvh])          (reads K once)
//   out[qh][d]   += sum_t p[qh][t] * V[t][kvh][d]  (reads V once)
// Formats: F16 (raw), Q8_0 (llama.cpp block layout, 34 B / 32), Q4_0 (18 B / 32),
//          EXP3 (lossless: per 128-value row 4 base exponents + overflow offset; per value
//                sign|3-bit exponent offset|10-bit mantissa = 14 bits; offset 7 = escape,
//                exact exponent in a per-row overflow byte list).
// Data: real F16 KV from a llama.cpp state export (K and V of all layers, tiled to T tokens).
// Correctness: every format's device decode is checked against a host reference
// (EXP3 must be bit-exact with the F16 source) before timing.
// Timing: cudaEvent around ITER back-to-back launches; formats interleaved ABBA, REPS rounds.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

static const int HKV = 8, D = 128, HQ = 32, G = HQ / HKV;
static const int ROW = HKV * D;               // values per token
static const int EXP3_ROW_BYTES = 232;        // 4 B bases + 2 B overflow offset + 224 B payload + 2 pad

enum Fmt { F16 = 0, Q8 = 1, Q4 = 2, EXP3 = 3 };
static const char *NAMES[] = {"f16", "q8_0", "q4_0", "exp3"};

// ---------------------------------------------------------------- host encoders
static uint16_t f2h_bits(float f) { __half h = __float2half_rn(f); uint16_t b; memcpy(&b, &h, 2); return b; }
static float h2f_bits(uint16_t b) { __half h; memcpy(&h, &b, 2); return __half2float(h); }

static void enc_q8(const uint16_t *x, uint8_t *out) {           // 32 values -> 34 B
    float v[32], amax = 0; for (int i = 0; i < 32; i++) { v[i] = h2f_bits(x[i]); amax = std::max(amax, fabsf(v[i])); }
    float d = amax / 127.f, id = d ? 1.f / d : 0.f; uint16_t db = f2h_bits(d); memcpy(out, &db, 2);
    for (int i = 0; i < 32; i++) out[2 + i] = (uint8_t)(int8_t)lrintf(v[i] * id);
}
static void enc_q4(const uint16_t *x, uint8_t *out) {           // 32 values -> 18 B (llama.cpp q4_0)
    float v[32], amax = 0, mx = 0; for (int i = 0; i < 32; i++) { v[i] = h2f_bits(x[i]); if (fabsf(v[i]) > amax) { amax = fabsf(v[i]); mx = v[i]; } }
    float d = mx / -8.f, id = d ? 1.f / d : 0.f; uint16_t db = f2h_bits(d); memcpy(out, &db, 2);
    for (int j = 0; j < 16; j++) {
        int a = std::min(15, (int)(v[j] * id + 8.5f)), b = std::min(15, (int)(v[j + 16] * id + 8.5f));
        a = std::max(a, 0); b = std::max(b, 0); out[2 + j] = (uint8_t)(a | (b << 4));
    }
}
// EXP3 per 128-value row (one kv head of one token). Returns escapes appended to ovf.
static void enc_exp3(const uint16_t *x, uint8_t *row, std::vector<uint8_t> &ovf, size_t &escapes) {
    uint8_t base[4];
    for (int g = 0; g < 4; g++) { int b = 0; for (int i = 0; i < 32; i++) b = std::max(b, (int)((x[g * 32 + i] >> 10) & 31)); base[g] = (uint8_t)b; }
    memcpy(row, base, 4);
    size_t start = ovf.size(); if (start > 0xFFFFFFull) { fprintf(stderr, "overflow index too large\n"); exit(1); }
    uint32_t s24 = (uint32_t)start;                              // 24-bit start, bytes 4..6
    row[4] = s24 & 255; row[5] = (s24 >> 8) & 255;               // low 16 bits here
    row[230] = (s24 >> 16) & 255; row[231] = 0;                  // high 8 bits in pad
    uint8_t *pay = row + 6; memset(pay, 0, 224);
    for (int i = 0; i < 128; i++) {
        uint16_t w = x[i]; int e = (w >> 10) & 31, off = base[i / 32] - e;
        uint32_t code;
        if (off >= 7) { code = ((w >> 15) << 13) | (7u << 10) | (w & 1023); ovf.push_back((uint8_t)e); escapes++; }
        else code = ((w >> 15) << 13) | ((uint32_t)off << 10) | (w & 1023);
        size_t bit = (size_t)i * 14;
        for (int k = 0; k < 14; k++) if (code >> k & 1) pay[(bit + k) >> 3] |= (uint8_t)(1u << ((bit + k) & 7));
    }
}

// ---------------------------------------------------------------- device decoders (4 values per lane)
// Lane l handles dims {4l..4l+3} of one 128-dim head row, except Q4 (layout-dictated dims).
struct Four { float v[4]; int d[4]; };

__device__ __forceinline__ Four dec_f16(const uint8_t *row, int lane) {
    Four r; uint2 u = *reinterpret_cast<const uint2 *>(row + lane * 8);
    __half2 a = *reinterpret_cast<__half2 *>(&u.x), b = *reinterpret_cast<__half2 *>(&u.y);
    float2 fa = __half22float2(a), fb = __half22float2(b);
    r.v[0] = fa.x; r.v[1] = fa.y; r.v[2] = fb.x; r.v[3] = fb.y;
    for (int k = 0; k < 4; k++) r.d[k] = lane * 4 + k; return r;
}
__device__ __forceinline__ Four dec_q8(const uint8_t *row, int lane) {     // row = 4 blocks x 34 B
    Four r; const uint8_t *blk = row + (lane >> 3) * 34; int j = (lane & 7) * 4;
    __half dh = *reinterpret_cast<const __half *>(blk); float d = __half2float(dh);
    const uint16_t *q = reinterpret_cast<const uint16_t *>(blk + 2 + j);
    uint16_t a = q[0], b = q[1];
    r.v[0] = d * (int8_t)(a & 255); r.v[1] = d * (int8_t)(a >> 8); r.v[2] = d * (int8_t)(b & 255); r.v[3] = d * (int8_t)(b >> 8);
    for (int k = 0; k < 4; k++) r.d[k] = lane * 4 + k; return r;
}
__device__ __forceinline__ Four dec_q4(const uint8_t *row, int lane) {     // row = 4 blocks x 18 B
    Four r; int blk_i = lane >> 3, j = (lane & 7) * 2; const uint8_t *blk = row + blk_i * 18;
    __half dh = *reinterpret_cast<const __half *>(blk); float d = __half2float(dh);
    uint16_t q = *reinterpret_cast<const uint16_t *>(blk + 2 + j);
    uint8_t q0 = q & 255, q1 = q >> 8;
    r.v[0] = d * ((q0 & 15) - 8); r.v[1] = d * ((q1 & 15) - 8); r.v[2] = d * ((q0 >> 4) - 8); r.v[3] = d * ((q1 >> 4) - 8);
    int b = blk_i * 32; r.d[0] = b + j; r.d[1] = b + j + 1; r.d[2] = b + j + 16; r.d[3] = b + j + 17; return r;
}
__device__ __forceinline__ void dec_exp3_bits(const uint8_t *row, const uint8_t *ovf, int lane, uint16_t out[4]) {
    const uint8_t *pay = row + 6; int byte0 = lane * 7;            // 56 bits per lane, starts at 7*lane bytes
    uint64_t w = 0; for (int k = 0; k < 7; k++) w |= (uint64_t)pay[byte0 + k] << (8 * k);
    uint32_t base = row[lane >> 3];
    uint32_t start = row[4] | (row[5] << 8) | (row[230] << 16);
    uint32_t code[4]; int esc = 0;
    for (int k = 0; k < 4; k++) { code[k] = (uint32_t)(w >> (14 * k)) & 0x3FFF; esc += ((code[k] >> 10) & 7) == 7; }
    unsigned m = __ballot_sync(0xffffffffu, esc != 0);
    int before = 0;
    if (m) {                                                       // warp prefix of escape counts
        int incl = esc;
        for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += n; }
        before = incl - esc;
    }
    for (int k = 0; k < 4; k++) {
        uint32_t c = code[k], off = (c >> 10) & 7, e;
        if (off == 7) e = ovf[start + before++]; else e = base - off;
        out[k] = (uint16_t)(((c >> 13) << 15) | (e << 10) | (c & 1023));
    }
}
__device__ __forceinline__ Four dec_exp3(const uint8_t *row, const uint8_t *ovf, int lane) {
    uint16_t b[4]; dec_exp3_bits(row, ovf, lane, b); Four r;
    for (int k = 0; k < 4; k++) { r.v[k] = __half2float(__ushort_as_half(b[k])); r.d[k] = lane * 4 + k; }
    return r;
}

template <int F> __device__ __forceinline__ Four dec(const uint8_t *row, const uint8_t *ovf, int lane) {
    if (F == F16) return dec_f16(row, lane);
    if (F == Q8) return dec_q8(row, lane);
    if (F == Q4) return dec_q4(row, lane);
    return dec_exp3(row, ovf, lane);
}

// row stride per (token, kv head) in bytes
__host__ __device__ constexpr int row_bytes(int f) { return f == F16 ? 256 : f == Q8 ? 136 : f == Q4 ? 72 : EXP3_ROW_BYTES; }

// K pass: grid-stride over (token, kv head) rows, one warp per row; q loaded once per block.
template <int F> __global__ void kpass(const uint8_t *K, const uint8_t *ovf, const float *q, float *scores, int T) {
    __shared__ float qs[HQ * D];
    for (int i = threadIdx.x; i < HQ * D; i += blockDim.x) qs[i] = q[i];
    __syncthreads();
    int lane = threadIdx.x & 31, wpb = blockDim.x >> 5;
    int warp = blockIdx.x * wpb + (threadIdx.x >> 5), nwarps = gridDim.x * wpb;
    for (int r = warp; r < T * HKV; r += nwarps) {
        int t = r / HKV, h = r % HKV;
        Four x = dec<F>(K + (size_t)r * row_bytes(F), ovf, lane);
        for (int g = 0; g < G; g++) {
            const float *qq = qs + (h * G + g) * D; float s = 0;
            for (int k = 0; k < 4; k++) s += qq[x.d[k]] * x.v[k];
            for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (lane == 0) scores[(size_t)(h * G + g) * T + t] = s;
        }
    }
}
// V pass: blockIdx.y = kv head; each warp walks TPW consecutive tokens; block reduces in shared
// memory and issues one set of atomics per block.
template <int F, int TPW> __global__ void vpass(const uint8_t *V, const uint8_t *ovf, const float *p, float *out, int T) {
    __shared__ float red[G * D];
    for (int i = threadIdx.x; i < G * D; i += blockDim.x) red[i] = 0;
    __syncthreads();
    int lane = threadIdx.x & 31, h = blockIdx.y, wpb = blockDim.x >> 5;
    int t0 = (blockIdx.x * wpb + (threadIdx.x >> 5)) * TPW;
    float acc[G][4] = {}; int dims[4] = {0, 0, 0, 0};
    for (int t = t0; t < min(T, t0 + TPW); t++) {
        Four x = dec<F>(V + ((size_t)t * HKV + h) * row_bytes(F), ovf, lane);
        for (int k = 0; k < 4; k++) dims[k] = x.d[k];
        for (int g = 0; g < G; g++) { float w = p[(size_t)(h * G + g) * T + t]; for (int k = 0; k < 4; k++) acc[g][k] += w * x.v[k]; }
    }
    for (int g = 0; g < G; g++) for (int k = 0; k < 4; k++) atomicAdd(&red[g * D + dims[k]], acc[g][k]);
    __syncthreads();
    for (int i = threadIdx.x; i < G * D; i += blockDim.x) atomicAdd(&out[h * G * D + i], red[i]);
}
// Full decode to F16 for correctness checks.
template <int F> __global__ void decode_all(const uint8_t *X, const uint8_t *ovf, float *outf, uint16_t *outbits, int T) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int t = warp / HKV, h = warp % HKV; if (t >= T) return;
    const uint8_t *row = X + ((size_t)t * HKV + h) * row_bytes(F);
    if (F == EXP3) { uint16_t b[4]; dec_exp3_bits(row, ovf, lane, b); for (int k = 0; k < 4; k++) outbits[((size_t)t * HKV + h) * D + lane * 4 + k] = b[k]; }
    Four x = dec<F>(row, ovf, lane);
    for (int k = 0; k < 4; k++) outf[((size_t)t * HKV + h) * D + x.d[k]] = x.v[k];
}

// ---------------------------------------------------------------- host
struct Buf { uint8_t *K = nullptr, *V = nullptr, *ovfK = nullptr, *ovfV = nullptr; size_t bytesK = 0, bytesV = 0, ovfBytes = 0, escapes = 0; };

static std::vector<uint16_t> load_tokens(const char *path, int T, bool wantV) {
    FILE *f = fopen(path, "rb"); if (!f) { perror("state"); exit(1); }
    uint32_t hdr[4]; fread(hdr, 4, 4, f); uint32_t n = hdr[3];
    long off = 16 + 12L * n; fseek(f, off, SEEK_SET); uint32_t vt_nl[2]; fread(vt_nl, 4, 2, f); off += 8;
    int L = vt_nl[1]; size_t per = (size_t)n * ROW;
    std::vector<uint16_t> all((size_t)L * per);                     // all layers of K or V, token-major per layer
    for (int c = 0; c < 2; c++) for (int l = 0; l < L; l++) {
        int32_t dt; uint64_t rb; fread(&dt, 4, 1, f); fread(&rb, 8, 1, f);
        if (dt != 1 || rb != ROW * 2) { fprintf(stderr, "unexpected layout\n"); exit(1); }
        if ((c == 1) == wantV) fread(all.data() + l * per, 2, per, f); else fseek(f, (long)(per * 2), SEEK_CUR);
    }
    fclose(f);
    std::vector<uint16_t> out((size_t)T * ROW);
    size_t total = (size_t)L * n;
    for (int t = 0; t < T; t++) memcpy(out.data() + (size_t)t * ROW, all.data() + (size_t)(t % total) * ROW, ROW * 2);
    return out;
}

static void encode(int f, const std::vector<uint16_t> &x, int T, std::vector<uint8_t> &enc, std::vector<uint8_t> &ovf, size_t &esc) {
    enc.assign((size_t)T * HKV * row_bytes(f), 0); ovf.clear(); esc = 0;
    for (size_t r = 0; r < (size_t)T * HKV; r++) {
        const uint16_t *src = x.data() + r * D; uint8_t *dst = enc.data() + r * row_bytes(f);
        if (f == F16) memcpy(dst, src, 256);
        else if (f == Q8) for (int b = 0; b < 4; b++) enc_q8(src + b * 32, dst + b * 34);
        else if (f == Q4) for (int b = 0; b < 4; b++) enc_q4(src + b * 32, dst + b * 18);
        else enc_exp3(src, dst, ovf, esc);
    }
    if (ovf.empty()) ovf.push_back(0);
}

template <int F> static void launch(const Buf &b, const float *q, float *sc, const float *p, float *out, int T) {
    kpass<F><<<82 * 8, 256>>>(b.K, b.ovfK, q, sc, T);            // 8 blocks per SM on the 3090 (82 SMs)
    const int TPW = 16, WPB = 8; dim3 grid((T + TPW * WPB - 1) / (TPW * WPB), HKV);
    vpass<F, TPW><<<grid, WPB * 32>>>(b.V, b.ovfV, p, out, T);
}
static void run(int f, const Buf &b, const float *q, float *sc, const float *p, float *out, int T) {
    if (f == F16) launch<F16>(b, q, sc, p, out, T); else if (f == Q8) launch<Q8>(b, q, sc, p, out, T);
    else if (f == Q4) launch<Q4>(b, q, sc, p, out, T); else launch<EXP3>(b, q, sc, p, out, T);
}
template <int F> static void dec_launch(const uint8_t *X, const uint8_t *o, float *of, uint16_t *ob, int T) {
    int w = T * HKV; decode_all<F><<<(w * 32 + 255) / 256, 256>>>(X, o, of, ob, T);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: kvread_bench state.bin [T...] (env REPS, ITER)\n"); return 2; }
    int REPS = getenv("REPS") ? atoi(getenv("REPS")) : 12, ITER = getenv("ITER") ? atoi(getenv("ITER")) : 20;
    std::vector<int> Ts; for (int i = 2; i < argc; i++) Ts.push_back(atoi(argv[i])); if (Ts.empty()) Ts = {4096, 8192, 16384, 32768};
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("{\"device\":\"%s\",\"sm\":%d%d,\"reps\":%d,\"iter\":%d}\n", pr.name, pr.major, pr.minor, REPS, ITER);
    for (int T : Ts) {
        std::vector<uint16_t> hk = load_tokens(argv[1], T, false), hv = load_tokens(argv[1], T, true);
        Buf b[4]; std::vector<uint8_t> enc, ovf;
        for (int f = 0; f < 4; f++) {
            for (int c = 0; c < 2; c++) {
                size_t esc; encode(f, c ? hv : hk, T, enc, ovf, esc);
                uint8_t *d, *o; CK(cudaMalloc(&d, enc.size())); CK(cudaMemcpy(d, enc.data(), enc.size(), cudaMemcpyHostToDevice));
                CK(cudaMalloc(&o, ovf.size())); CK(cudaMemcpy(o, ovf.data(), ovf.size(), cudaMemcpyHostToDevice));
                if (c) { b[f].V = d; b[f].ovfV = o; b[f].bytesV = enc.size(); } else { b[f].K = d; b[f].ovfK = o; b[f].bytesK = enc.size(); }
                b[f].ovfBytes += (f == EXP3) ? ovf.size() : 0; b[f].escapes += esc;
            }
        }
        // correctness: decode all on device, compare
        size_t N = (size_t)T * ROW; float *df; uint16_t *db; CK(cudaMalloc(&df, N * 4)); CK(cudaMalloc(&db, N * 2));
        std::vector<float> hf(N); std::vector<uint16_t> hb(N);
        double maxrel[4] = {0, 0, 0, 0}; bool exact = true;
        for (int f = 0; f < 4; f++) for (int c = 0; c < 2; c++) {
            const uint8_t *X = c ? b[f].V : b[f].K, *O = c ? b[f].ovfV : b[f].ovfK; const auto &src = c ? hv : hk;
            if (f == F16) dec_launch<F16>(X, O, df, db, T); else if (f == Q8) dec_launch<Q8>(X, O, df, db, T);
            else if (f == Q4) dec_launch<Q4>(X, O, df, db, T); else dec_launch<EXP3>(X, O, df, db, T);
            CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hf.data(), df, N * 4, cudaMemcpyDeviceToHost));
            if (f == EXP3) { CK(cudaMemcpy(hb.data(), db, N * 2, cudaMemcpyDeviceToHost)); if (memcmp(hb.data(), src.data(), N * 2)) exact = false; }
            double num = 0, den = 0; for (size_t i = 0; i < N; i++) { double a = h2f_bits(src[i]), e = hf[i] - a; num += e * e; den += a * a; }
            maxrel[f] = std::max(maxrel[f], sqrt(num / den));
        }
        CK(cudaFree(df)); CK(cudaFree(db));
        float *q, *sc, *p, *out; CK(cudaMalloc(&q, HQ * D * 4)); CK(cudaMalloc(&sc, (size_t)HQ * T * 4)); CK(cudaMalloc(&p, (size_t)HQ * T * 4)); CK(cudaMalloc(&out, HQ * D * 4));
        { std::vector<float> hq(HQ * D), hp((size_t)HQ * T); srand(4242); for (auto &v : hq) v = (rand() / (float)RAND_MAX - 0.5f); for (auto &v : hp) v = 1.f / T;
          CK(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(p, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice)); }
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int f = 0; f < 4; f++) for (int w = 0; w < 3; w++) run(f, b[f], q, sc, p, out, T);   // warm-up
        CK(cudaDeviceSynchronize());
        std::vector<float> ms[4];
        for (int r = 0; r < REPS; r++) {
            int order[4] = {0, 1, 2, 3}; if (r & 1) std::reverse(order, order + 4);           // ABBA
            for (int oi = 0; oi < 4; oi++) { int f = order[oi];
                CK(cudaEventRecord(e0)); for (int i = 0; i < ITER; i++) run(f, b[f], q, sc, p, out, T); CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1)); float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms[f].push_back(t / ITER); }
        }
        for (int f = 0; f < 4; f++) {
            auto v = ms[f]; std::sort(v.begin(), v.end()); float med = v[v.size() / 2];
            double bytes = (double)b[f].bytesK + b[f].bytesV + b[f].ovfBytes, f16bytes = (double)T * ROW * 2 * 2;
            printf("{\"T\":%d,\"format\":\"%s\",\"bytes\":%.0f,\"ratio_vs_f16\":%.4f,\"escapes\":%zu,\"median_ms\":%.5f,\"min_ms\":%.5f,\"max_ms\":%.5f,"
                   "\"GBps_stored\":%.1f,\"GBps_f16_equiv\":%.1f,\"rel_rms_err\":%.3e%s}\n",
                   T, NAMES[f], bytes, bytes / f16bytes, b[f].escapes, med, v.front(), v.back(), bytes / med / 1e6, f16bytes / med / 1e6, maxrel[f],
                   f == EXP3 ? (exact ? ",\"bit_exact\":true" : ",\"bit_exact\":false") : "");
        }
        fflush(stdout);
        for (int f = 0; f < 4; f++) { cudaFree(b[f].K); cudaFree(b[f].V); cudaFree(b[f].ovfK); cudaFree(b[f].ovfV); }
        cudaFree(q); cudaFree(sc); cudaFree(p); cudaFree(out);
        if (!exact) { fprintf(stderr, "EXP3 NOT BIT-EXACT\n"); return 3; }
    }
    return 0;
}
