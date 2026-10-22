// PoC-C: read cost of BASE vs BASE + OUR COMPRESSION, same decode-attention-shaped kernel as PoC-B.
// Bases: F16, Q8_0, Q4_0 (llama.cpp layouts). Ours (generic packed row format, header 12 B per
// 128-value row: 4 x f16 block scales + u32 overflow start; then 128 x B-bit codes, padded to 4 B):
//   F16D4 (lossy over F16):  B=12, value bits = code << 4 (4 mantissa bits dropped)
//   Q8L7  (lossy over Q8_0): B=7,  code = round(q/2) in [-64,63], value = d * 2 * code
//   Q4E3  (lossless over Q4_0): B=3, codes 5..11 stored as 0..6, 7 = escape -> exact 4-bit code
//                                from the per-row overflow list (one byte per escape)
// Checks before timing: Q4E3 decodes to exactly the Q4_0 base values (bit-exact floats);
// lossy variants report relative RMS error vs F16 and vs their base.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
static const int HKV = 8, D = 128, HQ = 32, G = HQ / HKV, ROW = HKV * D;
enum Fmt { F16 = 0, F16D4, Q8, Q8L7, Q4, Q4E3, NF };
static const char *NAMES[] = {"f16", "f16+drop4 (ours, lossy)", "q8_0", "q8_0+7bit (ours, lossy)", "q4_0", "q4_0+esc3 (ours, lossless)"};
static const int BASE[] = {F16, F16, Q8, Q8, Q4, Q4};
__host__ __device__ constexpr int pbits(int f) { return f == F16D4 ? 12 : f == Q8L7 ? 7 : 3; }
__host__ __device__ constexpr int row_bytes(int f) {
    return f == F16 ? 256 : f == Q8 ? 136 : f == Q4 ? 72 : 12 + ((128 * pbits(f) / 8 + 3) / 4) * 4;
}
static float h2f(uint16_t b) { __half h; memcpy(&h, &b, 2); return __half2float(h); }
static uint16_t f2h(float f) { __half h = __float2half_rn(f); uint16_t b; memcpy(&b, &h, 2); return b; }

struct Four { float v[4]; int d[4]; };
__device__ __forceinline__ Four dec_base(int f, const uint8_t *row, int lane) {
    Four r;
    if (f == F16) {
        uint2 u = *reinterpret_cast<const uint2 *>(row + lane * 8);
        float2 a = __half22float2(*reinterpret_cast<__half2 *>(&u.x)), b = __half22float2(*reinterpret_cast<__half2 *>(&u.y));
        r.v[0] = a.x; r.v[1] = a.y; r.v[2] = b.x; r.v[3] = b.y; for (int k = 0; k < 4; k++) r.d[k] = lane * 4 + k;
    } else if (f == Q8) {
        const uint8_t *blk = row + (lane >> 3) * 34; int j = (lane & 7) * 4; float d = __half2float(*reinterpret_cast<const __half *>(blk));
        const uint16_t *q = reinterpret_cast<const uint16_t *>(blk + 2 + j); uint16_t a = q[0], b = q[1];
        r.v[0] = d * (int8_t)(a & 255); r.v[1] = d * (int8_t)(a >> 8); r.v[2] = d * (int8_t)(b & 255); r.v[3] = d * (int8_t)(b >> 8);
        for (int k = 0; k < 4; k++) r.d[k] = lane * 4 + k;
    } else {  // Q4 (llama.cpp nibble layout: low nibble j, high nibble j+16)
        int bi = lane >> 3, j = (lane & 7) * 2; const uint8_t *blk = row + bi * 18; float d = __half2float(*reinterpret_cast<const __half *>(blk));
        uint16_t q = *reinterpret_cast<const uint16_t *>(blk + 2 + j); uint8_t q0 = q & 255, q1 = q >> 8;
        r.v[0] = d * ((q0 & 15) - 8); r.v[1] = d * ((q1 & 15) - 8); r.v[2] = d * ((q0 >> 4) - 8); r.v[3] = d * ((q1 >> 4) - 8);
        int b = bi * 32; r.d[0] = b + j; r.d[1] = b + j + 1; r.d[2] = b + j + 16; r.d[3] = b + j + 17;
    }
    return r;
}
// packed: lane holds values 4*lane..4*lane+3 (natural dim order)
template <int F> __device__ __forceinline__ Four dec_packed(const uint8_t *row, const uint8_t *ovf, int lane) {
    constexpr int B = pbits(F);
    const uint8_t *pay = row + 12; int bit0 = lane * 4 * B, byte0 = bit0 >> 3, sh = bit0 & 7, nbytes = (sh + 4 * B + 7) >> 3;
    uint64_t w = 0; for (int k = 0; k < nbytes; k++) w |= (uint64_t)pay[byte0 + k] << (8 * k);
    w >>= sh;
    Four r; uint32_t c[4]; for (int k = 0; k < 4; k++) { c[k] = (uint32_t)(w >> (B * k)) & ((1u << B) - 1); r.d[k] = lane * 4 + k; }
    if (F == F16D4) {
        for (int k = 0; k < 4; k++) r.v[k] = __half2float(__ushort_as_half((unsigned short)(c[k] << 4)));
    } else {
        float d = __half2float(*reinterpret_cast<const __half *>(row + 2 * (lane >> 3)));
        if (F == Q8L7) { for (int k = 0; k < 4; k++) { int v = (int)c[k] - ((c[k] & 64) ? 128 : 0); r.v[k] = d * 2.f * v; } }
        else {  // Q4E3
            int esc = 0; for (int k = 0; k < 4; k++) esc += c[k] == 7;
            int before = 0;
            if (__ballot_sync(0xffffffffu, esc != 0)) {
                int incl = esc; for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += n; }
                before = incl - esc;
            }
            uint32_t start = *reinterpret_cast<const uint32_t *>(row + 8);
            for (int k = 0; k < 4; k++) { int code = c[k] == 7 ? ovf[start + before++] : (int)c[k] + 5; r.v[k] = d * (code - 8); }
        }
    }
    return r;
}
template <int F> __device__ __forceinline__ Four dec(const uint8_t *row, const uint8_t *ovf, int lane) {
    if (F == F16 || F == Q8 || F == Q4) return dec_base(F, row, lane);
    return dec_packed<F>(row, ovf, lane);
}
template <int F> __global__ void kpass(const uint8_t *K, const uint8_t *ovf, const float *q, float *scores, int T) {
    __shared__ float qs[HQ * D];
    for (int i = threadIdx.x; i < HQ * D; i += blockDim.x) qs[i] = q[i];
    __syncthreads();
    int lane = threadIdx.x & 31, wpb = blockDim.x >> 5, warp = blockIdx.x * wpb + (threadIdx.x >> 5), nw = gridDim.x * wpb;
    for (int r = warp; r < T * HKV; r += nw) {
        int t = r / HKV, h = r % HKV; Four x = dec<F>(K + (size_t)r * row_bytes(F), ovf, lane);
        for (int g = 0; g < G; g++) {
            const float *qq = qs + (h * G + g) * D; float s = 0; for (int k = 0; k < 4; k++) s += qq[x.d[k]] * x.v[k];
            for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (lane == 0) scores[(size_t)(h * G + g) * T + t] = s;
        }
    }
}
template <int F, int TPW> __global__ void vpass(const uint8_t *V, const uint8_t *ovf, const float *p, float *out, int T) {
    __shared__ float red[G * D];
    for (int i = threadIdx.x; i < G * D; i += blockDim.x) red[i] = 0;
    __syncthreads();
    int lane = threadIdx.x & 31, h = blockIdx.y, t0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * TPW;
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
template <int F> __global__ void decode_all(const uint8_t *X, const uint8_t *ovf, float *out, int T) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; if (warp >= T * HKV) return;
    Four x = dec<F>(X + (size_t)warp * row_bytes(F), ovf, lane);
    for (int k = 0; k < 4; k++) out[(size_t)warp * D + x.d[k]] = x.v[k];
}

// ------------------------------------------------------------------ host encoders
static void enc_q8(const uint16_t *x, uint8_t *o, float *dq, int8_t *codes) {
    float v[32], am = 0; for (int i = 0; i < 32; i++) { v[i] = h2f(x[i]); am = std::max(am, fabsf(v[i])); }
    float d = am / 127.f, id = d ? 1.f / d : 0.f; uint16_t db = f2h(d); memcpy(o, &db, 2); *dq = h2f(db);
    for (int i = 0; i < 32; i++) { codes[i] = (int8_t)lrintf(v[i] * id); o[2 + i] = (uint8_t)codes[i]; }
}
static void enc_q4(const uint16_t *x, uint8_t *o, uint8_t *codes) {
    float v[32], am = 0, mx = 0; for (int i = 0; i < 32; i++) { v[i] = h2f(x[i]); if (fabsf(v[i]) > am) { am = fabsf(v[i]); mx = v[i]; } }
    float d = mx / -8.f, id = d ? 1.f / d : 0.f; uint16_t db = f2h(d); memcpy(o, &db, 2);
    for (int i = 0; i < 32; i++) codes[i] = (uint8_t)std::max(0, std::min(15, (int)(v[i] * id + 8.5f)));
    for (int j = 0; j < 16; j++) o[2 + j] = codes[j] | (codes[j + 16] << 4);
}
static void put_bits(uint8_t *pay, int idx, int B, uint32_t code) { size_t bit = (size_t)idx * B; for (int k = 0; k < B; k++) if (code >> k & 1) pay[(bit + k) >> 3] |= (uint8_t)(1u << ((bit + k) & 7)); }

static void encode(int f, const std::vector<uint16_t> &x, int T, std::vector<uint8_t> &enc, std::vector<uint8_t> &ovf) {
    enc.assign((size_t)T * HKV * row_bytes(f), 0); ovf.clear();
    for (size_t r = 0; r < (size_t)T * HKV; r++) {
        const uint16_t *src = x.data() + r * D; uint8_t *dst = enc.data() + r * row_bytes(f);
        if (f == F16) { memcpy(dst, src, 256); continue; }
        if (f == Q8) { float dq; int8_t c[32]; for (int b = 0; b < 4; b++) enc_q8(src + b * 32, dst + b * 34, &dq, c); continue; }
        if (f == Q4) { uint8_t c[32]; for (int b = 0; b < 4; b++) enc_q4(src + b * 32, dst + b * 18, c); continue; }
        uint8_t *pay = dst + 12; uint32_t start = (uint32_t)ovf.size(); memcpy(dst + 8, &start, 4);
        if (f == F16D4) { for (int i = 0; i < 128; i++) put_bits(pay, i, 12, src[i] >> 4); continue; }
        for (int b = 0; b < 4; b++) {
            if (f == Q8L7) {
                uint8_t tmp[34]; float dq; int8_t c[32]; enc_q8(src + b * 32, tmp, &dq, c); memcpy(dst + 2 * b, tmp, 2);
                for (int i = 0; i < 32; i++) { int v = (int)lrintf(c[i] / 2.f); v = std::max(-64, std::min(63, v)); put_bits(pay, b * 32 + i, 7, (uint32_t)(v & 127)); }
            } else {  // Q4E3
                uint8_t tmp[18], c[32]; enc_q4(src + b * 32, tmp, c); memcpy(dst + 2 * b, tmp, 2);
                for (int i = 0; i < 32; i++) { uint32_t s; if (c[i] >= 5 && c[i] <= 11) s = c[i] - 5; else { s = 7; ovf.push_back(c[i]); } put_bits(pay, b * 32 + i, 3, s); }
            }
        }
    }
    if (ovf.empty()) ovf.push_back(0);
}

static std::vector<uint16_t> load_tokens(const char *path, int T, bool wantV) {
    FILE *fp = fopen(path, "rb"); if (!fp) { perror("state"); exit(1); }
    uint32_t hdr[4]; fread(hdr, 4, 4, fp); uint32_t n = hdr[3]; fseek(fp, 16 + 12L * n, SEEK_SET); uint32_t vl[2]; fread(vl, 4, 2, fp);
    int L = vl[1]; size_t per = (size_t)n * ROW; std::vector<uint16_t> all((size_t)L * per);
    for (int c = 0; c < 2; c++) for (int l = 0; l < L; l++) { int32_t dt; uint64_t rb; fread(&dt, 4, 1, fp); fread(&rb, 8, 1, fp);
        if ((c == 1) == wantV) fread(all.data() + l * per, 2, per, fp); else fseek(fp, (long)(per * 2), SEEK_CUR); }
    fclose(fp); std::vector<uint16_t> out((size_t)T * ROW); size_t tot = (size_t)L * n;
    for (int t = 0; t < T; t++) memcpy(out.data() + (size_t)t * ROW, all.data() + (size_t)(t % tot) * ROW, ROW * 2);
    return out;
}
template <int F> static void run1(const uint8_t *K, const uint8_t *oK, const uint8_t *V, const uint8_t *oV, const float *q, float *sc, const float *p, float *out, int T) {
    kpass<F><<<82 * 8, 256>>>(K, oK, q, sc, T); dim3 g((T + 127) / 128, HKV); vpass<F, 16><<<g, 256>>>(V, oV, p, out, T);
}
static void run(int f, const uint8_t *K, const uint8_t *oK, const uint8_t *V, const uint8_t *oV, const float *q, float *sc, const float *p, float *out, int T) {
    switch (f) { case F16: run1<F16>(K, oK, V, oV, q, sc, p, out, T); break; case F16D4: run1<F16D4>(K, oK, V, oV, q, sc, p, out, T); break;
        case Q8: run1<Q8>(K, oK, V, oV, q, sc, p, out, T); break; case Q8L7: run1<Q8L7>(K, oK, V, oV, q, sc, p, out, T); break;
        case Q4: run1<Q4>(K, oK, V, oV, q, sc, p, out, T); break; default: run1<Q4E3>(K, oK, V, oV, q, sc, p, out, T); }
}
static void decode(int f, const uint8_t *X, const uint8_t *o, float *out, int T) {
    int w = T * HKV, b = (w * 32 + 255) / 256;
    switch (f) { case F16: decode_all<F16><<<b, 256>>>(X, o, out, T); break; case F16D4: decode_all<F16D4><<<b, 256>>>(X, o, out, T); break;
        case Q8: decode_all<Q8><<<b, 256>>>(X, o, out, T); break; case Q8L7: decode_all<Q8L7><<<b, 256>>>(X, o, out, T); break;
        case Q4: decode_all<Q4><<<b, 256>>>(X, o, out, T); break; default: decode_all<Q4E3><<<b, 256>>>(X, o, out, T); }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: kvread_base_plus state.bin [T...]\n"); return 2; }
    int REPS = getenv("REPS") ? atoi(getenv("REPS")) : 12, ITER = getenv("ITER") ? atoi(getenv("ITER")) : 10;
    std::vector<int> Ts; for (int i = 2; i < argc; i++) Ts.push_back(atoi(argv[i])); if (Ts.empty()) Ts = {4096, 32768, 122880};
    for (int T : Ts) {
        auto hk = load_tokens(argv[1], T, false), hv = load_tokens(argv[1], T, true);
        uint8_t *dK[NF], *dV[NF], *oK[NF], *oV[NF]; double bytes[NF], escRate[NF] = {0};
        for (int f = 0; f < NF; f++) {
            std::vector<uint8_t> eb, o; size_t esc = 0;
            for (int c = 0; c < 2; c++) {
                encode(f, c ? hv : hk, T, eb, o); esc += (f == Q4E3) ? o.size() : 0;
                uint8_t *d, *od; CK(cudaMalloc(&d, eb.size())); CK(cudaMemcpy(d, eb.data(), eb.size(), cudaMemcpyHostToDevice));
                CK(cudaMalloc(&od, o.size())); CK(cudaMemcpy(od, o.data(), o.size(), cudaMemcpyHostToDevice));
                (c ? dV : dK)[f] = d; (c ? oV : oK)[f] = od; bytes[f] = (c ? bytes[f] : 0) + eb.size() + (f == Q4E3 ? o.size() : 0);
            }
            escRate[f] = f == Q4E3 ? (double)esc / ((double)T * ROW * 2) : 0;
        }
        size_t N = (size_t)T * ROW; float *df; CK(cudaMalloc(&df, N * 4)); std::vector<std::vector<float>> dec(NF, std::vector<float>(N));
        double err16[NF], errb[NF]; bool exact = true;
        for (int c = 0; c < 2; c++) {
            const auto &src = c ? hv : hk;
            for (int f = 0; f < NF; f++) { decode(f, c ? dV[f] : dK[f], c ? oV[f] : oK[f], df, T); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(dec[f].data(), df, N * 4, cudaMemcpyDeviceToHost)); }
            if (memcmp(dec[Q4E3].data(), dec[Q4].data(), N * 4)) exact = false;
            for (int f = 0; f < NF; f++) {
                double n16 = 0, nb = 0, den = 0, denb = 0;
                for (size_t i = 0; i < N; i++) { double a = h2f(src[i]), e = dec[f][i] - a, eb = dec[f][i] - dec[BASE[f]][i]; n16 += e * e; den += a * a; nb += eb * eb; denb += (double)dec[BASE[f]][i] * dec[BASE[f]][i]; }
                err16[f] = (c ? (err16[f] + sqrt(n16 / den)) / 2 : sqrt(n16 / den)); errb[f] = (c ? (errb[f] + sqrt(nb / denb)) / 2 : sqrt(nb / denb));
            }
        }
        cudaFree(df);
        float *q, *sc, *p, *out; CK(cudaMalloc(&q, HQ * D * 4)); CK(cudaMalloc(&sc, (size_t)HQ * T * 4)); CK(cudaMalloc(&p, (size_t)HQ * T * 4)); CK(cudaMalloc(&out, HQ * D * 4));
        { std::vector<float> hq(HQ * D), hp((size_t)HQ * T, 1.f / T); srand(4242); for (auto &v : hq) v = rand() / (float)RAND_MAX - 0.5f;
          CK(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(p, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice)); }
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int f = 0; f < NF; f++) for (int w = 0; w < 3; w++) run(f, dK[f], oK[f], dV[f], oV[f], q, sc, p, out, T);
        CK(cudaDeviceSynchronize());
        std::vector<float> ms[NF];
        for (int r = 0; r < REPS; r++) {
            int ord[NF]; for (int i = 0; i < NF; i++) ord[i] = (r & 1) ? NF - 1 - i : i;
            for (int oi = 0; oi < NF; oi++) { int f = ord[oi]; CK(cudaEventRecord(e0)); for (int i = 0; i < ITER; i++) run(f, dK[f], oK[f], dV[f], oV[f], q, sc, p, out, T);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms[f].push_back(t / ITER); }
        }
        float med[NF]; for (int f = 0; f < NF; f++) { auto v = ms[f]; std::sort(v.begin(), v.end()); med[f] = v[v.size() / 2]; }
        for (int f = 0; f < NF; f++)
            printf("{\"T\":%d,\"format\":\"%s\",\"base\":\"%s\",\"bytes\":%.0f,\"size_vs_base\":%.4f,\"median_ms\":%.5f,\"speed_vs_base\":%.3f,"
                   "\"rel_err_vs_f16\":%.3e,\"rel_err_vs_base\":%.3e,\"escape_rate\":%.4f%s}\n", T, NAMES[f], NAMES[BASE[f]], bytes[f], bytes[f] / bytes[BASE[f]],
                   med[f], med[BASE[f]] / med[f], err16[f], errb[f], escRate[f], f == Q4E3 ? (exact ? ",\"exact_vs_base\":true" : ",\"exact_vs_base\":false") : "");
        fflush(stdout);
        for (int f = 0; f < NF; f++) { cudaFree(dK[f]); cudaFree(dV[f]); cudaFree(oK[f]); cudaFree(oV[f]); }
        cudaFree(q); cudaFree(sc); cudaFree(p); cudaFree(out);
        if (!exact) { fprintf(stderr, "Q4E3 not exact\n"); return 3; }
    }
    return 0;
}
