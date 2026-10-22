// PoC-D: lossless compression ON TOP OF BF16 weights, read inside a GEMV (y = W x), RTX 3090.
// Base: BF16 row-major W (rows x cols). Ours (BF16E3, lossless, fixed-rate, random access per 32):
//   per group of 32 weights: base exponent (8 bit, max exponent of the group) stored in a per-row header,
//   per weight 11 bits = sign(1) | 3-bit exponent offset (base - e; 7 = escape) | mantissa(7);
//   escaped exponents (full 8 bits) go to a per-row overflow list; the row header stores its start (u32).
// Row layout: [u32 overflow start][cols/32 base bytes][payload cols*11 bits, padded to 4 B].
// GEMV: one warp per row; each lane handles 4 consecutive weights per step (BF16: one 8-byte load).
// Checks: device decode == source bits for every weight. Timing: cudaEvent over ITER GEMVs, ABBA, REPS rounds.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float(((uint32_t) b) << 16); }

__global__ void gemv_bf16(const uint16_t *W, const float *x, float *y, int rows, int cols) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; if (warp >= rows) return;
    const uint16_t *w = W + (size_t) warp * cols; float s = 0;
    for (int c = lane * 4; c < cols; c += 128) {
        uint2 u = *reinterpret_cast<const uint2 *>(w + c);
        s += bf2f(u.x & 0xFFFF) * x[c] + bf2f(u.x >> 16) * x[c + 1] + bf2f(u.y & 0xFFFF) * x[c + 2] + bf2f(u.y >> 16) * x[c + 3];
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) y[warp] = s;
}

// decode 4 consecutive weights (global index c..c+3 of the row) -> bf16 bits
__device__ __forceinline__ void dec4(const uint8_t *row, const uint8_t *ovf, int cols, int c, int lane, uint16_t out[4], int &esc_before_lane_total) {
    const uint8_t *bases = row + 4, *pay = row + 4 + cols / 32;
    size_t bit0 = (size_t) c * 11; int byte0 = (int) (bit0 >> 3), sh = (int) (bit0 & 7);
    uint64_t w = 0; for (int k = 0; k < 7; k++) w |= (uint64_t) pay[byte0 + k] << (8 * k);
    w >>= sh;
    uint32_t base = bases[c / 32];
    uint32_t code[4]; int esc = 0;
    for (int k = 0; k < 4; k++) { code[k] = (uint32_t) (w >> (11 * k)) & 0x7FF; esc += ((code[k] >> 7) & 7) == 7; }
    // warp prefix of escapes within this step; plus running total from earlier steps
    int incl = esc; for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += n; }
    int before = esc_before_lane_total + incl - esc;
    int step_total = __shfl_sync(0xffffffffu, incl, 31);
    uint32_t start = *reinterpret_cast<const uint32_t *>(row);
    for (int k = 0; k < 4; k++) {
        uint32_t cc = code[k], off = (cc >> 7) & 7, e = off == 7 ? ovf[start + before++] : base - off;
        out[k] = (uint16_t) (((cc >> 10) << 15) | (e << 7) | (cc & 127));
    }
    esc_before_lane_total += step_total;
}

__global__ void gemv_e3(const uint8_t *P, const uint8_t *ovf, const float *x, float *y, int rows, int cols, size_t rowb) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; if (warp >= rows) return;
    const uint8_t *row = P + (size_t) warp * rowb; float s = 0; int tot = 0;
    for (int c = lane * 4, c0 = 0; c0 < cols; c += 128, c0 += 128) {
        uint16_t b[4]; dec4(row, ovf, cols, c, lane, b, tot);
        s += bf2f(b[0]) * x[c] + bf2f(b[1]) * x[c + 1] + bf2f(b[2]) * x[c + 2] + bf2f(b[3]) * x[c + 3];
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) y[warp] = s;
}
__global__ void decode_e3(const uint8_t *P, const uint8_t *ovf, uint16_t *out, int rows, int cols, size_t rowb) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; if (warp >= rows) return;
    const uint8_t *row = P + (size_t) warp * rowb; int tot = 0;
    for (int c = lane * 4, c0 = 0; c0 < cols; c += 128, c0 += 128) { uint16_t b[4]; dec4(row, ovf, cols, c, lane, b, tot); for (int k = 0; k < 4; k++) out[(size_t) warp * cols + c + k] = b[k]; }
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: bf16_gemv_bench rows.bin rows cols\n"); return 2; }
    int rows = atoi(argv[2]), cols = atoi(argv[3]); int REPS = getenv("REPS") ? atoi(getenv("REPS")) : 12, ITER = getenv("ITER") ? atoi(getenv("ITER")) : 20;
    size_t n = (size_t) rows * cols; std::vector<uint16_t> W(n);
    FILE *f = fopen(argv[1], "rb"); if (!f || fread(W.data(), 2, n, f) != n) { fprintf(stderr, "read failed\n"); return 2; } fclose(f);
    size_t rowb = 4 + cols / 32 + ((size_t) cols * 11 / 8 + 7) / 4 * 4;  // 7 B slack for 64-bit window reads
    std::vector<uint8_t> P((size_t) rows * rowb, 0), ovf; size_t esc = 0;
    for (int r = 0; r < rows; r++) {
        uint8_t *row = P.data() + (size_t) r * rowb; uint32_t start = (uint32_t) ovf.size(); memcpy(row, &start, 4);
        uint8_t *bases = row + 4, *pay = row + 4 + cols / 32; const uint16_t *src = W.data() + (size_t) r * cols;
        for (int g = 0; g < cols / 32; g++) { int b = 0; for (int i = 0; i < 32; i++) b = std::max(b, (int) ((src[g * 32 + i] >> 7) & 255)); bases[g] = (uint8_t) b; }
        for (int c = 0; c < cols; c++) {
            uint16_t w = src[c]; int e = (w >> 7) & 255, off = bases[c / 32] - e; uint32_t code;
            if (off >= 7) { code = ((w >> 15) << 10) | (7u << 7) | (w & 127); ovf.push_back((uint8_t) e); esc++; }
            else code = ((w >> 15) << 10) | ((uint32_t) off << 7) | (w & 127);
            size_t bit = (size_t) c * 11; for (int k = 0; k < 11; k++) if (code >> k & 1) pay[(bit + k) >> 3] |= (uint8_t) (1u << ((bit + k) & 7));
        }
    }
    if (ovf.empty()) ovf.push_back(0);
    double bytes_base = n * 2.0, bytes_e3 = (double) rows * (4 + cols / 32 + (cols * 11 + 7) / 8) + ovf.size();
    uint16_t *dW; uint8_t *dP, *dO; float *dx, *dy; uint16_t *dD;
    CK(cudaMalloc(&dW, n * 2)); CK(cudaMemcpy(dW, W.data(), n * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dP, P.size())); CK(cudaMemcpy(dP, P.data(), P.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dO, ovf.size())); CK(cudaMemcpy(dO, ovf.data(), ovf.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx, cols * 4)); CK(cudaMalloc(&dy, rows * 4)); CK(cudaMalloc(&dD, n * 2));
    { std::vector<float> hx(cols); srand(7); for (auto &v : hx) v = rand() / (float) RAND_MAX - 0.5f; CK(cudaMemcpy(dx, hx.data(), cols * 4, cudaMemcpyHostToDevice)); }
    int blocks = (rows * 32 + 255) / 256;
    decode_e3<<<blocks, 256>>>(dP, dO, dD, rows, cols, rowb); CK(cudaDeviceSynchronize());
    std::vector<uint16_t> back(n); CK(cudaMemcpy(back.data(), dD, n * 2, cudaMemcpyDeviceToHost));
    bool exact = memcmp(back.data(), W.data(), n * 2) == 0;
    std::vector<float> y1(rows), y2(rows);
    gemv_bf16<<<blocks, 256>>>(dW, dx, dy, rows, cols); CK(cudaMemcpy(y1.data(), dy, rows * 4, cudaMemcpyDeviceToHost));
    gemv_e3<<<blocks, 256>>>(dP, dO, dx, dy, rows, cols, rowb); CK(cudaMemcpy(y2.data(), dy, rows * 4, cudaMemcpyDeviceToHost));
    bool same_y = memcmp(y1.data(), y2.data(), rows * 4) == 0;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < 5; i++) { gemv_bf16<<<blocks, 256>>>(dW, dx, dy, rows, cols); gemv_e3<<<blocks, 256>>>(dP, dO, dx, dy, rows, cols, rowb); }
    CK(cudaDeviceSynchronize());
    std::vector<float> t[2];
    for (int r = 0; r < REPS; r++) for (int k = 0; k < 2; k++) { int fmt = (r & 1) ? 1 - k : k;
        CK(cudaEventRecord(e0)); for (int i = 0; i < ITER; i++) { if (fmt == 0) gemv_bf16<<<blocks, 256>>>(dW, dx, dy, rows, cols); else gemv_e3<<<blocks, 256>>>(dP, dO, dx, dy, rows, cols, rowb); }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t[fmt].push_back(ms / ITER); }
    float med[2]; for (int k = 0; k < 2; k++) { auto v = t[k]; std::sort(v.begin(), v.end()); med[k] = v[v.size() / 2]; }
    printf("{\"rows\":%d,\"cols\":%d,\"escape_rate\":%.4f,\"bytes_bf16\":%.0f,\"bytes_ours\":%.0f,\"size_vs_base\":%.4f,"
           "\"gemv_ms_bf16\":%.5f,\"gemv_ms_ours\":%.5f,\"speed_vs_base\":%.3f,\"GBps_bf16\":%.1f,\"GBps_ours_stored\":%.1f,"
           "\"decode_bit_exact\":%s,\"gemv_output_identical\":%s}\n", rows, cols, (double) esc / n, bytes_base, bytes_e3, bytes_e3 / bytes_base,
           med[0], med[1], med[0] / med[1], bytes_base / med[0] / 1e6, bytes_e3 / med[1] / 1e6, exact ? "true" : "false", same_y ? "true" : "false");
    return exact ? 0 : 3;
}
