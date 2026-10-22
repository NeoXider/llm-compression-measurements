// PoC-A: does a lazily mapped KV pool make process VRAM follow the filled tokens instead of -c?
// Model geometry: Qwen3-4B (36 layers, 8 KV heads x 128, F16 KV) -> 147456 B per token, ctx 32768.
// Arms (each measured with DXGI QueryVideoMemoryInfo, calling process, LOCAL segment, on the adapter
// whose LUID matches CUDA device 0):
//   eager   : cudaMalloc of the full 32768-token KV, touched with cudaMemset (what llama.cpp does today)
//   lazy1   : one shared pool: reserve VA for 32768 tokens, map 2 MiB physical chunks only as tokens fill
//   lazy72  : 72 separate pools (K and V per layer), each grown in 2 MiB chunks (granularity cost)
// Fill levels: 1024, 4096, 16384, 32768 tokens; every mapped byte is written by cudaMemset.
#include <cuda.h>
#include <cuda_runtime.h>
#include <windows.h>
#include <dxgi1_4.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define CU(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); fprintf(stderr, "CU %s line %d\n", s, __LINE__); exit(1); } } while (0)

static IDXGIAdapter3 *g_adapter = nullptr;
static double local_mib() {
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (FAILED(g_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) { fprintf(stderr, "DXGI query failed\n"); exit(1); }
    return info.CurrentUsage / 1048576.0;
}

int main() {
    const size_t TOK = 147456, CTX = 32768, TENSORS = 72, PER_TENSOR_TOK = TOK / TENSORS;   // 2048 B per token per tensor
    CK(cudaFree(0));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    LUID luid; memcpy(&luid, p.luid, sizeof(luid));
    IDXGIFactory4 *fac = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void **)&fac))) { fprintf(stderr, "no DXGI factory\n"); return 1; }
    if (FAILED(fac->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), (void **)&g_adapter))) { fprintf(stderr, "no adapter for CUDA LUID\n"); return 1; }
    CUdevice dev; CU(cuDeviceGet(&dev, 0));
    CUmemAllocationProp prop{}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
    size_t gran = 0; CU(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CUmemAccessDesc acc{}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    const size_t fills[] = {1024, 4096, 16384, 32768};
    double base = local_mib();
    printf("{\"device\":\"%s\",\"granularity_bytes\":%zu,\"baseline_local_mib\":%.1f,\"kv_bytes_per_token\":%zu,\"ctx\":%zu}\n", p.name, gran, base, TOK, CTX);

    // eager
    { void *d; size_t bytes = TOK * CTX; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 0, bytes)); CK(cudaDeviceSynchronize());
      printf("{\"arm\":\"eager\",\"filled_tokens\":\"any\",\"kv_bytes_needed_mib\":%.1f,\"local_mib\":%.1f,\"delta_mib\":%.1f}\n", bytes / 1048576.0, local_mib(), local_mib() - base);
      CK(cudaFree(d)); CK(cudaDeviceSynchronize()); }

    // lazy pools: npools pools; each pool holds tok_bytes per token
    for (int arm = 0; arm < 2; arm++) {
        size_t npools = arm == 0 ? 1 : TENSORS, tok_bytes = arm == 0 ? TOK : PER_TENSOR_TOK;
        size_t reserve = ((tok_bytes * CTX + gran - 1) / gran) * gran;
        std::vector<CUdeviceptr> va(npools); std::vector<size_t> mapped(npools, 0);
        std::vector<std::vector<CUmemGenericAllocationHandle>> handles(npools);
        for (size_t i = 0; i < npools; i++) CU(cuMemAddressReserve(&va[i], reserve, 0, 0, 0));
        double after_reserve = local_mib();
        printf("{\"arm\":\"%s\",\"filled_tokens\":0,\"local_mib\":%.1f,\"delta_mib\":%.1f,\"note\":\"VA reserved only\"}\n", arm ? "lazy72" : "lazy1", after_reserve, after_reserve - base);
        for (size_t f : fills) {
            size_t need = ((tok_bytes * f + gran - 1) / gran) * gran;
            for (size_t i = 0; i < npools; i++) {
                while (mapped[i] < need) {
                    CUmemGenericAllocationHandle h; CU(cuMemCreate(&h, gran, &prop, 0));
                    CU(cuMemMap(va[i] + mapped[i], gran, 0, h, 0)); CU(cuMemSetAccess(va[i] + mapped[i], gran, &acc, 1));
                    handles[i].push_back(h); mapped[i] += gran;
                }
                CK(cudaMemset((void *)va[i], 0, tok_bytes * f));
            }
            CK(cudaDeviceSynchronize());
            double now = local_mib(); size_t tot = 0; for (auto m : mapped) tot += m;
            printf("{\"arm\":\"%s\",\"filled_tokens\":%zu,\"kv_bytes_needed_mib\":%.1f,\"mapped_mib\":%.1f,\"local_mib\":%.1f,\"delta_mib\":%.1f}\n",
                   arm ? "lazy72" : "lazy1", f, tok_bytes * f * npools / 1048576.0, tot / 1048576.0, now, now - base);
        }
        for (size_t i = 0; i < npools; i++) { CU(cuMemUnmap(va[i], mapped[i])); for (auto h : handles[i]) CU(cuMemRelease(h)); CU(cuMemAddressFree(va[i], reserve)); }
        CK(cudaDeviceSynchronize());
    }
    printf("{\"end_local_mib\":%.1f}\n", local_mib());
    return 0;
}
