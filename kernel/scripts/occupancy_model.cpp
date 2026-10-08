// Offline resource model only: this never launches a CUDA kernel.
// Build with a host C++ compiler and -I "$CUDA_HOME/include".
#include <cuda_occupancy.h>
#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "usage: occupancy_model sm threads registers static_shared dynamic_shared\n";
        return 2;
    }
    int sm = std::atoi(argv[1]);
    int threads = std::atoi(argv[2]);
    if (sm != 75 && sm != 86) return 2;
    cudaOccDeviceProp device;
    device.computeMajor = sm / 10;
    device.computeMinor = sm % 10;
    device.maxThreadsPerBlock = 1024;
    device.maxThreadsPerMultiprocessor = sm == 75 ? 1024 : 1536;
    device.regsPerBlock = device.regsPerMultiprocessor = 65536;
    device.warpSize = 32;
    device.sharedMemPerBlock = 49152;
    device.sharedMemPerMultiprocessor = sm == 75 ? 65536 : 102400;
    device.sharedMemPerBlockOptin = sm == 75 ? 65536 : 101376;
    device.reservedSharedMemPerBlock = sm == 75 ? 0 : 1024;
    device.numSms = 1; // Per-SM calculation; board SM count is irrelevant here.
    cudaOccFuncAttributes function;
    function.maxThreadsPerBlock = 1024;
    function.numRegs = std::atoi(argv[3]);
    function.sharedSizeBytes = std::strtoull(argv[4], nullptr, 10);
    function.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
    function.maxDynamicSharedSizeBytes = device.sharedMemPerBlockOptin - function.sharedSizeBytes;
    cudaOccDeviceState state;
    state.carveoutConfig = 100; // Maximum-shared-memory theoretical ceiling.
    cudaOccResult result{};
    auto status = cudaOccMaxActiveBlocksPerMultiprocessor(
        &result, &device, &function, &state, threads,
        std::strtoull(argv[5], nullptr, 10));
    if (status != CUDA_OCC_SUCCESS) {
        std::cerr << "occupancy calculation failed: " << int(status) << '\n';
        return 1;
    }
    std::cout << "{\"ctas_per_sm\":" << result.activeBlocksPerMultiprocessor
              << ",\"warp_occupancy\":"
              << double(result.activeBlocksPerMultiprocessor * ((threads + 31) / 32)) /
                     (device.maxThreadsPerMultiprocessor / 32)
              << ",\"register_limit_ctas\":" << result.blockLimitRegs
              << ",\"shared_limit_ctas\":" << result.blockLimitSharedMem
              << ",\"warp_limit_ctas\":" << result.blockLimitWarps
              << ",\"allocated_registers_per_cta\":" << result.allocatedRegistersPerBlock
              << ",\"allocated_shared_per_cta\":" << result.allocatedSharedMemPerBlock
              << ",\"limiting_factors\":" << result.limitingFactors << "}\n";
}
