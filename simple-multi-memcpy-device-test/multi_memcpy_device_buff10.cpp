// C++ standard headers
#include <cassert>
#include <iostream>
#include <vector>
#include <iomanip>  // For std::setprecision

// HIP headers
#include <hip/hip_runtime.h>

// Local headers
#include "../check_hip.h"

// Here you can set the device ID
#define MYDEVICE 0

#define N 4096  // 4096 elements
#define NSTEP 10000

// HIP kernel to add 10 arrays element-wise
__global__ void add_arrays(float *a1, float *a2, float *a3, float *a4, float *a5,
                           float *a6, float *a7, float *a8, float *a9, float *a10,
                           float *result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result[i] = a1[i] + a2[i] + a3[i] + a4[i] + a5[i]
                  + a6[i] + a7[i] + a8[i] + a9[i] + a10[i];
    }
}

// Program main
int main()
{
    // Choose one HIP device
    HIP_CHECK(hipSetDevice(MYDEVICE));

    // Create a HIP stream to execute asynchronous operations on this device
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    size_t memSize = N * sizeof(float);
    
    // Allocate pinned host memory for h_a and h_result
    float* h_a;
    float* h_result;
    HIP_CHECK(hipHostMalloc(&h_a, memSize));
    HIP_CHECK(hipHostMalloc(&h_result, memSize));

    // Initialize h_a
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
    }

    // Pointers for device memory
    float *d_result;
    float *d_a1, *d_a2, *d_a3, *d_a4, *d_a5, *d_a6, *d_a7, *d_a8, *d_a9, *d_a10;

    // Allocate the device memory
    HIP_CHECK(hipMallocAsync(&d_a1, memSize, stream));
    HIP_CHECK(hipMallocAsync(&d_a2, memSize, stream));
    HIP_CHECK(hipMallocAsync(&d_a3, memSize, stream));
    HIP_CHECK(hipMallocAsync(&d_a4, memSize, stream));
    HIP_CHECK(hipMallocAsync(&d_a5, memSize, stream));
    HIP_CHECK(hipMallocAsync(&d_a6, memSize, stream));
    HIP_CHECK(hipMallocAsync(&d_a7, memSize, stream));
    HIP_CHECK(hipMallocAsync(&d_a8, memSize, stream));
    HIP_CHECK(hipMallocAsync(&d_a9, memSize, stream));
    HIP_CHECK(hipMallocAsync(&d_a10, memSize, stream));
    HIP_CHECK(hipMallocAsync(&d_result, memSize, stream)); // Allocate device memory for result

    // Initialize d_result to zero
    HIP_CHECK(hipMemsetAsync(d_result, 0, memSize, stream));

    // Set up grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Set Timer for first run
    hipEvent_t firstCreateStart, firstCreateStop;
    float firstTime = 0.0f;
    HIP_CHECK(hipEventCreate(&firstCreateStart));
    HIP_CHECK(hipEventCreate(&firstCreateStop));

    // Host to device memory copy
    HIP_CHECK(hipMemcpyAsync(d_a1, h_a, memSize, hipMemcpyHostToDevice, stream));

    // Start Timer
    HIP_CHECK(hipEventRecord(firstCreateStart, stream));

    // Reset d_result to zero
    HIP_CHECK(hipMemsetAsync(d_result, 0, memSize, stream));

    // Device to device memory copies
    HIP_CHECK(hipMemcpyAsync(d_a2, d_a1, memSize, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a3, d_a2, memSize, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a4, d_a3, memSize, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a5, d_a4, memSize, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a6, d_a5, memSize, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a7, d_a6, memSize, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a8, d_a7, memSize, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a9, d_a8, memSize, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a10, d_a9, memSize, hipMemcpyDeviceToDevice, stream));

    // Single kernel launch after all memcpys
    hipLaunchKernelGGL(add_arrays, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream,
                       d_a1, d_a2, d_a3, d_a4, d_a5,
                       d_a6, d_a7, d_a8, d_a9, d_a10,
                       d_result);
    HIP_CHECK(hipGetLastError());  // Check for kernel launch errors

    // Device to host memory copy
    HIP_CHECK(hipMemcpyAsync(h_result, d_result, memSize, hipMemcpyDeviceToHost, stream));

    // Wait for the execution to finish
    HIP_CHECK(hipStreamSynchronize(stream));

    // End Timer
    HIP_CHECK(hipEventRecord(firstCreateStop, stream));
    HIP_CHECK(hipEventSynchronize(firstCreateStop));
    HIP_CHECK(hipEventElapsedTime(&firstTime, firstCreateStart, firstCreateStop));

    // Measure execution time
    hipEvent_t execStart, execStop;
    HIP_CHECK(hipEventCreate(&execStart));
    HIP_CHECK(hipEventCreate(&execStop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    int skipBy = 100;
    // Variables for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    for (int istep = 0; istep < NSTEP - 1; istep++) {
        // Start Timer for each run
        HIP_CHECK(hipEventRecord(execStart, stream));

        // Reset d_result to zero
        HIP_CHECK(hipMemsetAsync(d_result, 0, memSize, stream));

        // Device to device memory copies
        HIP_CHECK(hipMemcpyAsync(d_a2, d_a1, memSize, hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a3, d_a2, memSize, hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a4, d_a3, memSize, hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a5, d_a4, memSize, hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a6, d_a5, memSize, hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a7, d_a6, memSize, hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a8, d_a7, memSize, hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a9, d_a8, memSize, hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a10, d_a9, memSize, hipMemcpyDeviceToDevice, stream));

        // Single kernel launch after all memcpys
        hipLaunchKernelGGL(add_arrays, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream,
                           d_a1, d_a2, d_a3, d_a4, d_a5,
                           d_a6, d_a7, d_a8, d_a9, d_a10,
                           d_result);
        HIP_CHECK(hipGetLastError());  // Check for kernel launch errors

        // Device to host memory copy
        HIP_CHECK(hipMemcpyAsync(h_result, d_result, memSize, hipMemcpyDeviceToHost, stream));

        // Wait for the execution to finish
        HIP_CHECK(hipStreamSynchronize(stream));

        // End Timer for each run
        HIP_CHECK(hipEventRecord(execStop, stream));
        HIP_CHECK(hipEventSynchronize(execStop));
        HIP_CHECK(hipEventElapsedTime(&elapsedTime, execStart, execStop));

        // Time calculations
        if (istep >= skipBy) {
            totalTime += elapsedTime;

            // Welford's algorithm for calculating mean and variance
            count++;
            double delta = elapsedTime - mean;
            mean += delta / count;
            double delta2 = elapsedTime - mean;
            M2 += delta * delta2;

            if (elapsedTime > upperTime) {
                upperTime = elapsedTime;
            }
            if (elapsedTime < lowerTime) {
                lowerTime = elapsedTime;
            }
            if (istep == skipBy) {
                lowerTime = elapsedTime;
            }
        }
    }

    // Calculate mean and standard deviation
    float meanTime = (totalTime + firstTime) / (NSTEP - skipBy);
    double varianceTime = 0.0;
    if (count > 1) {
        varianceTime = M2 / (count - 1);
    }
    // Ensure variance is not negative due to floating-point errors
    if (varianceTime < 0.0) {
        varianceTime = 0.0;
    }
    double stdDevTime = sqrt(varianceTime);

    // Print out the time statistics
    std::cout << "=======Setup=======" << std::endl;
    std::cout << "Iterations: " << NSTEP << std::endl;
    std::cout << "Skip By: " << skipBy << std::endl;
    std::cout << "Kernels: " << 1 << std::endl;
    std::cout << "Block Size: " << threadsPerBlock << std::endl;
    std::cout << "Grid Size: " << blocksPerGrid << std::endl;
    std::cout << "Array Size: " << N << std::endl;
    std::cout << "=======Results=======" << std::endl;
    std::cout << "First Run: " << firstTime << std::endl;
    std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without firstRun: " << (totalTime / (NSTEP - 1 - skipBy)) << " ms" << std::endl;
    std::cout << "Variance: " <<  varianceTime << " ms" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << " ms" << std::endl;
    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with firstRun: " << totalTime + firstTime << " ms" << std::endl;

    // **Print h_result before testing**
    std::cout << "h_result contents before verification:\n";
    std::cout << std::fixed << std::setprecision(2); // Set precision for floating-point output
    std::cout << "h_result[" << N - 1 << "] = " << h_result[N - 1] << "\n";

    // Verify the data on the host is correct
    for (int i = 0; i < N; ++i)
    {
        float expected = i * 10.0f; // Since each a_i contains i, and we sum over 10 arrays
        if (h_result[i] != expected) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected << ", got " << h_result[i] << std::endl;
            assert(false);
        }
    }

    // Destroy the HIP stream
    HIP_CHECK(hipEventDestroy(execStart));
    HIP_CHECK(hipEventDestroy(execStop));
    HIP_CHECK(hipEventDestroy(firstCreateStart));
    HIP_CHECK(hipEventDestroy(firstCreateStop));
    HIP_CHECK(hipStreamDestroy(stream));

    // Free pinned host memory
    HIP_CHECK(hipHostFree(h_a));
    HIP_CHECK(hipHostFree(h_result));

    // Free the device memory
    HIP_CHECK(hipFree(d_a1));
    HIP_CHECK(hipFree(d_a2));
    HIP_CHECK(hipFree(d_a3));
    HIP_CHECK(hipFree(d_a4));
    HIP_CHECK(hipFree(d_a5));
    HIP_CHECK(hipFree(d_a6));
    HIP_CHECK(hipFree(d_a7));
    HIP_CHECK(hipFree(d_a8));
    HIP_CHECK(hipFree(d_a9));
    HIP_CHECK(hipFree(d_a10));
    HIP_CHECK(hipFree(d_result));

    // No run-time errors
    std::cout << "Correct!" << std::endl;

    return 0;
}
