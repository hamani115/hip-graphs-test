// C++ standard headers
#include <cassert>
#include <iostream>
#include <vector>
#include <iomanip>  // For std::setprecision

// HIP headers
#include <hip/hip_runtime.h>

// Define error checking macro
#define HIP_CHECK(call)                                                           \
    do {                                                                          \
        hipError_t err = call;                                                    \
        if (err != hipSuccess) {                                                  \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,       \
                    hipGetErrorString(err));                                      \
            exit(err);                                                            \
        }                                                                         \
    } while (0)

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

#define NSTEP 10000

// Define the array size
const int N = 4096;
const int dimA = N;

// HIP kernel to add 10 arrays element-wise
__global__ void add_arrays(float *a1, float *a2, float *a3, float *a4, float *a5,
                           float *a6, float *a7, float *a8, float *a9, float *a10,
                           float *result, int N) {
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
    hipStream_t queue;
    HIP_CHECK(hipStreamCreate(&queue));

    size_t memSize = dimA * sizeof(float);
    // Allocate pinned host memory for h_a and h_result
    float* h_a;
    HIP_CHECK(hipHostMalloc(&h_a, memSize));

    float* h_result;
    HIP_CHECK(hipHostMalloc(&h_result, memSize));

    // Initialize h_a
    for (int i = 0; i < dimA; ++i) {
        h_a[i] = i;
    }

    // Pointers for device memory
    float *d_result;
    float *d_a1, *d_a2, *d_a3, *d_a4, *d_a5, *d_a6, *d_a7, *d_a8, *d_a9, *d_a10;

    // Part 1 of 5: allocate the device memory
    HIP_CHECK(hipMallocAsync(&d_a1, memSize, queue));
    HIP_CHECK(hipMallocAsync(&d_a2, memSize, queue));
    HIP_CHECK(hipMallocAsync(&d_a3, memSize, queue));
    HIP_CHECK(hipMallocAsync(&d_a4, memSize, queue));
    HIP_CHECK(hipMallocAsync(&d_a5, memSize, queue));
    HIP_CHECK(hipMallocAsync(&d_a6, memSize, queue));
    HIP_CHECK(hipMallocAsync(&d_a7, memSize, queue));
    HIP_CHECK(hipMallocAsync(&d_a8, memSize, queue));
    HIP_CHECK(hipMallocAsync(&d_a9, memSize, queue));
    HIP_CHECK(hipMallocAsync(&d_a10, memSize, queue));
    HIP_CHECK(hipMallocAsync(&d_result, memSize, queue)); // Allocate device memory for result

    // Initialize d_result to zero
    HIP_CHECK(hipMemsetAsync(d_result, 0, memSize, queue));

    // Set up grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create HIP events for timings
    hipEvent_t start, stop;
    float elapsedTime = 0.0f;
    float firstTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    int skipBy = 100;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Part 2 of 5: host to device memory copy
    HIP_CHECK(hipMemcpyAsync(d_a1, h_a, memSize, hipMemcpyHostToDevice, queue));

    // Start Timer
    HIP_CHECK(hipEventRecord(start, queue));

    // Reset d_result to zero
    HIP_CHECK(hipMemsetAsync(d_result, 0, memSize, queue));

    // Part 3 of 5: device to device memory copies
    HIP_CHECK(hipMemcpyAsync(d_a2, d_a1, memSize, hipMemcpyDeviceToDevice, queue));
    HIP_CHECK(hipMemcpyAsync(d_a3, d_a2, memSize, hipMemcpyDeviceToDevice, queue));
    HIP_CHECK(hipMemcpyAsync(d_a4, d_a3, memSize, hipMemcpyDeviceToDevice, queue));
    HIP_CHECK(hipMemcpyAsync(d_a5, d_a4, memSize, hipMemcpyDeviceToDevice, queue));
    HIP_CHECK(hipMemcpyAsync(d_a6, d_a5, memSize, hipMemcpyDeviceToDevice, queue));
    HIP_CHECK(hipMemcpyAsync(d_a7, d_a6, memSize, hipMemcpyDeviceToDevice, queue));
    HIP_CHECK(hipMemcpyAsync(d_a8, d_a7, memSize, hipMemcpyDeviceToDevice, queue));
    HIP_CHECK(hipMemcpyAsync(d_a9, d_a8, memSize, hipMemcpyDeviceToDevice, queue));
    HIP_CHECK(hipMemcpyAsync(d_a10, d_a9, memSize, hipMemcpyDeviceToDevice, queue));

    // Part 4 of 5: single kernel launch after all memcpys
    hipLaunchKernelGGL(add_arrays, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, queue,
                       d_a1, d_a2, d_a3, d_a4, d_a5,
                       d_a6, d_a7, d_a8, d_a9, d_a10,
                       d_result, N);

    // Part 5 of 5: device to host memory copy
    HIP_CHECK(hipMemcpyAsync(h_result, d_result, memSize, hipMemcpyDeviceToHost, queue));

    // Wait for the execution to finish
    HIP_CHECK(hipStreamSynchronize(queue));

    // End Timer
    HIP_CHECK(hipEventRecord(stop, queue));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&firstTime, start, stop));

    for (int istep = 0; istep < NSTEP-1; istep++) {
        // Start Timer
        HIP_CHECK(hipEventRecord(start, queue));

        // Reset d_result to zero
        HIP_CHECK(hipMemsetAsync(d_result, 0, memSize, queue));

        // Part 3 of 5: device to device memory copies
        HIP_CHECK(hipMemcpyAsync(d_a2, d_a1, memSize, hipMemcpyDeviceToDevice, queue));
        HIP_CHECK(hipMemcpyAsync(d_a3, d_a2, memSize, hipMemcpyDeviceToDevice, queue));
        HIP_CHECK(hipMemcpyAsync(d_a4, d_a3, memSize, hipMemcpyDeviceToDevice, queue));
        HIP_CHECK(hipMemcpyAsync(d_a5, d_a4, memSize, hipMemcpyDeviceToDevice, queue));
        HIP_CHECK(hipMemcpyAsync(d_a6, d_a5, memSize, hipMemcpyDeviceToDevice, queue));
        HIP_CHECK(hipMemcpyAsync(d_a7, d_a6, memSize, hipMemcpyDeviceToDevice, queue));
        HIP_CHECK(hipMemcpyAsync(d_a8, d_a7, memSize, hipMemcpyDeviceToDevice, queue));
        HIP_CHECK(hipMemcpyAsync(d_a9, d_a8, memSize, hipMemcpyDeviceToDevice, queue));
        HIP_CHECK(hipMemcpyAsync(d_a10, d_a9, memSize, hipMemcpyDeviceToDevice, queue));

        // Part 4 of 5: single kernel launch after all memcpys
        hipLaunchKernelGGL(add_arrays, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, queue,
                           d_a1, d_a2, d_a3, d_a4, d_a5,
                           d_a6, d_a7, d_a8, d_a9, d_a10,
                           d_result, N);

        // Part 5 of 5: device to host memory copy
        HIP_CHECK(hipMemcpyAsync(h_result, d_result, memSize, hipMemcpyDeviceToHost, queue));

        // Wait for the execution to finish
        HIP_CHECK(hipStreamSynchronize(queue));

        // End Timer
        HIP_CHECK(hipEventRecord(stop, queue));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(&elapsedTime, start, stop));
        if (istep >= skipBy) {
            totalTime += elapsedTime;
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

    float AverageTime = (totalTime + firstTime) / (NSTEP - skipBy);
    std::cout << "Average Time: " << AverageTime << "ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << "ms" << std::endl;
    std::cout << "Total Time without first run: " << totalTime << "ms" << std::endl;
    std::cout << "Total Time with first run: " << (totalTime + firstTime) << "ms" << std::endl;

    // **Print h_result before testing**
    std::cout << "h_result contents before verification:\n";
    std::cout << std::fixed << std::setprecision(2); // Set precision for floating-point output
    std::cout << "h_result[" << dimA - 1 << "] = " << h_result[dimA - 1] << "\n";

    // Verify the data on the host is correct
    for (int i = 0; i < dimA; ++i)
    {
        float expected = i * 10.0f; // Since each a_i contains i, and we sum over 10 arrays
        if (h_result[i] != expected) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected << ", got " << h_result[i] << std::endl;
            assert(false);
        }
    }

    // Destroy the HIP stream
    HIP_CHECK(hipStreamDestroy(queue));

    // Destroy the events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

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

    // If the program makes it this far, then the results are correct and
    // there are no run-time errors. Good work!
    std::cout << "Correct!" << std::endl;

    return 0;
}
