#include <stdio.h>
#include <iostream>

// HIP API header
#include <hip/hip_runtime.h>

// Local headers
#include "../check_hip.h"

#define N (1 << 12)  // Size of the arrays - 4096
#define NSTEP 10000  // Number of steps

// HIP kernel to add 10 arrays element-wise
__global__ void add_arrays(float* a1, float* a2, float* a3, float* a4, float* a5,
                           float* a6, float* a7, float* a8, float* a9, float* a10,
                           float* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result[i] = a1[i] + a2[i] + a3[i] + a4[i] + a5[i]
                  + a6[i] + a7[i] + a8[i] + a9[i] + a10[i];
    }
}

int main() {
    const int size = N * sizeof(float);

    // Allocate host memory
    float* h_a1 = (float*)malloc(size);
    float* h_a2 = (float*)malloc(size);
    float* h_a3 = (float*)malloc(size);
    float* h_a4 = (float*)malloc(size);
    float* h_a5 = (float*)malloc(size);
    float* h_a6 = (float*)malloc(size);
    float* h_a7 = (float*)malloc(size);
    float* h_a8 = (float*)malloc(size);
    float* h_a9 = (float*)malloc(size);
    float* h_a10 = (float*)malloc(size);
    float* h_result = (float*)malloc(size);

    // Initialize host arrays with values
    for (int i = 0; i < N; i++) {
        h_a1[i] = 1.0f;
        h_a2[i] = 2.0f;
        h_a3[i] = 3.0f;
        h_a4[i] = 4.0f;
        h_a5[i] = 5.0f;
        h_a6[i] = 6.0f;
        h_a7[i] = 7.0f;
        h_a8[i] = 8.0f;
        h_a9[i] = 9.0f;
        h_a10[i] = 10.0f;
        h_result[i] = 0.0f;  // Initialize result array to zero
    }

    // Allocate device memory
    float *d_a1, *d_a2, *d_a3, *d_a4, *d_a5;
    float *d_a6, *d_a7, *d_a8, *d_a9, *d_a10;
    float* d_result;

    HIP_CHECK(hipMalloc((void**)&d_a1, size));
    HIP_CHECK(hipMalloc((void**)&d_a2, size));
    HIP_CHECK(hipMalloc((void**)&d_a3, size));
    HIP_CHECK(hipMalloc((void**)&d_a4, size));
    HIP_CHECK(hipMalloc((void**)&d_a5, size));
    HIP_CHECK(hipMalloc((void**)&d_a6, size));
    HIP_CHECK(hipMalloc((void**)&d_a7, size));
    HIP_CHECK(hipMalloc((void**)&d_a8, size));
    HIP_CHECK(hipMalloc((void**)&d_a9, size));
    HIP_CHECK(hipMalloc((void**)&d_a10, size));
    HIP_CHECK(hipMalloc((void**)&d_result, size));

    // Set Timer
    hipEvent_t start, stop;
    float elapsedTime = 0.0f;
    float firstTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    int skipBy = 100;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Create a HIP stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Define execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Start Timer
    HIP_CHECK(hipEventRecord(start, stream));

    // Copy host arrays to device arrays asynchronously
    HIP_CHECK(hipMemcpyAsync(d_a1, h_a1, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a2, h_a2, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a3, h_a3, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a4, h_a4, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a5, h_a5, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a6, h_a6, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a7, h_a7, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a8, h_a8, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a9, h_a9, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_a10, h_a10, size, hipMemcpyHostToDevice, stream));

    // Launch kernel to add arrays on the created stream
    hipLaunchKernelGGL(add_arrays, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream,
                       d_a1, d_a2, d_a3, d_a4, d_a5,
                       d_a6, d_a7, d_a8, d_a9, d_a10,
                       d_result);
    HIP_CHECK(hipGetLastError()); // Check for any errors from the launch

    // Copy result back to host asynchronously
    HIP_CHECK(hipMemcpyAsync(h_result, d_result, size, hipMemcpyDeviceToHost, stream));

    // Synchronize the stream to ensure all operations are complete
    HIP_CHECK(hipStreamSynchronize(stream));

    // End Timer
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&firstTime, start, stop));

    for (int istep = 0; istep < NSTEP - 1; istep++) {
        // Modifying buffers
        for (int i = 0; i < N; i++) {
            h_a1[i] += 1.0f; // or any other modification
            h_a2[i] += 1.0f;
            h_a3[i] += 1.0f;
            h_a4[i] += 1.0f;
            h_a5[i] += 1.0f;
            h_a6[i] += 1.0f;
            h_a7[i] += 1.0f;
            h_a8[i] += 1.0f;
            h_a9[i] += 1.0f;
            h_a10[i] += 1.0f;
        }

        // Start Timer
        HIP_CHECK(hipEventRecord(start, stream));

        // Copy host arrays to device arrays asynchronously
        HIP_CHECK(hipMemcpyAsync(d_a1, h_a1, size, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a2, h_a2, size, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a3, h_a3, size, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a4, h_a4, size, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a5, h_a5, size, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a6, h_a6, size, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a7, h_a7, size, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a8, h_a8, size, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a9, h_a9, size, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_a10, h_a10, size, hipMemcpyHostToDevice, stream));

        // Launch kernel to add arrays on the created stream
        hipLaunchKernelGGL(add_arrays, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream,
                           d_a1, d_a2, d_a3, d_a4, d_a5,
                           d_a6, d_a7, d_a8, d_a9, d_a10,
                           d_result);
        HIP_CHECK(hipGetLastError()); // Check for any errors from the launch

        // Copy result back to host asynchronously
        HIP_CHECK(hipMemcpyAsync(h_result, d_result, size, hipMemcpyDeviceToHost, stream));

        // Synchronize the stream to ensure all operations are complete
        HIP_CHECK(hipStreamSynchronize(stream));

        // End Timer
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(&elapsedTime, start, stop));

        // Time Calculations
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
        // std::cout << "Elapsed time " << istep << ": " << elapsedTime << "ms" << std::endl;
    }

    // Time Calculations
    float AverageTime = (totalTime + firstTime) / (NSTEP - skipBy);
    std::cout << "Average Time: " << AverageTime << "ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << "ms" << std::endl;
    std::cout << "Total Time without first run: " << totalTime << "ms" << std::endl;
    std::cout << "Total Time with first run: " << (totalTime + firstTime) << "ms" << std::endl;

    // Verify the result on the host
    int correct = 1;
    for (int i = 0; i < N; i++) {
        float expected = h_a1[i] + h_a2[i] + h_a3[i] + h_a4[i] + h_a5[i]
                       + h_a6[i] + h_a7[i] + h_a8[i] + h_a9[i] + h_a10[i];
        if (h_result[i] != expected) {
            correct = 0;
            printf("Error at index %d: Expected %f, got %f\n", i, expected, h_result[i]);
            break;
        }
    }

    if (correct) {
        printf("Test PASSED\n");
    } else {
        printf("Test FAILED\n");
    }

    // Destroy the stream
    HIP_CHECK(hipStreamDestroy(stream));

    // Free device memory
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

    // Free host memory
    free(h_a1);
    free(h_a2);
    free(h_a3);
    free(h_a4);
    free(h_a5);
    free(h_a6);
    free(h_a7);
    free(h_a8);
    free(h_a9);
    free(h_a10);
    free(h_result);

    return 0;
}
