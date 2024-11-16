#include <stdio.h>
#include <iostream>
#include <hip/hip_runtime.h>
#include "../check_hip.h"

#define N 64 //1024 // Matrix dimensions (1024x1024)
#define NSTEP 100000
#define NKERNEL 10 // Number of kernels
#define SKIPBY 0

// HIP kernel for matrix multiplication
__global__ void matMulKernel(float* A, float* B, float* C, int width) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    float sum = 0.0f;
    if (row < width && col < width) {
        for (int i = 0; i < width; i++) {
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Function for non-graph implementation
float matrixMultiplyNoGraph(int width) {
    dim3 block(32, 32); // 1024 threads
    dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);

    // Allocate host memory
    float* h_A = (float*)malloc(width * width * sizeof(float));
    float* h_B = (float*)malloc(width * width * sizeof(float));
    float* h_C = (float*)malloc(width * width * sizeof(float));

    // Initialize matrices using index i
    for (int i = 0; i < width * width; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }
    // for (int i = 0; i < N * N; i++) {
    //     h_A[i] = rand() % 100;
    //     h_B[i] = rand() % 100;
    // }

    // Allocate device memory
    float* d_A, * d_B, * d_C;
    HIP_CHECK(hipMalloc(&d_A, width * width * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, width * width * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, width * width * sizeof(float)));

    // Copy matrices to device
    HIP_CHECK(hipMemcpy(d_A, h_A, width * width * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, width * width * sizeof(float), hipMemcpyHostToDevice));

    // Create a stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Setup the timer variables
    hipEvent_t firstCreateStart, firstCreateStop;
    float firstTime = 0.0f;
    HIP_CHECK(hipEventCreate(&firstCreateStart));
    HIP_CHECK(hipEventCreate(&firstCreateStop));

    // Start the timer for the first run
    HIP_CHECK(hipEventRecord(firstCreateStart, stream));

    // Launch the kernel NKERNEL times
    for (int i = 0; i < NKERNEL; i++) {
        hipLaunchKernelGGL(matMulKernel, grid, block, 0, stream, d_A, d_B, d_C, width);
    }
    HIP_CHECK(hipGetLastError());  // Check for kernel launch errors
    // Synchronize after all kernels have been launched
    HIP_CHECK(hipStreamSynchronize(stream)); // Ensure all kernels finish

    // Stop the timer for the first run
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
    // int skipBy = 0;
    // Variables for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    for (int j = 0; j < NSTEP - 1; j++) {

        // Start the timer for each run
        HIP_CHECK(hipEventRecord(execStart, stream));

        // Launch the kernel NKERNEL times
        for (int i = 0; i < NKERNEL; i++) {  // Run NKERNEL iterations
            hipLaunchKernelGGL(matMulKernel, grid, block, 0, stream, d_A, d_B, d_C, width);
        }
        HIP_CHECK(hipGetLastError());  // Check for kernel launch errors
        HIP_CHECK(hipStreamSynchronize(stream)); // Ensure all kernels finish

        // Stop the timer for each run
        HIP_CHECK(hipEventRecord(execStop, stream));
        HIP_CHECK(hipEventSynchronize(execStop));
        HIP_CHECK(hipEventElapsedTime(&elapsedTime, execStart, execStop));

        // Calculate the total time and the time spread
        if (j >= SKIPBY) {
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
            if (j == SKIPBY) {
                lowerTime = elapsedTime;
            }
        }
    }

    // Calculate mean and standard deviation
    float meanTime = (totalTime + firstTime) / (NSTEP - SKIPBY);
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
    std::cout << "=======Setup (No Graph)=======" << std::endl;
    std::cout << "Iterations: " << NSTEP << std::endl;
    std::cout << "Skip By: " << SKIPBY << std::endl;
    std::cout << "Kernels: " << NKERNEL << std::endl;
    std::cout << "Block Size: " << block.x << " x " << block.y << std::endl;
    std::cout << "Grid Size: " << grid.x << " x " << grid.y << std::endl;
    std::cout << "Matrix Size: " << width << " x " << width << std::endl;
    std::cout << "=======Results (No Graph)=======" << std::endl;
    std::cout << "First Run: " << firstTime << " ms" << std::endl;
    std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without firstRun: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << " ms" << std::endl;
    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with firstRun: " << totalTime + firstTime << " ms" << std::endl;

    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_C, d_C, width * width * sizeof(float), hipMemcpyDeviceToHost));

    // Clean up
    HIP_CHECK(hipEventDestroy(execStart));
    HIP_CHECK(hipEventDestroy(execStop));
    HIP_CHECK(hipEventDestroy(firstCreateStart));
    HIP_CHECK(hipEventDestroy(firstCreateStop));
    HIP_CHECK(hipStreamDestroy(stream));

    free(h_A); free(h_B); free(h_C);
    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_B)); HIP_CHECK(hipFree(d_C));

    // Return the total time including the first run
    return totalTime + firstTime;
}

// Function for graph implementation
float matrixMultiplyWithGraph(int width) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);

    // Allocate host memory
    float* h_A = (float*)malloc(width * width * sizeof(float));
    float* h_B = (float*)malloc(width * width * sizeof(float));
    float* h_C = (float*)malloc(width * width * sizeof(float));

    // Initialize matrices using index i
    for (int i = 0; i < width * width; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_A, * d_B, * d_C;
    HIP_CHECK(hipMalloc(&d_A, width * width * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, width * width * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, width * width * sizeof(float)));

    // Copy matrices to device
    HIP_CHECK(hipMemcpy(d_A, h_A, width * width * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, width * width * sizeof(float), hipMemcpyHostToDevice));

    // Create a stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Create the HIP graph
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipEvent_t graphCreateStart, graphCreateStop;
    float graphCreateTime = 0.0f;
    HIP_CHECK(hipEventCreate(&graphCreateStart));
    HIP_CHECK(hipEventCreate(&graphCreateStop));

    // Start the timer for the graph creation
    HIP_CHECK(hipEventRecord(graphCreateStart, stream));

    // Begin graph capture
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    // Launch the kernel NKERNEL times
    for (int i = 0; i < NKERNEL; i++) {  // Run NKERNEL iterations
        hipLaunchKernelGGL(matMulKernel, grid, block, 0, stream, d_A, d_B, d_C, width);
    }
    HIP_CHECK(hipGetLastError());  // Check for kernel launch errors

    // End graph capture
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    HIP_CHECK(hipGraphDestroy(graph));

    // Stop the timer for the graph creation
    HIP_CHECK(hipEventRecord(graphCreateStop, stream));
    HIP_CHECK(hipEventSynchronize(graphCreateStop));
    HIP_CHECK(hipEventElapsedTime(&graphCreateTime, graphCreateStart, graphCreateStop));

    // Now measure the execution time separately
    hipEvent_t execStart, execStop;
    HIP_CHECK(hipEventCreate(&execStart));
    HIP_CHECK(hipEventCreate(&execStop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    // int skipBy = 0;
    // Variables for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Launch the graph NSTEP times
    for (int i = 0; i < NSTEP - 1; i++) {
        // Start the timer for each run
        HIP_CHECK(hipEventRecord(execStart, stream));

        // Launch the graph
        HIP_CHECK(hipGraphLaunch(graphExec, stream));
        HIP_CHECK(hipStreamSynchronize(stream)); // Ensure all kernels finish

        // Stop the timer for each run
        HIP_CHECK(hipEventRecord(execStop, stream));
        HIP_CHECK(hipEventSynchronize(execStop));
        HIP_CHECK(hipEventElapsedTime(&elapsedTime, execStart, execStop));

        // Calculate the total time and the time spread
        if (i >= SKIPBY) {
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
            if (i == SKIPBY) {
                lowerTime = elapsedTime;
            }
        }
    }

    //  Welford's algorithm for calculating variance
    float meanTime = (totalTime + graphCreateTime) / (NSTEP - SKIPBY);
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
    std::cout << "=======Setup (With Graph)=======" << std::endl;
    std::cout << "Iterations: " << NSTEP << std::endl;
    std::cout << "Skip By: " << SKIPBY << std::endl;
    std::cout << "Kernels: " << NKERNEL << std::endl;
    std::cout << "Block Size: " << block.x << " x " << block.y << std::endl;
    std::cout << "Grid Size: " << grid.x << " x " << grid.y << std::endl;
    std::cout << "Matrix Size: " << width << " x " << width << std::endl;
    std::cout << "=======Results (With Graph)=======" << std::endl;
    std::cout << "Graph Creation: " << graphCreateTime << " ms" << std::endl;
    std::cout << "Average Time with Graph: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without Graph: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << " ms" << std::endl;
    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;

    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_C, d_C, width * width * sizeof(float), hipMemcpyDeviceToHost));

    // Cleanup
    HIP_CHECK(hipEventDestroy(execStart));
    HIP_CHECK(hipEventDestroy(execStop));
    HIP_CHECK(hipEventDestroy(graphCreateStart));
    HIP_CHECK(hipEventDestroy(graphCreateStop));
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipStreamDestroy(stream));

    free(h_A); free(h_B); free(h_C);
    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_B)); HIP_CHECK(hipFree(d_C));

    // Return the total time including graph creation
    return totalTime + graphCreateTime;
}

int main() {
    // Measure time for non-graph implementation
    float nonGraphTotalTime = matrixMultiplyNoGraph(N);

    // Measure time for graph implementation
    float graphTotalTime = matrixMultiplyWithGraph(N);

    // Compute the difference
    float difference = nonGraphTotalTime - graphTotalTime;
    float diffPerKernel = difference / (NSTEP * NKERNEL);
    float diffPercentage = (difference / nonGraphTotalTime) * 100;

    // Print the differences
    std::cout << "=======Comparison=======" << std::endl;
    std::cout << "Difference: " << difference << " ms" << std::endl;
    std::cout << "Difference per kernel: " << diffPerKernel * 1000 << " μs" << std::endl;
    std::cout << "Difference percentage: " << diffPercentage << "%" << std::endl;

    return 0;
}
