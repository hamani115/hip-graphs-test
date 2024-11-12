#include <stdio.h>
// #include <chrono>
#include <iostream>

// HIP API header
#include <hip/hip_runtime.h>

// local header
#include "../check_hip.h"

#define N 64 //(1<<6) // Matrix dimensions (4096x4096)

#define NSTEP 100000
#define NKERNEL 100 // INDEPENDENT VARIABLE: CHANGE THE NUMBER OF KERNELS (10 OR 100)

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

void matrixMultiplyWithGraph(float* A, float* B, float* C, int width) {
    // Setup block and grid sizes
    dim3 block(32, 32);
    // dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y); //()im
    dim3 grid(6, 6);

    // Create a stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Create the HIP graph
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipEvent_t start, stop;
    float graphCreateTime = 0.0f;
    HIP_CHECK(hipEventCreate(&start)); 
    HIP_CHECK(hipEventCreate(&stop)); 

    // Start the timer for the graph creation
    HIP_CHECK(hipEventRecord(start, stream));

    // Begin graph capture
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    
    // Launch the kernel NKERNEL times
    for (int i = 0; i < NKERNEL; i++) {  // Run NKERNEL iterations
        hipLaunchKernelGGL(matMulKernel, grid, block, 0, stream, A, B, C, width);
    }
    HIP_CHECK(hipGetLastError());  // Check for kernel launch errors
    
    // End graph capture 
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    
    // Stop the timer for the graph creation
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop)); 
    HIP_CHECK(hipEventElapsedTime(&graphCreateTime, start, stop)); 

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    int skipBy = 100;
    // Variables for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Launch the graph NSTEP times
    for (int i = 0; i < NSTEP - 1; i++) {
        // Start the timer for each runs
        HIP_CHECK(hipEventRecord(start, stream));  
        
        // Launch the graph
        HIP_CHECK(hipGraphLaunch(graphExec, stream));
        HIP_CHECK(hipStreamSynchronize(stream)); // Ensure all kernels finish

        // Stop the timer for each runs
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop)); 
        HIP_CHECK(hipEventElapsedTime(&elapsedTime, start, stop));  
        
        // Calculate the total time and the time spread
        if(i >= skipBy){
            totalTime += elapsedTime;  

            // Welford's algorithm for calculating mean and variance
            count++;
            double delta = elapsedTime - mean;
            mean += delta / count;
            double delta2 = elapsedTime - mean;
            M2 += delta * delta2;

            if(elapsedTime > upperTime) { 
                upperTime = elapsedTime; 
            } 
            if(elapsedTime < lowerTime) { 
                lowerTime = elapsedTime; 
            }  
            if(i == skipBy){ 
                lowerTime = elapsedTime; 
            } 
        }
    }

    //  Welford's algorithm for calculating variance
    float meanTime = (totalTime + graphCreateTime) / (NSTEP - skipBy);
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
    std::cout << "Kernels: " << NKERNEL << std::endl;
    std::cout << "Block Size: " << block.x << " x " << block.y << std::endl;
    std::cout << "Grid Size: " << grid.x << " x " << grid.y << std::endl;
    std::cout << "Matrix Size: " << width << " x " << width << std::endl;
    std::cout << "=======Results=======" << std::endl;
    std::cout << "Graph Creation: " << graphCreateTime << std::endl;
    std::cout << "Average Time with Graph: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without Graph: " << (totalTime / (NSTEP - skipBy)) << " ms" << std::endl;
    std::cout << "Variance: " <<  varianceTime << " ms" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << " ms" << std::endl;
    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;
    
    // Cleanup
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipStreamDestroy(stream));
}

int main() {
    // Allocate host memory
    float* h_A = (float*)malloc(N * N * sizeof(float));
    float* h_B = (float*)malloc(N * N * sizeof(float));
    float* h_C = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    // Allocate device memory
    float* d_A, * d_B, * d_C;
    HIP_CHECK(hipMalloc(&d_A, N * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, N * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, N * N * sizeof(float)));

    // Copy matrices to device
    HIP_CHECK(hipMemcpy(d_A, h_A, N * N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, N * N * sizeof(float), hipMemcpyHostToDevice));

    // Measure time
    // auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyWithGraph(d_A, d_B, d_C, N);
    // auto end = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_C, d_C, N * N * sizeof(float), hipMemcpyDeviceToHost));

    // Calculate elapsed time
    // std::chrono::duration<double> elapsed = end - start;
    // printf("Elapsed time with HIP Graphs: %f seconds\n", elapsed.count());

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_B)); HIP_CHECK(hipFree(d_C));

    return 0;
}
