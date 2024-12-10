// Standard headers
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>    // For sqrt in standard deviation calculation

// Local headers
#include "../check_hip.h"

#define NSTEP 1000
#define SKIPBY 0

// Kernel functions
__global__ void kernelA(double* arrayA, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayA[x] *= 2.0; }
}

__global__ void kernelB(int* arrayB, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayB[x] = 3; }
}

__global__ void kernelC(double* arrayA, const int* arrayB, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayA[x] += arrayB[x]; }
}

__global__ void kernelD(float* arrayD, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayD[x] = sinf(arrayD[x]); }
}

__global__ void kernelE(int* arrayE, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayE[x] += 5; }
}

// Function for non-graph implementation
void runWithoutGraph(float* totalTimeWith, float* totalTimeWithout) {
    // Define different array sizes
    constexpr size_t arraySizeA = 1U << 20; // 1,048,576 elements
    constexpr size_t arraySizeB = 1U << 18; // 262,144 elements
    constexpr size_t arraySizeC = 1U << 16; // 65,536 elements
    constexpr size_t arraySizeD = 1U << 17; // 131,072 elements
    constexpr size_t arraySizeE = 1U << 19; // 524,288 elements

    constexpr int threadsPerBlock = 256;

    // Compute the number of blocks for each kernel
    const int numBlocksA = (arraySizeA + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksB = (arraySizeB + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksC = (arraySizeC + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksD = (arraySizeD + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksE = (arraySizeE + threadsPerBlock - 1) / threadsPerBlock;

    constexpr double initValue = 2.0;

    // Host and device memory
    double* d_arrayA;
    int* d_arrayB;
    double* d_arrayC;
    float* d_arrayD;
    int* d_arrayE;

    double* h_arrayA = nullptr;
    int* h_arrayB = nullptr;
    float* h_arrayD = nullptr;
    int* h_arrayE = nullptr;
    HIP_CHECK(hipHostMalloc((void**)&h_arrayA, arraySizeA * sizeof(double)));
    HIP_CHECK(hipHostMalloc((void**)&h_arrayB, arraySizeB * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void**)&h_arrayD, arraySizeD * sizeof(float)));
    HIP_CHECK(hipHostMalloc((void**)&h_arrayE, arraySizeE * sizeof(int)));

    // Initialize host arrays
    for (size_t i = 0; i < arraySizeA; i++) {
        h_arrayA[i] = initValue;
    }
    for (size_t i = 0; i < arraySizeB; i++) {
        h_arrayB[i] = 1;
    }
    for (size_t i = 0; i < arraySizeD; i++) {
        h_arrayD[i] = static_cast<float>(i) * 0.01f;
    }
    for (size_t i = 0; i < arraySizeE; i++) {
        h_arrayE[i] = 1;
    }

    // Set Timer for first run
    hipEvent_t firstCreateStart, firstCreateStop;
    float firstCreateTime = 0.0f;
    HIP_CHECK(hipEventCreate(&firstCreateStart));
    HIP_CHECK(hipEventCreate(&firstCreateStop));

    // Create a stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Start measuring first run time
    HIP_CHECK(hipEventRecord(firstCreateStart, stream));

    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_arrayA, arraySizeA * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_arrayB, arraySizeB * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_arrayC, arraySizeC * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_arrayD, arraySizeD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_arrayE, arraySizeE * sizeof(int)));
    // HIP_CHECK(hipMallocAsync(&d_arrayA, arraySizeA * sizeof(double), stream));
    // HIP_CHECK(hipMallocAsync(&d_arrayB, arraySizeB * sizeof(int), stream));
    // HIP_CHECK(hipMallocAsync(&d_arrayC, arraySizeC * sizeof(double), stream));
    // HIP_CHECK(hipMallocAsync(&d_arrayD, arraySizeD * sizeof(float), stream));
    // HIP_CHECK(hipMallocAsync(&d_arrayE, arraySizeE * sizeof(int), stream));

    // Copy h_arrayA to device
    HIP_CHECK(hipMemcpyAsync(d_arrayA, h_arrayA, arraySizeA * sizeof(double), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_arrayB, h_arrayB, arraySizeB * sizeof(int), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_arrayD, h_arrayD, arraySizeD * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_arrayE, h_arrayE, arraySizeE * sizeof(int), hipMemcpyHostToDevice, stream));

    // Launch kernels
    hipLaunchKernelGGL(kernelA, dim3(numBlocksA), dim3(threadsPerBlock), 0, stream, d_arrayA, arraySizeA);
    hipLaunchKernelGGL(kernelB, dim3(numBlocksB), dim3(threadsPerBlock), 0, stream, d_arrayB, arraySizeB);
    hipLaunchKernelGGL(kernelC, dim3(numBlocksC), dim3(threadsPerBlock), 0, stream, d_arrayA, d_arrayB, arraySizeC);
    hipLaunchKernelGGL(kernelD, dim3(numBlocksD), dim3(threadsPerBlock), 0, stream, d_arrayD, arraySizeD);
    hipLaunchKernelGGL(kernelE, dim3(numBlocksE), dim3(threadsPerBlock), 0, stream, d_arrayE, arraySizeE);

    // Copy data back to host
    HIP_CHECK(hipMemcpyAsync(h_arrayA, d_arrayA, arraySizeA * sizeof(double), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(h_arrayD, d_arrayD, arraySizeD * sizeof(float), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(h_arrayE, d_arrayE, arraySizeE * sizeof(int), hipMemcpyDeviceToHost, stream));

    // Wait for all operations to complete
    HIP_CHECK(hipStreamSynchronize(stream));

    // Free device memory
    HIP_CHECK(hipFree(d_arrayA));
    HIP_CHECK(hipFree(d_arrayB));
    HIP_CHECK(hipFree(d_arrayC));
    HIP_CHECK(hipFree(d_arrayD));
    HIP_CHECK(hipFree(d_arrayE));
    // HIP_CHECK(hipFreeAsync(d_arrayA, stream));
    // HIP_CHECK(hipFreeAsync(d_arrayB, stream));
    // HIP_CHECK(hipFreeAsync(d_arrayC, stream));
    // HIP_CHECK(hipFreeAsync(d_arrayD, stream));
    // HIP_CHECK(hipFreeAsync(d_arrayE, stream));

    // Stop measuring first run time
    HIP_CHECK(hipEventRecord(firstCreateStop, stream));
    HIP_CHECK(hipEventSynchronize(firstCreateStop));
    HIP_CHECK(hipEventElapsedTime(&firstCreateTime, firstCreateStart, firstCreateStop));

    // Measure execution time
    hipEvent_t execStart, execStop;
    HIP_CHECK(hipEventCreate(&execStart));
    HIP_CHECK(hipEventCreate(&execStop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Execute the sequence multiple times
    for(int i = 0; i < NSTEP; ++i){

        // Initialize host arrays
        for (size_t j = 0; j < arraySizeA; j++) {
            h_arrayA[j] = initValue;
        }
        for (size_t j = 0; j < arraySizeB; j++) {
            h_arrayB[j] = 1;
        }
        for (size_t j = 0; j < arraySizeD; j++) {
            h_arrayD[j] = static_cast<float>(j) * 0.01f;
        }
        for (size_t i = 0; i < arraySizeE; i++) {
            h_arrayE[i] = 1;
        }

        HIP_CHECK(hipEventRecord(execStart, stream));

        // Allocate device memory
        HIP_CHECK(hipMalloc(&d_arrayA, arraySizeA * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_arrayB, arraySizeB * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_arrayC, arraySizeC * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_arrayD, arraySizeD * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_arrayE, arraySizeE * sizeof(int)));
        // HIP_CHECK(hipMallocAsync(&d_arrayA, arraySizeA * sizeof(double), stream));
        // HIP_CHECK(hipMallocAsync(&d_arrayB, arraySizeB * sizeof(int), stream));
        // HIP_CHECK(hipMallocAsync(&d_arrayC, arraySizeC * sizeof(double), stream));
        // HIP_CHECK(hipMallocAsync(&d_arrayD, arraySizeD * sizeof(float), stream));
        // HIP_CHECK(hipMallocAsync(&d_arrayE, arraySizeE * sizeof(int), stream));

        // Copy h_arrayA to device
        HIP_CHECK(hipMemcpyAsync(d_arrayA, h_arrayA, arraySizeA * sizeof(double), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_arrayB, h_arrayB, arraySizeB * sizeof(int), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_arrayD, h_arrayD, arraySizeD * sizeof(float), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_arrayE, h_arrayE, arraySizeE * sizeof(int), hipMemcpyHostToDevice, stream));

        // Launch kernels
        hipLaunchKernelGGL(kernelA, dim3(numBlocksA), dim3(threadsPerBlock), 0, stream, d_arrayA, arraySizeA);
        hipLaunchKernelGGL(kernelB, dim3(numBlocksB), dim3(threadsPerBlock), 0, stream, d_arrayB, arraySizeB);
        hipLaunchKernelGGL(kernelC, dim3(numBlocksC), dim3(threadsPerBlock), 0, stream, d_arrayA, d_arrayB, arraySizeC);
        hipLaunchKernelGGL(kernelD, dim3(numBlocksD), dim3(threadsPerBlock), 0, stream, d_arrayD, arraySizeD);
        hipLaunchKernelGGL(kernelE, dim3(numBlocksE), dim3(threadsPerBlock), 0, stream, d_arrayE, arraySizeE);

        // Copy data back to host
        HIP_CHECK(hipMemcpyAsync(h_arrayA, d_arrayA, arraySizeA * sizeof(double), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(h_arrayD, d_arrayD, arraySizeD * sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(h_arrayE, d_arrayE, arraySizeE * sizeof(int), hipMemcpyDeviceToHost, stream));

        // Wait for all operations to complete
        HIP_CHECK(hipStreamSynchronize(stream));

        // Free device memory
        HIP_CHECK(hipFree(d_arrayA));
        HIP_CHECK(hipFree(d_arrayB));
        HIP_CHECK(hipFree(d_arrayC));
        HIP_CHECK(hipFree(d_arrayD));
        HIP_CHECK(hipFree(d_arrayE));
        // HIP_CHECK(hipFreeAsync(d_arrayA, stream));
        // HIP_CHECK(hipFreeAsync(d_arrayB, stream));
        // HIP_CHECK(hipFreeAsync(d_arrayC, stream));
        // HIP_CHECK(hipFreeAsync(d_arrayD, stream));
        // HIP_CHECK(hipFreeAsync(d_arrayE, stream));

        HIP_CHECK(hipEventRecord(execStop, stream));
        HIP_CHECK(hipEventSynchronize(execStop));
        HIP_CHECK(hipEventElapsedTime(&elapsedTime, execStart, execStop));

        // Time calculations
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
            if (elapsedTime < lowerTime || lowerTime == 0.0f) {
                lowerTime = elapsedTime;
            }
        }
    }

    // Calculate mean and standard deviation
    float meanTime = (totalTime + firstCreateTime) / NSTEP;
    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
    double stdDevTime = sqrt(varianceTime);

    // Print out the time statistics
    std::cout << "=======Setup (No Graph)=======" << std::endl;
    std::cout << "Iterations: " << NSTEP << std::endl;
    std::cout << "Skip By: " << SKIPBY << std::endl;
    std::cout << "Kernels: kernelA, kernelB, kernelC, kernelD, kernelE" << std::endl;
    std::cout << "Array Sizes: " << arraySizeA << ", " << arraySizeB << ", " << arraySizeC << ", " << arraySizeD << ", " << arraySizeE << std::endl;
    std::cout << "=======Results (No Graph)=======" << std::endl;
    std::cout << "First Run: " << firstCreateTime << " ms" << std::endl;
    std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without firstRun: " << (totalTime / (NSTEP - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with firstRun: " << totalTime + firstCreateTime << " ms" << std::endl;
    
    std::cout << "h_arrayA: " << h_arrayA[5] << std::endl;
    std::cout << "h_arrayD: " << h_arrayD[5] << std::endl;
    std::cout << "h_arrayE: " << h_arrayE[5] << std::endl;

    // Verify results (simple check for demonstration purposes)
    constexpr double expectedA = initValue * 2.0 + 3; // For kernelA and kernelC
    bool passed = true;
    for(size_t i = 0; i < arraySizeA; i++){
        if(h_arrayA[i] != expectedA){
            passed = false;
            std::cerr << "Validation failed! Expected " << expectedA << " got " << h_arrayA[i] << " at index " << i << std::endl;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    // Clean up
    HIP_CHECK(hipEventDestroy(execStart));
    HIP_CHECK(hipEventDestroy(execStop));
    HIP_CHECK(hipEventDestroy(firstCreateStart));
    HIP_CHECK(hipEventDestroy(firstCreateStop));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipHostFree(h_arrayA));
    HIP_CHECK(hipHostFree(h_arrayB));
    HIP_CHECK(hipHostFree(h_arrayD));

    // Return total time including first run
    *totalTimeWith = totalTime + firstCreateTime;
    *totalTimeWithout = totalTime;
}

// Function for graph implementation
void runWithGraph(float* totalTimeWith, float* totalTimeWithout) {
    // Define different array sizes
    constexpr size_t arraySizeA = 1U << 20; // 1,048,576 elements
    constexpr size_t arraySizeB = 1U << 18; // 262,144 elements
    constexpr size_t arraySizeC = 1U << 16; // 65,536 elements
    constexpr size_t arraySizeD = 1U << 17; // 131,072 elements
    constexpr size_t arraySizeE = 1U << 19; // 524,288 elements

    constexpr int threadsPerBlock = 256;

    // Compute the number of blocks for each kernel
    const int numBlocksA = (arraySizeA + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksB = (arraySizeB + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksC = (arraySizeC + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksD = (arraySizeD + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksE = (arraySizeE + threadsPerBlock - 1) / threadsPerBlock;

    constexpr double initValue = 2.0;

    double* d_arrayA;
    int* d_arrayB;
    double* d_arrayC;
    float* d_arrayD;
    int* d_arrayE;

    double* h_arrayA = nullptr;
    int* h_arrayB = nullptr;
    float* h_arrayD = nullptr;
    int* h_arrayE = nullptr;
    HIP_CHECK(hipHostMalloc((void**)&h_arrayA, arraySizeA * sizeof(double)));
    HIP_CHECK(hipHostMalloc((void**)&h_arrayB, arraySizeB * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void**)&h_arrayD, arraySizeD * sizeof(float)));
    HIP_CHECK(hipHostMalloc((void**)&h_arrayE, arraySizeE * sizeof(int)));

    hipStream_t captureStream;
    HIP_CHECK(hipStreamCreate(&captureStream));

    // Initialize host arrays
    for (size_t i = 0; i < arraySizeA; i++) {
        h_arrayA[i] = initValue;
    }
    for (size_t i = 0; i < arraySizeB; i++) {
        h_arrayB[i] = 1;
    }
    for (size_t i = 0; i < arraySizeD; i++) {
        h_arrayD[i] = static_cast<float>(i) * 0.01f;
    }
    for (size_t i = 0; i < arraySizeE; i++) {
        h_arrayE[i] = 1;
    }

    // Set Timer for graph creation
    hipEvent_t graphCreateStart, graphCreateStop;
    float graphCreateTime = 0.0f;
    HIP_CHECK(hipEventCreate(&graphCreateStart));
    HIP_CHECK(hipEventCreate(&graphCreateStop));

    // Start measuring graph creation time
    HIP_CHECK(hipEventRecord(graphCreateStart, captureStream));

    // Start capturing operations
    HIP_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));

    // Allocate device memory asynchronously
    HIP_CHECK(hipMallocAsync(&d_arrayA, arraySizeA * sizeof(double), captureStream));
    HIP_CHECK(hipMallocAsync(&d_arrayB, arraySizeB * sizeof(int), captureStream));
    HIP_CHECK(hipMallocAsync(&d_arrayC, arraySizeC * sizeof(double), captureStream));
    HIP_CHECK(hipMallocAsync(&d_arrayD, arraySizeD * sizeof(float), captureStream));
    HIP_CHECK(hipMallocAsync(&d_arrayE, arraySizeE * sizeof(int), captureStream));

    // Copy arrays to device
    HIP_CHECK(hipMemcpyAsync(d_arrayA, h_arrayA, arraySizeA * sizeof(double), hipMemcpyHostToDevice, captureStream));
    HIP_CHECK(hipMemcpyAsync(d_arrayB, h_arrayB, arraySizeB * sizeof(int), hipMemcpyHostToDevice, captureStream));
    HIP_CHECK(hipMemcpyAsync(d_arrayD, h_arrayD, arraySizeD * sizeof(float), hipMemcpyHostToDevice, captureStream));
    HIP_CHECK(hipMemcpyAsync(d_arrayE, h_arrayE, arraySizeE * sizeof(int), hipMemcpyHostToDevice, captureStream));

    // Launch kernels
    hipLaunchKernelGGL(kernelA, dim3(numBlocksA), dim3(threadsPerBlock), 0, captureStream, d_arrayA, arraySizeA);
    hipLaunchKernelGGL(kernelB, dim3(numBlocksB), dim3(threadsPerBlock), 0, captureStream, d_arrayB, arraySizeB);
    hipLaunchKernelGGL(kernelC, dim3(numBlocksC), dim3(threadsPerBlock), 0, captureStream, d_arrayA, d_arrayB, arraySizeC);
    hipLaunchKernelGGL(kernelD, dim3(numBlocksD), dim3(threadsPerBlock), 0, captureStream, d_arrayD, arraySizeD);
    hipLaunchKernelGGL(kernelE, dim3(numBlocksE), dim3(threadsPerBlock), 0, captureStream, d_arrayE, arraySizeE);

    // Copy data back to host
    HIP_CHECK(hipMemcpyAsync(h_arrayA, d_arrayA, arraySizeA * sizeof(double), hipMemcpyDeviceToHost, captureStream));
    HIP_CHECK(hipMemcpyAsync(h_arrayD, d_arrayD, arraySizeD * sizeof(float), hipMemcpyDeviceToHost, captureStream));
    HIP_CHECK(hipMemcpyAsync(h_arrayE, d_arrayE, arraySizeE * sizeof(int), hipMemcpyDeviceToHost, captureStream));


    // Free device memory asynchronously
    HIP_CHECK(hipFreeAsync(d_arrayA, captureStream));
    HIP_CHECK(hipFreeAsync(d_arrayB, captureStream));
    HIP_CHECK(hipFreeAsync(d_arrayC, captureStream));
    HIP_CHECK(hipFreeAsync(d_arrayD, captureStream));
    HIP_CHECK(hipFreeAsync(d_arrayE, captureStream));

    // Stop capturing
    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(captureStream, &graph));

    // Create an executable graph
    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Destroy the graph template if not needed
    HIP_CHECK(hipGraphDestroy(graph));

    // Stop measuring graph creation time
    HIP_CHECK(hipEventRecord(graphCreateStop, captureStream));
    HIP_CHECK(hipEventSynchronize(graphCreateStop));
    HIP_CHECK(hipEventElapsedTime(&graphCreateTime, graphCreateStart, graphCreateStop));

    // Measure execution time
    hipEvent_t execStart, execStop;
    HIP_CHECK(hipEventCreate(&execStart));
    HIP_CHECK(hipEventCreate(&execStop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Launch the graph multiple times
    for(int i = 0; i < NSTEP; ++i){
        // Initialize host arrays
        for (size_t j = 0; j < arraySizeA; j++) {
            h_arrayA[j] = initValue;
        }
        for (size_t j = 0; j < arraySizeB; j++) {
            h_arrayB[j] = 1;
        }
        for (size_t j = 0; j < arraySizeD; j++) {
            h_arrayD[j] = static_cast<float>(j) * 0.01f;
        }
        for (size_t i = 0; i < arraySizeE; i++) {
            h_arrayE[i] = 1;
        }

        HIP_CHECK(hipEventRecord(execStart, captureStream));

        HIP_CHECK(hipGraphLaunch(graphExec, captureStream));
        HIP_CHECK(hipStreamSynchronize(captureStream));

        HIP_CHECK(hipEventRecord(execStop, captureStream));
        HIP_CHECK(hipEventSynchronize(execStop));
        HIP_CHECK(hipEventElapsedTime(&elapsedTime, execStart, execStop));

        // Time calculations
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
            if (elapsedTime < lowerTime || lowerTime == 0.0f) {
                lowerTime = elapsedTime;
            }
        }
    }

    // Calculate mean and standard deviation
    float meanTime = (totalTime + graphCreateTime) / NSTEP;
    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
    double stdDevTime = sqrt(varianceTime);

    // Print out the time statistics
    std::cout << "=======Setup (With Graph)=======" << std::endl;
    std::cout << "Iterations: " << NSTEP << std::endl;
    std::cout << "Skip By: " << SKIPBY << std::endl;
    std::cout << "Kernels: kernelA, kernelB, kernelC, kernelD, kernelE" << std::endl;
    std::cout << "Array Sizes: " << arraySizeA << ", " << arraySizeB << ", " << arraySizeC << ", " << arraySizeD << ", " << arraySizeE << std::endl;
    std::cout << "=======Results (With Graph)=======" << std::endl;
    std::cout << "Graph Creation Time: " << graphCreateTime << " ms" << std::endl;
    std::cout << "Average Time with Graph: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without Graph: " << (totalTime / (NSTEP - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;

    std::cout << "h_arrayA: " << h_arrayA[5] << std::endl;
    std::cout << "h_arrayD: " << h_arrayD[5] << std::endl;
    std::cout << "h_arrayE: " << h_arrayE[5] << std::endl;

    // Verify results (simple check for demonstration purposes)
    constexpr double expectedA = initValue * 2.0 + 3; // For kernelA and kernelC
    bool passed = true;
    for(size_t i = 0; i < arraySizeA; i++){
        if(h_arrayA[i] != expectedA){
            passed = false;
            std::cerr << "Validation failed! Expected " << expectedA << " got " << h_arrayA[i] << " at index " << i << std::endl;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    // Clean up
    HIP_CHECK(hipEventDestroy(execStart));
    HIP_CHECK(hipEventDestroy(execStop));
    HIP_CHECK(hipEventDestroy(graphCreateStart));
    HIP_CHECK(hipEventDestroy(graphCreateStop));
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipStreamDestroy(captureStream));
    HIP_CHECK(hipHostFree(h_arrayA));
    HIP_CHECK(hipHostFree(h_arrayB));
    HIP_CHECK(hipHostFree(h_arrayD));

    // Return total time including graph creation
    *totalTimeWith = totalTime + graphCreateTime;
    *totalTimeWithout = totalTime;
}

int main() {
    // Measure time for non-graph implementation
    float nonGraphTotalTime, nonGraphTotalTimeWithout;
    runWithoutGraph(&nonGraphTotalTime, &nonGraphTotalTimeWithout);

    // Measure time for graph implementation
    float graphTotalTime, graphTotalTimeWithout;
    runWithGraph(&graphTotalTime, &graphTotalTimeWithout);

    // Compute the difference
    float difference = nonGraphTotalTime - graphTotalTime;
    float diffPerKernel = difference / NSTEP;
    float diffPercentage = (difference / nonGraphTotalTime) * 100;

    // Compute the difference without including graph creation time
    float difference2 = nonGraphTotalTimeWithout - graphTotalTimeWithout;
    float diffPerKernel2 = difference2 / NSTEP;
    float diffPercentage2 = (difference2 / nonGraphTotalTimeWithout) * 100;

    // Print the differences
    std::cout << "=======Comparison without Graph Creation=======" << std::endl;
    std::cout << "Difference: " << difference2 << " ms" << std::endl;
    std::cout << "Difference per step: " << diffPerKernel2 << " ms" << std::endl;
    std::cout << "Difference percentage: " << diffPercentage2 << "%" << std::endl;

    // Print the differences including graph creation time
    std::cout << "=======Comparison=======" << std::endl;
    std::cout << "Difference: " << difference << " ms" << std::endl;
    std::cout << "Difference per step: " << diffPerKernel << " ms" << std::endl;
    std::cout << "Difference percentage: " << diffPercentage << "%" << std::endl;

    return 0;
}
