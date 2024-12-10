// Standard headers
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>    // For sqrt in standard deviation calculation

#include  "../check_hip.h"

#define DEFAULT_NSTEP 100000
#define DEFAULT_SKIPBY 0

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

__global__ void kernelD(double* arrayA, size_t size) {
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayA[x] += 2.0; }
}

__global__ void kernelE(int* arrayB, size_t size) {
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayB[x] += 2; }
}

// struct set_vector_args {
//     double* h_array;
//     double value;
//     size_t size;
// };

// void set_vector(void* args) {
//     set_vector_args* h_args = reinterpret_cast<set_vector_args*>(args);
//     double* array = h_args->h_array;
//     size_t size = h_args->size;
//     double value = h_args->value;

//     // Initialize h_array with the specified value
//     for (size_t i = 0; i < size; i++) {
//         array[i] = value;
//     }
// }

// Function for non-graph implementation with multiple streams
void runWithoutGraph(float* totalTimeWith, float* totalTimeWithout, int nstep, int skipby) {
    const int NSTEP = nstep;
    const int SKIPBY = skipby;

    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 1U << 20;
    constexpr double initValue = 2.0;

    double* d_arrayA;
    int* d_arrayB;
    double* h_array = nullptr;
    HIP_CHECK(hipHostMalloc((void**)&h_array, arraySize * sizeof(double)));

    // Initialize host array
    // std::fill_n(h_array, arraySize, initValue);
    for (size_t i = 0; i < arraySize; i++) {
        h_array[i] = initValue;
    }

    // Create streams
    hipStream_t stream1, stream2;
    HIP_CHECK(hipStreamCreate(&stream1));
    HIP_CHECK(hipStreamCreate(&stream2));

    // Allocate device memory
    HIP_CHECK(hipMallocAsync(&d_arrayA, arraySize * sizeof(double), stream1));
    HIP_CHECK(hipMallocAsync(&d_arrayB, arraySize * sizeof(int), stream1));

    // Set Timer for first run
    hipEvent_t firstCreateStart, firstCreateStop;
    float firstCreateTime = 0.0f;
    HIP_CHECK(hipEventCreate(&firstCreateStart));
    HIP_CHECK(hipEventCreate(&firstCreateStop));

    // START measuring first run time
    HIP_CHECK(hipEventRecord(firstCreateStart, stream1));
        
    // Initialize host array
    // std::fill_n(h_array, arraySize, initValue);
    // for (size_t i = 0; i < arraySize; i++) {
    //     h_array[i] = initValue;
    // }
    // set_vector_args* args = new set_vector_args{h_array, initValue, arraySize};
    // HIP_CHECK(hipLaunchHostFunc(stream1, set_vector, args));

    // Copy h_array to device on stream1
    HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), hipMemcpyHostToDevice, stream1));

    // Use events to synchronize between streams
    hipEvent_t event1, event2;
    HIP_CHECK(hipEventCreate(&event1));
    HIP_CHECK(hipEventCreate(&event2));

    // Launch kernelA on stream1
    hipLaunchKernelGGL(kernelA, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, arraySize);

    // Record event1 after kernelA in stream1
    HIP_CHECK(hipEventRecord(event1, stream1));

    hipLaunchKernelGGL(kernelD, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, arraySize);

    // Make stream2 wait for event1
    HIP_CHECK(hipStreamWaitEvent(stream2, event1, 0));

    // Launch kernelB on stream2
    hipLaunchKernelGGL(kernelB, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream2, d_arrayB, arraySize);

    hipLaunchKernelGGL(kernelE, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream2, d_arrayB, arraySize);

    // Record event2 after kernelB in stream2
    HIP_CHECK(hipEventRecord(event2, stream2));

    // Make stream1 wait for event2
    HIP_CHECK(hipStreamWaitEvent(stream1, event2, 0));

    // Launch kernelC on stream1 (depends on d_arrayA and d_arrayB)
    hipLaunchKernelGGL(kernelC, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, d_arrayB, arraySize);

    // Copy data back to host on stream1
    HIP_CHECK(hipMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), hipMemcpyDeviceToHost, stream1));

    HIP_CHECK(hipStreamSynchronize(stream1));

    // Wait for all operations to complete
    HIP_CHECK(hipEventRecord(firstCreateStop, stream1));
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
        for (size_t j = 0; j < arraySize; j++) {
            h_array[j] = initValue;
        }

        HIP_CHECK(hipEventRecord(execStart, stream1));

        // Reinitialize host array
        // std::fill_n(h_array, arraySize, initValue);
        // for (size_t j = 0; j < arraySize; j++) {
        //     h_array[j] = initValue;
        // }
        // set_vector_args* args = new set_vector_args{h_array, initValue, arraySize};
        // HIP_CHECK(hipLaunchHostFunc(stream1, set_vector, args));

        // Copy h_array to device on stream1
        HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), hipMemcpyHostToDevice, stream1));

        // Launch kernelA on stream1
        hipLaunchKernelGGL(kernelA, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, arraySize);

        // Record event1 after kernelA in stream1
        HIP_CHECK(hipEventRecord(event1, stream1));

        hipLaunchKernelGGL(kernelD, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, arraySize);

        // Make stream2 wait for event1
        HIP_CHECK(hipStreamWaitEvent(stream2, event1, 0));

        // Launch kernelB on stream2
        hipLaunchKernelGGL(kernelB, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream2, d_arrayB, arraySize);

        hipLaunchKernelGGL(kernelE, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream2, d_arrayB, arraySize);

        // Record event2 after kernelB in stream2
        HIP_CHECK(hipEventRecord(event2, stream2));

        // Make stream1 wait for event2
        HIP_CHECK(hipStreamWaitEvent(stream1, event2, 0));

        // Launch kernelC on stream1 (depends on d_arrayA and d_arrayB)
        hipLaunchKernelGGL(kernelC, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, d_arrayB, arraySize);

        // Copy data back to host on stream1
        HIP_CHECK(hipMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), hipMemcpyDeviceToHost, stream1));
            
        HIP_CHECK(hipStreamSynchronize(stream1));

        // Wait for all operations to complete
        HIP_CHECK(hipEventRecord(execStop, stream1));
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
    std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
    std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
    std::cout << "Array Size: " << arraySize << std::endl;
    std::cout << "=======Results (No Graph)=======" << std::endl;
    std::cout << "First Run: " << firstCreateTime << " ms" << std::endl;
    std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without firstRun: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with firstRun: " << totalTime + firstCreateTime << " ms" << std::endl;

    // Verify results
    constexpr double expected = (initValue * 2.0 + 2.0) + (3 + 2);
    std::cout << "Validation passed!" << " Expected " << expected << std::endl;
    bool passed = true;
    for(size_t i = 0; i < arraySize; i++){
        if(h_array[i] != expected){
            passed = false;
            std::cerr << "Validation failed! Index " << i << ": Expected " << expected << " got " << h_array[i] << std::endl;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    HIP_CHECK(hipFreeAsync(d_arrayA, stream1));
    HIP_CHECK(hipFreeAsync(d_arrayB, stream1));
    // Clean up
    HIP_CHECK(hipEventDestroy(execStart));
    HIP_CHECK(hipEventDestroy(execStop));
    HIP_CHECK(hipEventDestroy(firstCreateStart));
    HIP_CHECK(hipEventDestroy(firstCreateStop));
    HIP_CHECK(hipEventDestroy(event1));
    HIP_CHECK(hipEventDestroy(event2));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipStreamDestroy(stream2));
    HIP_CHECK(hipHostFree(h_array));

    // Return total time including first run
    *totalTimeWith = totalTime + firstCreateTime;
    *totalTimeWithout = totalTime;
}

// Function for graph implementation with multiple streams
void runWithGraph(float* totalTimeWith, float* totalTimeWithout, int nstep, int skipby) {
    const int NSTEP = nstep;
    const int SKIPBY = skipby;

    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 1U << 20;
    constexpr double initValue = 2.0;

    double* d_arrayA;
    int* d_arrayB;
    double* h_array = nullptr;
    HIP_CHECK(hipHostMalloc((void**)&h_array, arraySize * sizeof(double)));

    for (size_t j = 0; j < arraySize; j++) {
        h_array[j] = initValue;
    }

    // Create streams
    hipStream_t stream1, stream2;
    HIP_CHECK(hipStreamCreate(&stream1));
    HIP_CHECK(hipStreamCreate(&stream2));

    // Allocate device memory
    HIP_CHECK(hipMallocAsync(&d_arrayA, arraySize * sizeof(double), stream1));
    HIP_CHECK(hipMallocAsync(&d_arrayB, arraySize * sizeof(int), stream1));

    // Set Timer for graph creation
    hipEvent_t graphCreateStart, graphCreateStop;
    float graphCreateTime = 0.0f;
    HIP_CHECK(hipEventCreate(&graphCreateStart));
    HIP_CHECK(hipEventCreate(&graphCreateStop));
    // set_vector_args* args = new set_vector_args{h_array, initValue, arraySize};

    // Start measuring graph creation time
    HIP_CHECK(hipEventRecord(graphCreateStart, stream1));

    // Begin graph capture on stream1 only
    HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));

    // Launch host function to initialize h_array
    // HIP_CHECK(hipLaunchHostFunc(stream1, set_vector, args));

    // Copy h_array to device on stream1
    HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), hipMemcpyHostToDevice, stream1));

    // Launch kernelA on stream1
    hipLaunchKernelGGL(kernelA, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, arraySize);

    // Use events to synchronize between streams
    // Record event1 to be used by stream2
    hipEvent_t event1;
    HIP_CHECK(hipEventCreate(&event1));
    HIP_CHECK(hipEventRecord(event1, stream1));

    // Launch kernelD on stream1 
    hipLaunchKernelGGL(kernelD, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, arraySize);

    // All operations done before going to stream2
    HIP_CHECK(hipStreamWaitEvent(stream2, event1, 0));

    // Launch kernelB on stream2
    hipLaunchKernelGGL(kernelB, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream2, d_arrayB, arraySize);

    hipLaunchKernelGGL(kernelE, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream2, d_arrayB, arraySize);
    
    // Record event2 to be used by stream1
    hipEvent_t event2;
    HIP_CHECK(hipEventCreate(&event2));
    HIP_CHECK(hipEventRecord(event2, stream2));

    // Waiting for event2 in stream 1
    HIP_CHECK(hipStreamWaitEvent(stream1, event2, 0));

    // Launch kernelC on stream1 (depends on d_arrayA and d_arrayB)
    hipLaunchKernelGGL(kernelC, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, d_arrayB, arraySize);

    // Copy data back to host on stream1
    HIP_CHECK(hipMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), hipMemcpyDeviceToHost, stream1));

    // End graph capture
    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(stream1, &graph));

    // Create an executable graph
    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Destroy the graph template if not needed
    HIP_CHECK(hipGraphDestroy(graph));

    // Stop measuring graph creation time
    HIP_CHECK(hipEventRecord(graphCreateStop, stream1));
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

        for (size_t j = 0; j < arraySize; j++) {
            h_array[j] = initValue;
        }

        HIP_CHECK(hipEventRecord(execStart, stream1));

        // Launch the graph
        HIP_CHECK(hipGraphLaunch(graphExec, stream1));
        HIP_CHECK(hipStreamSynchronize(stream1));

        // Wait for all operations to complete
        HIP_CHECK(hipEventRecord(execStop, stream1));
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
    std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
    std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
    std::cout << "Array Size: " << arraySize << std::endl;
    std::cout << "=======Results (With Graph)=======" << std::endl;
    std::cout << "Graph Creation Time: " << graphCreateTime << " ms" << std::endl;
    std::cout << "Average Time with Graph: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without Graph: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;

    // Verify results
    constexpr double expected = (initValue * 2.0 + 2.0) + (3 + 2);
    std::cout << "Validation passed!" << " Expected " << expected << std::endl;
    bool passed = true;
    for(size_t i = 0; i < arraySize; i++){
        if(h_array[i] != expected){
            passed = false;
            std::cerr << "Validation failed! Index " << i << ": Expected " << expected << " got " << h_array[i] << std::endl;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    // Free device memory asynchronously
    HIP_CHECK(hipFreeAsync(d_arrayA, stream1));
    HIP_CHECK(hipFreeAsync(d_arrayB, stream1));

    // Clean up
    HIP_CHECK(hipEventDestroy(execStart));
    HIP_CHECK(hipEventDestroy(execStop));
    HIP_CHECK(hipEventDestroy(graphCreateStart));
    HIP_CHECK(hipEventDestroy(graphCreateStop));
    HIP_CHECK(hipEventDestroy(event1));
    HIP_CHECK(hipEventDestroy(event2));
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipStreamDestroy(stream2));
    HIP_CHECK(hipHostFree(h_array));

    // Return total time including graph creation
    *totalTimeWith = totalTime + graphCreateTime;
    *totalTimeWithout = totalTime;
}

int main(int argc, char* argv[]) {
    const int NSTEP = (argc > 1) ? atoi(argv[1]) : DEFAULT_NSTEP;
    const int SKIPBY = (argc > 2) ? atoi(argv[2]) : DEFAULT_SKIPBY;

    // Measure time for non-graph implementation
    float nonGraphTotalTime, nonGraphTotalTimeWithout;
    runWithoutGraph(&nonGraphTotalTime, &nonGraphTotalTimeWithout, NSTEP, SKIPBY);

    // Measure time for graph implementation
    float graphTotalTime, graphTotalTimeWithout;
    runWithGraph(&graphTotalTime, &graphTotalTimeWithout, NSTEP, SKIPBY);
        
    // Compute the difference
    float difference = nonGraphTotalTime - graphTotalTime;
    float diffPerKernel = difference / (NSTEP);
    float diffPercentage = (difference / nonGraphTotalTime) * 100;

    // Compute the difference for without including Graph
    float difference2 = nonGraphTotalTimeWithout - graphTotalTimeWithout;
    float diffPerKernel2 = difference2 / (NSTEP-1);
    float diffPercentage2 = (difference2 / nonGraphTotalTimeWithout) * 100;

    // Print the differences
    std::cout << "=======Comparison without Graph Creation=======" << std::endl;
    std::cout << "Difference: " << difference2 << " ms" << std::endl;
    std::cout << "Difference per step: " << diffPerKernel2 << " ms" << std::endl;
    std::cout << "Difference percentage: " << diffPercentage2 << "%" << std::endl;

    // Print the differences
    std::cout << "=======Comparison=======" << std::endl;
    std::cout << "Difference: " << difference << " ms" << std::endl;
    std::cout << "Difference per step: " << diffPerKernel << " ms" << std::endl;
    std::cout << "Difference percentage: " << diffPercentage << "%" << std::endl;

    return 0;
}
