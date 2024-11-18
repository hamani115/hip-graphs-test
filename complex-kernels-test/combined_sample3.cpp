#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>    // For sqrt in standard deviation calculation
#include <cassert>  // For assert in result verification
#include  "../check_hip.h"

#define NSTEP 10
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

struct set_vector_args {
    double* h_array;
    double value;
    size_t size;
};

void set_vector(void* args) {
    set_vector_args* h_args = reinterpret_cast<set_vector_args*>(args);
    double* array = h_args->h_array;
    size_t size = h_args->size;
    double value = h_args->value;

    // Initialize h_array with the specified value
    for (size_t i = 0; i < size; ++i) {
        array[i] = value;
    }

    // Do NOT delete h_args here
}

// Function for non-graph implementation
float runWithoutGraph() {
    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 1U << 20; // 1,048,576 elements

    double* d_arrayA;
    int* d_arrayB;
    std::vector<double> h_array(arraySize);
    constexpr double initValue = 2.0;

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
    HIP_CHECK(hipMalloc(&d_arrayA, arraySize * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_arrayB, arraySize * sizeof(int)));

    // Initialize host array using index i
    // for (size_t i = 0; i < arraySize; ++i) {
    //     h_array[i] = static_cast<double>(i);
    // }

    // Initialize host array
    h_array.assign(h_array.size(), initValue);

    // Copy h_array to device
    HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array.data(), arraySize * sizeof(double), hipMemcpyHostToDevice, stream));

    // Launch kernels
    kernelA<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, arraySize);
    kernelB<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayB, arraySize);
    kernelC<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, d_arrayB, arraySize);

    // Copy data back to host
    HIP_CHECK(hipMemcpyAsync(h_array.data(), d_arrayA, arraySize * sizeof(double), hipMemcpyDeviceToHost, stream));

    // Wait for all operations to complete
    HIP_CHECK(hipStreamSynchronize(stream));

    // Free device memory
    HIP_CHECK(hipFree(d_arrayA));
    HIP_CHECK(hipFree(d_arrayB));

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
    // int skipBy = 0;
    // Variables for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Execute the sequence multiple times
    // constexpr int iterations = 1000;
    for(int i = 0; i < NSTEP - 1; ++i){
        HIP_CHECK(hipEventRecord(execStart, stream));

        // Allocate device memory
        HIP_CHECK(hipMalloc(&d_arrayA, arraySize * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_arrayB, arraySize * sizeof(int)));

        // Initialize host array using index i
        // for (size_t j = 0; j < arraySize; ++j) {
        //     h_array[j] = static_cast<double>(j);
        // }

        // Initialize host array
        h_array.assign(h_array.size(), initValue);

        // Copy h_array to device
        HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array.data(), arraySize * sizeof(double), hipMemcpyHostToDevice, stream));

        // Launch kernels
        kernelA<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, arraySize);
        kernelB<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayB, arraySize);
        kernelC<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, d_arrayB, arraySize);

        // Copy data back to host
        HIP_CHECK(hipMemcpyAsync(h_array.data(), d_arrayA, arraySize * sizeof(double), hipMemcpyDeviceToHost, stream));

        // Wait for all operations to complete
        HIP_CHECK(hipStreamSynchronize(stream));

        // Free device memory
        HIP_CHECK(hipFree(d_arrayA));
        HIP_CHECK(hipFree(d_arrayB));

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
    std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
    std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
    std::cout << "Array Size: " << arraySize << std::endl;
    std::cout << "=======Results (No Graph)=======" << std::endl;
    std::cout << "First Run: " << firstCreateTime << " ms" << std::endl;
    std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without firstRun: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " <<  varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with firstRun: " << totalTime + firstCreateTime << " ms" << std::endl;

    // Verify results
    constexpr double expected = initValue * 2.0 + 3;
    bool passed = true;
    for(size_t i = 0; i < arraySize; ++i){
        if(h_array[i] != expected){
            passed = false;
            std::cerr << "Validation failed! Expected " << expected << " got " << h_array[i] << " at index " << i << std::endl;
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

    // Return total time including first run
    return totalTime + firstCreateTime;
}

// Function for graph implementation
float runWithGraph() {
    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 1U << 20; // 1,048,576 elements

    double* d_arrayA;
    int* d_arrayB;
    double* h_array = nullptr;
    HIP_CHECK(hipHostMalloc((void**)&h_array, arraySize * sizeof(double), hipHostMallocDefault));

    constexpr double initValue = 2.0;

    // Set Timer for graph creation
    hipEvent_t graphCreateStart, graphCreateStop;
    float graphCreateTime = 0.0f;
    HIP_CHECK(hipEventCreate(&graphCreateStart));
    HIP_CHECK(hipEventCreate(&graphCreateStop));

    hipStream_t captureStream;
    HIP_CHECK(hipStreamCreate(&captureStream));

    // Start measuring graph creation time
    HIP_CHECK(hipEventRecord(graphCreateStart, captureStream));

    // ##### Start capturing operations
    HIP_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));

    // Allocate device memory asynchronously
    HIP_CHECK(hipMallocAsync(&d_arrayA, arraySize * sizeof(double), captureStream));
    HIP_CHECK(hipMallocAsync(&d_arrayB, arraySize * sizeof(int), captureStream));

    // Assign host function to the stream to initialize h_array
    // struct set_vector_args {
    //     double* h_array;
    //     double value;
    //     size_t size;
    // };

    // auto set_vector = [](void* args) {
    //     set_vector_args* h_args = reinterpret_cast<set_vector_args*>(args);
    //     double* array = h_args->h_array;
    //     size_t size = h_args->size;
    //     double value = h_args->value;

    //     // Initialize h_array with the specified value
    //     for (size_t i = 0; i < size; ++i) {
    //         array[i] = static_cast<double>(i);
    //     }
    // };
    set_vector_args* args = new set_vector_args{h_array, initValue, arraySize};
    HIP_CHECK(hipLaunchHostFunc(captureStream, set_vector, args));

    // set_vector_args args{h_array, initValue, arraySize};
    // HIP_CHECK(hipLaunchHostFunc(captureStream, set_vector, &args));

    HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), hipMemcpyHostToDevice, captureStream));

    kernelA<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, arraySize);
    kernelB<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayB, arraySize);
    kernelC<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, d_arrayB, arraySize);

    HIP_CHECK(hipMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), hipMemcpyDeviceToHost, captureStream));

    HIP_CHECK(hipFreeAsync(d_arrayA, captureStream));
    HIP_CHECK(hipFreeAsync(d_arrayB, captureStream));

    // ###### Stop capturing
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

    // Measure the execution time separately
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

    // Launch the graph multiple times
    // constexpr int iterations = 1000;
    for(int i = 0; i < NSTEP - 1; ++i){
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
    constexpr double expected = initValue * 2.0 + 3;
    bool passed = true;
    for(size_t i = 0; i < arraySize; ++i){
        if(h_array[i] != expected){
            passed = false;
            std::cerr << "Validation failed! Expected " << expected << " got " << h_array[i] << " at index " << i << std::endl;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    HIP_CHECK(hipEventDestroy(execStart));
    HIP_CHECK(hipEventDestroy(execStop));
    HIP_CHECK(hipEventDestroy(graphCreateStart));
    HIP_CHECK(hipEventDestroy(graphCreateStop));
    // Free graph and stream resources after usage
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipStreamDestroy(captureStream));
    HIP_CHECK(hipHostFree(h_array));

    // Return total time including graph creation
    return totalTime + graphCreateTime;
}

int main() {
    // Measure time for non-graph implementation
    float nonGraphTotalTime = runWithoutGraph();

    // Measure time for graph implementation
    float graphTotalTime = runWithGraph();

    // Compute the difference
    float difference = nonGraphTotalTime - graphTotalTime;
    float diffPerStep = difference / NSTEP; // iterations
    float diffPercentage = (difference / nonGraphTotalTime) * 100;

    // Print the differences
    std::cout << "=======Comparison=======" << std::endl;
    std::cout << "Difference: " << difference << " ms" << std::endl;
    std::cout << "Difference per step: " << diffPerStep << " ms" << std::endl;
    std::cout << "Difference percentage: " << diffPercentage << "%" << std::endl;

    return 0;
}
