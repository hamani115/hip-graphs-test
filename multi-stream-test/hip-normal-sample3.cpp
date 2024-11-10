#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#define HIP_CHECK(expression)                \
{                                            \
    const hipError_t status = expression;    \
    if(status != hipSuccess){                \
            std::cerr << "HIP error "        \
                << status << ": "            \
                << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;    \
    }                                        \
}

__global__ void kernelA(double* arrayA, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){arrayA[x] *= 2.0;}
}

__global__ void kernelB(int* arrayB, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){arrayB[x] = 3;}
}

__global__ void kernelC(double* arrayA, const int* arrayB, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){arrayA[x] += arrayB[x];}
}

int main(){
    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 1U << 20;

    // This example assumes that kernelA operates on data that needs to be initialized on
    // and copied from the host, while kernelB initializes the array that is passed to it.
    // Both arrays are then used as input to kernelC, where arrayA is also used as
    // output, that is copied back to the host, while arrayB is only read from and not modified.

    double* d_arrayA;
    int* d_arrayB;
    std::vector<double> h_array(arraySize);
    constexpr double initValue = 2.0;

    // Set Timer for graph creation
    hipEvent_t firstCreateStart, firstCreateStop;
    float firstCreateTime = 0.0f;
    HIP_CHECK(hipEventCreate(&firstCreateStart));
    HIP_CHECK(hipEventCreate(&firstCreateStop));

    // Create a stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Start measuring graph creation time
    HIP_CHECK(hipEventRecord(firstCreateStart, stream));

    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_arrayA, arraySize * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_arrayB, arraySize * sizeof(int)));

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
    
    // Stop measuring graph creation time
    HIP_CHECK(hipEventRecord(firstCreateStop, stream));
    HIP_CHECK(hipEventSynchronize(firstCreateStop));
    HIP_CHECK(hipEventElapsedTime(&firstCreateTime, firstCreateStart, firstCreateStop));

    // Measure execution time
    hipEvent_t execStart, execStop;
    // float elapsedTime = 0.0f;
    HIP_CHECK(hipEventCreate(&execStart));
    HIP_CHECK(hipEventCreate(&execStop));
    float elapsedTime = 0.0f;
    // float graphCreateTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    int skipBy = 0;
    // Variables for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // HIP_CHECK(hipEventRecord(execStart, stream));

    // Execute the sequence multiple times
    constexpr int iterations = 1000;
    for(int i = 0; i < iterations; ++i){
        HIP_CHECK(hipEventRecord(execStart, stream));
        // Allocate device memory
        HIP_CHECK(hipMalloc(&d_arrayA, arraySize * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_arrayB, arraySize * sizeof(int)));

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
        if (i >= skipBy) {
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
            if (i == skipBy) {
                lowerTime = elapsedTime;
            }
            // Uncomment to see elapsed time per iteration
            // std::cout << "Elapsed time " << i << ": " << elapsedTime << " ms" << std::endl;
        }
    }
    // HIP_CHECK(hipStreamSynchronize(stream));

    // HIP_CHECK(hipEventRecord(execStop, stream));
    // HIP_CHECK(hipEventSynchronize(execStop));
    // HIP_CHECK(hipEventElapsedTime(&execTime, execStart, execStop));

    // Calculate mean and standard deviation
    float meanTime = (totalTime + firstCreateTime) / (iterations - skipBy);
    double varianceTime3 = 0.0;
    if (count > 1) {
        varianceTime3 = M2 / (count - 1);
    }
    // Ensure variance is not negative due to floating-point errors
    if (varianceTime3 < 0.0) {
        varianceTime3 = 0.0;
    }
    double stdDevTime3 = sqrt(varianceTime3);

    std::cout << "New Measurements: " << std::endl;
    std::cout << "Average Time with Graph: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without Graph: " << (totalTime) / (iterations - 1 - skipBy)  << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime3 << " ms" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime3 << " ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << " ms" << std::endl;
    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with Graph Creation: " << totalTime + firstCreateTime << " ms" << std::endl;

    std::cout << "Old measurements: " << std::endl;
    std::cout << "First Run: " << firstCreateTime << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    // std::cout << "Average Execution Time per Iteration without firstRun: " << (execTime / (iterations-1)) << "ms" << std::endl;
    // std::cout << "Total Time with firstRun: " << execTime + firstCreateTime << "ms" << std::endl;
    // std::cout << "New Average Execution Time per Iteration with firstRun: " <<  ((execTime + firstCreateTime) / (iterations)) << std::endl;

    // ... [Verify results and clean up]
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

    return 0;
}
