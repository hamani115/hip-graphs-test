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

    double* d_arrayA;
    int* d_arrayB;
    std::vector<double> h_array(arraySize);
    constexpr double initValue = 2.0;

    // Set Timer
    hipEvent_t start, stop;
    float elapsedTime = 0.0f;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Create a stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    HIP_CHECK(hipEventRecord(start, stream));

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

    // Record stop event
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&elapsedTime, start, stop));

    std::cout << "Total Time: " << elapsedTime << "ms" << std::endl;

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
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipStreamDestroy(stream));

    return 0;
}
