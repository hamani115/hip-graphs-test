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

// struct set_vector_args{
//     std::vector<double>& h_array;
//     double value;
// };

// void set_vector(void* args){
//     set_vector_args h_args{*(reinterpret_cast<set_vector_args*>(args))};

//     std::vector<double>& vec{h_args.h_array};
//     vec.assign(vec.size(), h_args.value);
// }

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
    // std::vector<double> h_array(arraySize);
    // double* h_array = new double[arraySize];
    // double* h_array = (double*)malloc(arraySize * sizeof(double));
    double* h_array = nullptr;
    HIP_CHECK(hipHostMalloc((void**)&h_array, arraySize * sizeof(double), hipHostMallocNumaUser));
    
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

    // hipMallocAsync and hipMemcpyAsync are needed, to be able to assign it to a stream
    HIP_CHECK(hipMallocAsync(&d_arrayA, arraySize*sizeof(double), captureStream));
    HIP_CHECK(hipMallocAsync(&d_arrayB, arraySize*sizeof(int), captureStream));

    // Assign host function to the stream
    // Needs a custom struct to pass the arguments
    // set_vector_args args{h_array, initValue};
    // HIP_CHECK(hipLaunchHostFunc(captureStream, set_vector, &args));
    set_vector_args* args = new set_vector_args{h_array, initValue, arraySize};
    HIP_CHECK(hipLaunchHostFunc(captureStream, set_vector, args));

    HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array, arraySize*sizeof(double), hipMemcpyHostToDevice, captureStream));

    kernelA<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, arraySize);
    kernelB<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayB, arraySize);
    kernelC<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, d_arrayB, arraySize);

    HIP_CHECK(hipMemcpyAsync(h_array, d_arrayA, arraySize*sizeof(*d_arrayA), hipMemcpyDeviceToHost, captureStream));

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

    // Now measure the execution time separately
    hipEvent_t execStart, execStop;
    float execTime = 0.0f;
    HIP_CHECK(hipEventCreate(&execStart));
    HIP_CHECK(hipEventCreate(&execStop));

    HIP_CHECK(hipEventRecord(execStart, captureStream));

    // Launch the graph multiple times
    constexpr int iterations = 2;
    for(int i = 0; i < iterations; ++i){
        // std::cout << "Inside loop: " << i << std::endl;
        HIP_CHECK(hipGraphLaunch(graphExec, captureStream));
    }
    HIP_CHECK(hipStreamSynchronize(captureStream));

    HIP_CHECK(hipEventRecord(execStop, captureStream));
    HIP_CHECK(hipEventSynchronize(execStop));
    HIP_CHECK(hipEventElapsedTime(&execTime, execStart, execStop));

    std::cout << "Graph Creation Time: " << graphCreateTime << "ms" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Average Execution Time per Iteration: " << (execTime / iterations) << "ms" << std::endl;
    std::cout << "Total Time: " << graphCreateTime + execTime << "ms" << std::endl;
    std::cout << "Average Execution Time per Iteration: " << ((execTime + graphCreateTime) / (iterations)) << "ms" << std::endl;

    // Verify results
    constexpr double expected = initValue * 2.0 + 3;
    bool passed = true;
    for(size_t i = 0; i < arraySize; ++i){
            if(h_array[i] != expected){
                    passed = false;
                    std::cerr << "Validation failed! Expected " << expected << " got " << h_array[0] << std::endl;
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
    delete args;
    HIP_CHECK(hipHostFree(h_array));

    return 0;
}
