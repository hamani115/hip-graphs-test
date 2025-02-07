// Standard headers
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>    // For sqrt in standard deviation calculation
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>

// Local headers
#include "../hip_check.h"
#include "../util/csv_util.h"

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

std::vector<int> generateSequence(int N) {
    std::vector<int> sequence;
    int current = 5; 
    bool multiplyByTwo = true;
    while (current <= N) {
        sequence.push_back(current);
        if (multiplyByTwo) {
            current *= 2;
        } else {
            current *= 5;
        }
        multiplyByTwo = !multiplyByTwo;
    }
    return sequence;
}

void runWithoutGraph(std::vector<float> &totalTimeWithArr, std::vector<float> &totalTimeWithoutArr,
                     std::vector<float> &chronoTotalTimeWithArr, std::vector<float> &chronoTotalTimeWithoutArr,
                     std::vector<float> &chronoTotalLaunchTimeWithArr, std::vector<float> &chronoTotalLaunchTimeWithoutArr,
                     int nstep, int skipby) {
    std::cout << "START OF runWithGraph" << '\n';

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

    hipStream_t stream1, stream2;
    HIP_CHECK(hipStreamCreate(&stream1));
    HIP_CHECK(hipStreamCreate(&stream2));

    HIP_CHECK(hipMallocAsync(&d_arrayA, arraySize * sizeof(double), stream1));
    HIP_CHECK(hipMallocAsync(&d_arrayB, arraySize * sizeof(int), stream1));

    hipEvent_t firstCreateStart, firstCreateStop;
    float firstCreateTime = 0.0f;
    HIP_CHECK(hipEventCreate(&firstCreateStart));
    HIP_CHECK(hipEventCreate(&firstCreateStop));

    HIP_CHECK(hipEventRecord(firstCreateStart, stream1));
    const auto graphStart = std::chrono::steady_clock::now();

    // Copy h_array to device on stream1
    HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), hipMemcpyHostToDevice, stream1));

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

    // Launch kernelB & E on stream2
    hipLaunchKernelGGL(kernelB, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream2, d_arrayB, arraySize);
    hipLaunchKernelGGL(kernelE, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream2, d_arrayB, arraySize);

    // Record event2 after kernelB in stream2
    HIP_CHECK(hipEventRecord(event2, stream2));

    // Make stream1 wait for event2
    HIP_CHECK(hipStreamWaitEvent(stream1, event2, 0));

    hipLaunchKernelGGL(kernelC, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, d_arrayB, arraySize);

    HIP_CHECK(hipMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), hipMemcpyDeviceToHost, stream1));
    HIP_CHECK(hipStreamSynchronize(stream1));

    const auto graphEnd = std::chrono::steady_clock::now();
    HIP_CHECK(hipEventRecord(firstCreateStop, stream1));
    HIP_CHECK(hipEventSynchronize(firstCreateStop));
    const auto graphEnd2 = std::chrono::steady_clock::now();
    HIP_CHECK(hipEventElapsedTime(&firstCreateTime, firstCreateStart, firstCreateStop));
    const std::chrono::duration<double> graphCreateTimeChrono = graphEnd - graphStart;
    const std::chrono::duration<double> graphCreateTimeChrono2 = graphEnd2 - graphStart;

    hipEvent_t execStart, execStop;
    HIP_CHECK(hipEventCreate(&execStart));
    HIP_CHECK(hipEventCreate(&execStop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;

    std::chrono::duration<double> totalTimeChrono(0.0);
    std::chrono::duration<double> totalLunchTimeChrono(0.0);
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    std::vector<int> nsteps = generateSequence(NSTEP);

    for(int i = 1; i <= NSTEP; i++){

        for (size_t j = 0; j < arraySize; j++) {
            h_array[j] = initValue;
        }

        HIP_CHECK(hipEventRecord(execStart, stream1));
        const auto start = std::chrono::steady_clock::now();

        HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), hipMemcpyHostToDevice, stream1));

        hipLaunchKernelGGL(kernelA, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, arraySize);
        HIP_CHECK(hipEventRecord(event1, stream1));
        hipLaunchKernelGGL(kernelD, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, arraySize);

        HIP_CHECK(hipStreamWaitEvent(stream2, event1, 0));
        hipLaunchKernelGGL(kernelB, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream2, d_arrayB, arraySize);
        hipLaunchKernelGGL(kernelE, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream2, d_arrayB, arraySize);

        HIP_CHECK(hipEventRecord(event2, stream2));
        HIP_CHECK(hipStreamWaitEvent(stream1, event2, 0));
        hipLaunchKernelGGL(kernelC, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, d_arrayB, arraySize);

        HIP_CHECK(hipMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), hipMemcpyDeviceToHost, stream1));

        const auto end = std::chrono::steady_clock::now();
        HIP_CHECK(hipEventRecord(execStop, stream1));
        HIP_CHECK(hipEventSynchronize(execStop));
        const auto end2 = std::chrono::steady_clock::now();
        HIP_CHECK(hipEventElapsedTime(&elapsedTime, execStart, execStop));

        if (i >= SKIPBY) {
            const std::chrono::duration<double> launchTimeChrono = end - start;
            const std::chrono::duration<double> lunchExecTimeChrono = end2 - start;
            totalTimeChrono += lunchExecTimeChrono;
            totalLunchTimeChrono += launchTimeChrono;
            totalTime += elapsedTime;

            count++;
            double delta = elapsedTime - mean;
            mean += delta / count;
            double delta2 = elapsedTime - mean;
            M2 += delta * delta2;

            if (elapsedTime > upperTime) upperTime = elapsedTime;
            if (elapsedTime < lowerTime || lowerTime == 0.0f) lowerTime = elapsedTime;

            for (auto num: nsteps) {
                if (num == i) {
                    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
                    double stdDevTime = sqrt(varianceTime);

                    std::cout << "=======Setup (No Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "Iterations: " << i << std::endl;
                    std::cout << "Skip By: " << SKIPBY << std::endl;
                    std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
                    std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
                    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
                    std::cout << "Array Size: " << arraySize << std::endl;
                    std::cout << "=======Results (No Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "First Run: " << firstCreateTime << " ms" << std::endl;
                    std::cout << "Average Time with firstRun: " << (totalTime + firstCreateTime) / (i + 1 - SKIPBY) << " ms" << std::endl;
                    std::cout << "Average Time without firstRun: " << (totalTime / (i - SKIPBY)) << " ms" << std::endl;
                    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
                    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
                    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
                    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
                    std::cout << "Total Time with firstRun: " << totalTime + firstCreateTime << " ms" << std::endl;

                    float totalTimeWith = totalTime + firstCreateTime;
                    const std::chrono::duration<double> totalTimeWithChrono = totalTimeChrono + graphCreateTimeChrono2;
                    const std::chrono::duration<double> totalLunchTimeWithChrono = totalLunchTimeChrono + graphCreateTimeChrono;
                    std::cout << "Graph Creation Chrono Launch before Sync: " << graphCreateTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Graph Creation Chrono LunchExec after Sync: " << graphCreateTimeChrono2.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch without Graph Creation: " << totalLunchTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch with Graph Creation: " << totalLunchTimeWithChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec without Graph Creation: " << totalTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec with Graph Creation: " << totalTimeWithChrono.count() * 1000 << " ms" << std::endl;

                    totalTimeWithArr.push_back(totalTimeWith);
                    totalTimeWithoutArr.push_back(totalTime);
                    chronoTotalTimeWithArr.push_back(totalTimeWithChrono.count()*1000);
                    chronoTotalTimeWithoutArr.push_back(totalTimeChrono.count()*1000);
                    chronoTotalLaunchTimeWithArr.push_back(totalLunchTimeWithChrono.count()*1000);
                    chronoTotalLaunchTimeWithoutArr.push_back(totalLunchTimeChrono.count()*1000);
                }
            }
        }
    }

    // Verify results
    constexpr double expected = (initValue * 2.0 + 2.0) + (3 + 2);
    bool passed = true;
    for(size_t i = 0; i < arraySize; i++){
        if(h_array[i] != expected){
            passed = false;
            break;
        }
    }
    if(!passed){
        std::cerr << "Validation failed." << std::endl;
    }

    HIP_CHECK(hipFreeAsync(d_arrayA, stream1));
    HIP_CHECK(hipFreeAsync(d_arrayB, stream1));
    HIP_CHECK(hipEventDestroy(execStart));
    HIP_CHECK(hipEventDestroy(execStop));
    HIP_CHECK(hipEventDestroy(firstCreateStart));
    HIP_CHECK(hipEventDestroy(firstCreateStop));
    HIP_CHECK(hipEventDestroy(event1));
    HIP_CHECK(hipEventDestroy(event2));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipStreamDestroy(stream2));
    HIP_CHECK(hipHostFree(h_array));

    std::cout << "END OF runWithoutGraph" << '\n';
}


void runWithGraph(std::vector<float>& totalTimeWithArr, std::vector<float>& totalTimeWithoutArr,
                  std::vector<float>& chronoTotalTimeWithArr, std::vector<float>& chronoTotalTimeWithoutArr,
                  std::vector<float>& chronoTotalLaunchTimeWithArr, std::vector<float>& chronoTotalLaunchTimeWithoutArr,
                  int nstep, int skipby) {
    std::cout << "START OF runWithGraph" << '\n';

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

    hipStream_t stream1, stream2;
    HIP_CHECK(hipStreamCreate(&stream1));
    HIP_CHECK(hipStreamCreate(&stream2));

    HIP_CHECK(hipMallocAsync(&d_arrayA, arraySize * sizeof(double), stream1));
    HIP_CHECK(hipMallocAsync(&d_arrayB, arraySize * sizeof(int), stream1));

    hipEvent_t graphCreateStart, graphCreateStop;
    float graphCreateTime = 0.0f;
    HIP_CHECK(hipEventCreate(&graphCreateStart));
    HIP_CHECK(hipEventCreate(&graphCreateStop));

    hipEvent_t event1, event2;
    HIP_CHECK(hipEventCreate(&event1));
    HIP_CHECK(hipEventCreate(&event2));

    HIP_CHECK(hipEventRecord(graphCreateStart, stream1));
    const auto graphStart = std::chrono::steady_clock::now();

    HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));

    HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), hipMemcpyHostToDevice, stream1));

    hipLaunchKernelGGL(kernelA, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, arraySize);

    HIP_CHECK(hipEventRecord(event1, stream1));

    hipLaunchKernelGGL(kernelD, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, arraySize);

    HIP_CHECK(hipStreamWaitEvent(stream2, event1, 0));
    hipLaunchKernelGGL(kernelB, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream2, d_arrayB, arraySize);
    hipLaunchKernelGGL(kernelE, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream2, d_arrayB, arraySize);

    HIP_CHECK(hipEventRecord(event2, stream2));
    HIP_CHECK(hipStreamWaitEvent(stream1, event2, 0));
    hipLaunchKernelGGL(kernelC, dim3(numOfBlocks), dim3(threadsPerBlock), 0, stream1, d_arrayA, d_arrayB, arraySize);

    HIP_CHECK(hipMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), hipMemcpyDeviceToHost, stream1));

    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(stream1, &graph));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphDestroy(graph));

    HIP_CHECK(hipGraphLaunch(graphExec, stream1));

    const auto graphEnd = std::chrono::steady_clock::now();
    HIP_CHECK(hipEventRecord(graphCreateStop, stream1));
    HIP_CHECK(hipEventSynchronize(graphCreateStop));
    const auto graphEnd2 = std::chrono::steady_clock::now();
    HIP_CHECK(hipEventElapsedTime(&graphCreateTime, graphCreateStart, graphCreateStop));
    const std::chrono::duration<double> graphCreateTimeChrono = graphEnd - graphStart;
    const std::chrono::duration<double> graphCreateTimeChrono2 = graphEnd2 - graphStart;

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

    std::vector<int> nsteps = generateSequence(NSTEP);

    std::chrono::duration<double> totalTimeChrono(0.0);
    std::chrono::duration<double> totalLunchTimeChrono(0.0);

    for(int i = 1; i <= NSTEP; i++){

        for (size_t j = 0; j < arraySize; j++) {
            h_array[j] = initValue;
        }

        HIP_CHECK(hipEventRecord(execStart, stream1));
        const auto start = std::chrono::steady_clock::now();

        HIP_CHECK(hipGraphLaunch(graphExec, stream1));

        const auto end = std::chrono::steady_clock::now();
        HIP_CHECK(hipEventRecord(execStop, stream1));
        HIP_CHECK(hipEventSynchronize(execStop));
        const auto end2 = std::chrono::steady_clock::now();
        HIP_CHECK(hipEventElapsedTime(&elapsedTime, execStart, execStop));

        if (i >= SKIPBY) {
            const std::chrono::duration<double> launchTimeChrono = end - start;
            const std::chrono::duration<double> lunchExecTimeChrono = end2 - start;
            totalTimeChrono += lunchExecTimeChrono;
            totalLunchTimeChrono += launchTimeChrono;
            totalTime += elapsedTime;

            count++;
            double delta = elapsedTime - mean;
            mean += delta / count;
            double delta2 = elapsedTime - mean;
            M2 += delta * delta2;

            if (elapsedTime > upperTime) upperTime = elapsedTime;
            if (elapsedTime < lowerTime || lowerTime == 0.0f) lowerTime = elapsedTime;

            for (auto num: nsteps) {
                if (num == i) {
                    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
                    double stdDevTime = sqrt(varianceTime);

                    std::cout << "=======Setup (With Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "Iterations: " << i << std::endl;
                    std::cout << "Skip By: " << SKIPBY << std::endl;
                    std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
                    std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
                    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
                    std::cout << "Array Size: " << arraySize << std::endl;
                    std::cout << "=======Results (With Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "Graph Creation Time: " << graphCreateTime << " ms" << std::endl;
                    std::cout << "Average Time with Graph: " << (totalTime + graphCreateTime) / (i + 1 - SKIPBY) << " ms" << std::endl;
                    std::cout << "Average Time without Graph: " << (totalTime / (i - SKIPBY)) << " ms" << std::endl;
                    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
                    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
                    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
                    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
                    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;
                    
                    float totalTimeWith = totalTime + graphCreateTime;
                    const std::chrono::duration<double> totalTimeWithChrono = totalTimeChrono + graphCreateTimeChrono2;
                    const std::chrono::duration<double> totalLunchTimeWithChrono = totalLunchTimeChrono + graphCreateTimeChrono;
                    std::cout << "Graph Creation Chrono Launch before Sync: " << graphCreateTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Graph Creation Chrono LunchExec after Sync: " << graphCreateTimeChrono2.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch without Graph Creation: " << totalLunchTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch with Graph Creation: " << totalLunchTimeWithChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec without Graph Creation: " << totalTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec with Graph Creation: " << totalTimeWithChrono.count() * 1000 << " ms" << std::endl;

                    totalTimeWithArr.push_back(totalTimeWith);
                    totalTimeWithoutArr.push_back(totalTime);
                    chronoTotalTimeWithArr.push_back(totalTimeWithChrono.count()*1000);
                    chronoTotalTimeWithoutArr.push_back(totalTimeChrono.count()*1000);
                    chronoTotalLaunchTimeWithArr.push_back(totalLunchTimeWithChrono.count()*1000);
                    chronoTotalLaunchTimeWithoutArr.push_back(totalLunchTimeChrono.count()*1000);
                }
            }
        }
    }

    // Verify results
    constexpr double expected = (initValue * 2.0 + 2.0) + (3 + 2);
    bool passed = true;
    for(size_t i = 0; i < arraySize; i++){
        if(h_array[i] != expected){
            passed = false;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    HIP_CHECK(hipFreeAsync(d_arrayA, stream1));
    HIP_CHECK(hipFreeAsync(d_arrayB, stream1));
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

    std::cout << "END OF runWithGraph" << '\n';
}


int main(int argc, char* argv[]) {
    const int NSTEP = (argc > 1) ? atoi(argv[1]) : DEFAULT_NSTEP;
    const int SKIPBY = (argc > 2) ? atoi(argv[2]) : DEFAULT_SKIPBY;
    const int NUM_RUNS = (argc > 3) ? std::atoi(argv[3]) : 4;
    std::string FILENAME;
    if (argc > 4) {
        FILENAME = argv[4];
        // Automatically append ".csv" if user didn't include it
        if (FILENAME.size() < 4 || FILENAME.compare(FILENAME.size() - 4, 4, ".csv") != 0) {
            FILENAME += ".csv";
        }
    } else {
        FILENAME = "complex_multi_malloc.csv";
    }


    std::cout << "==============COMPLEX MULTI STREAM KERNELS TEST==============" << std::endl;

    std::cout << "NSTEP    = " << NSTEP    << "\n";
    std::cout << "SKIPBY   = " << SKIPBY   << "\n";
    std::cout << "NUM_RUNS = " << NUM_RUNS << "\n";
    std::cout << "FILENAME = " << FILENAME << "\n\n";

    std::vector<int> nsteps = generateSequence(NSTEP);
    std::vector<CSVData> newDatas(nsteps.size());
    for (auto &newData : newDatas) {
        // Resize each vector to hold 'NUM_RUNS' elements
        newData.noneGraphTotalTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.GraphTotalTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.noneGraphTotalTimeWith.resize(NUM_RUNS, 0.0f);
        newData.GraphTotalTimeWith.resize(NUM_RUNS, 0.0f);
        newData.DiffTotalWithout.resize(NUM_RUNS, 0.0f);
        newData.DiffPerStepWithout.resize(NUM_RUNS, 0.0f);
        newData.DiffPercentWithout.resize(NUM_RUNS, 0.0f);
        newData.DiffTotalWith.resize(NUM_RUNS, 0.0f);
        newData.DiffPerStepWith.resize(NUM_RUNS, 0.0f);
        newData.DiffPercentWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoNoneGraphTotalTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoGraphTotalTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoNoneGraphTotalLaunchTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoGraphTotalLaunchTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoNoneGraphTotalTimeWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoGraphTotalTimeWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoNoneGraphTotalLaunchTimeWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoGraphTotalLaunchTimeWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffTotalTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffPerStepWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffPercentWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffTotalTimeWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffPerStepWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffPercentWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffLaunchTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffLaunchPercentWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffLaunchTimeWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffLaunchPercentWith.resize(NUM_RUNS, 0.0f);
    }


    for (int r = 0; r < NUM_RUNS; r++) {
        std::cout << "==============FOR RUN" << r+1 << "==============" << std::endl;
        std::vector<float> noneGraphTotalTimeWithArr, noneGraphTotalTimeWithoutArr;
        std::vector<float> chronoNoneGraphTotalTimeWithArr, chronoNoneGraphTotalTimeWithoutArr;
        std::vector<float> chronoNoneGraphTotalLaunchTimeWithArr, chronoNoneGraphTotalLaunchTimeWithoutArr;

        runWithoutGraph(noneGraphTotalTimeWithArr, noneGraphTotalTimeWithoutArr,
                        chronoNoneGraphTotalTimeWithArr, chronoNoneGraphTotalTimeWithoutArr,
                        chronoNoneGraphTotalLaunchTimeWithArr, chronoNoneGraphTotalLaunchTimeWithoutArr,
                        NSTEP, SKIPBY);

        std::vector<float> graphTotalTimeWithArr, graphTotalTimeWithoutArr;
        std::vector<float> chronoGraphTotalTimeWithArr, chronoGraphTotalTimeWithoutArr;
        std::vector<float> chronoGraphTotalLaunchTimeWithArr, chronoGraphTotalLaunchTimeWithoutArr;

        runWithGraph(graphTotalTimeWithArr, graphTotalTimeWithoutArr,
                     chronoGraphTotalTimeWithArr, chronoGraphTotalTimeWithoutArr,
                     chronoGraphTotalLaunchTimeWithArr, chronoGraphTotalLaunchTimeWithoutArr,
                     NSTEP, SKIPBY);

        for (int i = 0; i < (int)nsteps.size(); i++) {
            float difference = noneGraphTotalTimeWithArr[i] - graphTotalTimeWithArr[i];
            float diffPerKernel = difference / (nsteps[i] + 1);
            float diffPercentage = (difference / noneGraphTotalTimeWithArr[i]) * 100;

            float difference2 = noneGraphTotalTimeWithoutArr[i] - graphTotalTimeWithoutArr[i];
            float diffPerKernel2 = difference2 / (nsteps[i]);
            float diffPercentage2 = (difference2 / noneGraphTotalTimeWithoutArr[i]) * 100;

            float chronoDiffTotalTimeWith = chronoNoneGraphTotalTimeWithArr[i] - chronoGraphTotalTimeWithArr[i];
            float chronoDiffTotalTimeWithout = chronoNoneGraphTotalTimeWithoutArr[i] - chronoGraphTotalTimeWithoutArr[i];

            float chronoDiffPerStepWith = chronoDiffTotalTimeWith / (nsteps[i]+ 1); 
            float chronoDiffPercentWith = (chronoDiffTotalTimeWith / chronoNoneGraphTotalTimeWithArr[i]) * 100;

            float chronoDiffPerStepWithout = chronoDiffTotalTimeWithout / (nsteps[i]); 
            float chronoDiffPercentWithout = (chronoDiffTotalTimeWithout / chronoNoneGraphTotalTimeWithoutArr[i]) * 100;

            float chronoDiffLaunchTimeWith = chronoNoneGraphTotalLaunchTimeWithArr[i] - chronoGraphTotalLaunchTimeWithArr[i];
            float chronoDiffLaunchTimeWithout = chronoNoneGraphTotalLaunchTimeWithoutArr[i] - chronoGraphTotalLaunchTimeWithoutArr[i];

            float chronoDiffLaunchPercentWithout = (chronoDiffLaunchTimeWithout / chronoNoneGraphTotalLaunchTimeWithoutArr[i]) * 100;
            float chronoDiffLaunchPercentWith = (chronoDiffLaunchTimeWith / chronoNoneGraphTotalLaunchTimeWithArr[i]) * 100;

            std::cout << "==============For NSTEP " << nsteps[i] << "==============" << std::endl;
            std::cout << "=======Comparison without Graph Creation=======" << std::endl;
            std::cout << "Difference: " << difference2 << " ms" << std::endl;
            std::cout << "Difference per step: " << diffPerKernel2 << " ms" << std::endl;
            std::cout << "Difference percentage: " << diffPercentage2 << "%" << std::endl;

            std::cout << "=======Comparison=======" << std::endl;
            std::cout << "Difference: " << difference << " ms" << std::endl;
            std::cout << "Difference per step: " << diffPerKernel << " ms" << std::endl;
            std::cout << "Difference percentage: " << diffPercentage << "%" << std::endl;

            newDatas[i].NSTEP = nsteps[i];
            newDatas[i].SKIPBY = SKIPBY;
            newDatas[i].noneGraphTotalTimeWithout[r] = noneGraphTotalTimeWithoutArr[i];
            newDatas[i].GraphTotalTimeWithout[r] = graphTotalTimeWithoutArr[i];
            newDatas[i].noneGraphTotalTimeWith[r] = noneGraphTotalTimeWithArr[i];
            newDatas[i].GraphTotalTimeWith[r] = graphTotalTimeWithArr[i];
            newDatas[i].DiffTotalWithout[r] = difference2;
            newDatas[i].DiffPerStepWithout[r] = diffPerKernel2;
            newDatas[i].DiffPercentWithout[r] = diffPercentage2;
            newDatas[i].DiffTotalWith[r] = difference;
            newDatas[i].DiffPerStepWith[r] = diffPerKernel;
            newDatas[i].DiffPercentWith[r] = diffPercentage;
            newDatas[i].ChronoNoneGraphTotalTimeWithout[r] = chronoNoneGraphTotalTimeWithoutArr[i];
            newDatas[i].ChronoGraphTotalTimeWithout[r] = chronoGraphTotalTimeWithoutArr[i];
            newDatas[i].ChronoNoneGraphTotalLaunchTimeWithout[r] = chronoNoneGraphTotalLaunchTimeWithoutArr[i];
            newDatas[i].ChronoGraphTotalLaunchTimeWithout[r] = chronoGraphTotalLaunchTimeWithoutArr[i];
            newDatas[i].ChronoNoneGraphTotalTimeWith[r] = chronoNoneGraphTotalTimeWithArr[i];
            newDatas[i].ChronoGraphTotalTimeWith[r] = chronoGraphTotalTimeWithArr[i];
            newDatas[i].ChronoNoneGraphTotalLaunchTimeWith[r] = chronoNoneGraphTotalLaunchTimeWithArr[i];
            newDatas[i].ChronoGraphTotalLaunchTimeWith[r] = chronoGraphTotalLaunchTimeWithArr[i];
            newDatas[i].ChronoDiffTotalTimeWithout[r] = chronoDiffTotalTimeWithout;
            newDatas[i].ChronoDiffPerStepWithout[r] = chronoDiffPerStepWithout;
            newDatas[i].ChronoDiffPercentWithout[r] = chronoDiffPercentWithout;
            newDatas[i].ChronoDiffTotalTimeWith[r] = chronoDiffTotalTimeWith;
            newDatas[i].ChronoDiffPerStepWith[r] = chronoDiffPerStepWith;
            newDatas[i].ChronoDiffPercentWith[r] = chronoDiffPercentWith;
            newDatas[i].ChronoDiffLaunchTimeWithout[r] = chronoDiffLaunchTimeWithout;
            newDatas[i].ChronoDiffLaunchPercentWithout[r] = chronoDiffLaunchPercentWithout;
            newDatas[i].ChronoDiffLaunchTimeWith[r] = chronoDiffLaunchTimeWith;
            newDatas[i].ChronoDiffLaunchPercentWith[r] = chronoDiffLaunchPercentWith;
        }
    }

    // for (const auto &newData : newDatas) {
        // updateOrAppendCSV(FILENAME, newData);
    // }
    rewriteCSV(FILENAME, newDatas, NUM_RUNS);

    return 0;
}
