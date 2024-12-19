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

struct CSVData {
    int NSTEP;
    int SKIPBY;
    float noneGraphTotalTimeWithout[4];
    float GraphTotalTimeWithout[4];
    float noneGraphTotalTimeWith[4];
    float GraphTotalTimeWith[4];
    float DiffTotalWithout[4];
    float DiffPerStepWithout[4];
    float DiffPercentWithout[4];
    float DiffTotalWith[4];
    float DiffPerStepWith[4];
    float DiffPercentWith[4];
    float ChronoNoneGraphTotalTimeWithout[4];
    float ChronoGraphTotalTimeWithout[4];
    float ChronoNoneGraphTotalLaunchTimeWithout[4];
    float ChronoGraphTotalLaunchTimeWithout[4];
    float ChronoNoneGraphTotalTimeWith[4];
    float ChronoGraphTotalTimeWith[4];
    float ChronoNoneGraphTotalLaunchTimeWith[4];
    float ChronoGraphTotalLaunchTimeWith[4];
    float ChronoDiffTotalTimeWithout[4];
    float ChronoDiffPerStepWithout[4];
    float ChronoDiffPercentWithout[4];
    float ChronoDiffTotalTimeWith[4];
    float ChronoDiffPerStepWith[4];
    float ChronoDiffPercentWith[4];
    float ChronoDiffLaunchTimeWithout[4];
    float ChronoDiffLaunchPercentWithout[4];
    float ChronoDiffLaunchTimeWith[4];
    float ChronoDiffLaunchPercentWith[4];
};

// Helper function to read a float with error checking:
bool readFloatToken(std::istringstream &ss, float &val) {
    std::string token;
    if (!std::getline(ss, token, ',')) return false;
    val = std::stof(token);
    return true;
}

void updateOrAppendCSV(const std::string &filename, const CSVData &newData) {
    std::vector<CSVData> csvData;
    std::ifstream csvFileIn(filename);
    if (csvFileIn.is_open()) {
        std::string line;
        if (std::getline(csvFileIn, line));

        while (std::getline(csvFileIn, line)) {
            std::istringstream ss(line);
            CSVData data;
            std::string token;
            if (!std::getline(ss, token, ',')) continue;
            data.NSTEP = std::stoi(token);
            if (!std::getline(ss, token, ',')) continue;
            data.SKIPBY = std::stoi(token);

            // Read all arrays of 4 values:
            auto readArrayOf4 = [&](float arr[4]) {
                for (int i = 0; i < 4; i++) {
                    if(!readFloatToken(ss, arr[i])) return false;
                }
                return true;
            };

            if(!readArrayOf4(data.noneGraphTotalTimeWithout)) continue;
            if(!readArrayOf4(data.GraphTotalTimeWithout)) continue;
            if(!readArrayOf4(data.noneGraphTotalTimeWith)) continue;
            if(!readArrayOf4(data.GraphTotalTimeWith)) continue;
            if(!readArrayOf4(data.DiffTotalWithout)) continue;
            if(!readArrayOf4(data.DiffPerStepWithout)) continue;
            if(!readArrayOf4(data.DiffPercentWithout)) continue;
            if(!readArrayOf4(data.DiffTotalWith)) continue;
            if(!readArrayOf4(data.DiffPerStepWith)) continue;
            if(!readArrayOf4(data.DiffPercentWith)) continue;
            if(!readArrayOf4(data.ChronoNoneGraphTotalTimeWithout)) continue;
            if(!readArrayOf4(data.ChronoGraphTotalTimeWithout)) continue;
            if(!readArrayOf4(data.ChronoNoneGraphTotalLaunchTimeWithout)) continue;
            if(!readArrayOf4(data.ChronoGraphTotalLaunchTimeWithout)) continue;
            if(!readArrayOf4(data.ChronoNoneGraphTotalTimeWith)) continue;
            if(!readArrayOf4(data.ChronoGraphTotalTimeWith)) continue;
            if(!readArrayOf4(data.ChronoNoneGraphTotalLaunchTimeWith)) continue;
            if(!readArrayOf4(data.ChronoGraphTotalLaunchTimeWith)) continue;
            if(!readArrayOf4(data.ChronoDiffTotalTimeWithout)) continue;
            if(!readArrayOf4(data.ChronoDiffPerStepWithout)) continue;
            if(!readArrayOf4(data.ChronoDiffPercentWithout)) continue;
            if(!readArrayOf4(data.ChronoDiffTotalTimeWith)) continue;
            if(!readArrayOf4(data.ChronoDiffPerStepWith)) continue;
            if(!readArrayOf4(data.ChronoDiffPercentWith)) continue;
            if(!readArrayOf4(data.ChronoDiffLaunchTimeWithout)) continue;
            if(!readArrayOf4(data.ChronoDiffLaunchPercentWithout)) continue;
            if(!readArrayOf4(data.ChronoDiffLaunchTimeWith)) continue;
            if(!readArrayOf4(data.ChronoDiffLaunchPercentWith)) continue;

            csvData.push_back(data); 
        }
        csvFileIn.close();
    }

    // Update or append
    bool updated = false;
    for (auto &entry : csvData) {
        if (entry.NSTEP == newData.NSTEP && entry.SKIPBY == newData.SKIPBY) {
            entry = newData;
            updated = true;
            break;
        }
    }

    if (!updated) {
        csvData.push_back(newData);
    }

    std::string tempFILENAME = "complex_3_different_kernels.tmp";
    {
        std::ofstream tempFile(tempFILENAME);
        if (!tempFile.is_open()) {
            std::cerr << "Failed to open the temporary file for writing!" << std::endl;
            return;
        }

        tempFile << "NSTEP,SKIPBY,";

        // For each metric, add the four columns with suffixes 1..4
        auto writeCols = [&](const std::string &baseName) {
            for (int i = 1; i <= 4; i++) {
                tempFile << baseName << i << ",";
            }
        };

        writeCols("noneGraphTotalTimeWithout");
        writeCols("GraphTotalTimeWithout");
        writeCols("noneGraphTotalTimeWith");
        writeCols("GraphTotalTimeWith");
        writeCols("DiffTotalWithout");
        writeCols("DiffPerStepWithout");
        writeCols("DiffPercentWithout");
        writeCols("DiffTotalWith");
        writeCols("DiffPerStepWith");
        writeCols("DiffPercentWith");
        writeCols("ChronoNoneGraphTotalTimeWithout");
        writeCols("ChronoGraphTotalTimeWithout");
        writeCols("ChronoNoneGraphTotalLaunchTimeWithout");
        writeCols("ChronoGraphTotalLaunchTimeWithout");
        writeCols("ChronoNoneGraphTotalTimeWith");
        writeCols("ChronoGraphTotalTimeWith");
        writeCols("ChronoNoneGraphTotalLaunchTimeWith");
        writeCols("ChronoGraphTotalLaunchTimeWith");
        writeCols("ChronoDiffTotalTimeWithout");
        writeCols("ChronoDiffPerStepWithout");
        writeCols("ChronoDiffPercentWithout");
        writeCols("ChronoDiffTotalTimeWith");
        writeCols("ChronoDiffPerStepWith");
        writeCols("ChronoDiffPercentWith");
        writeCols("ChronoDiffLaunchTimeWithout");
        writeCols("ChronoDiffLaunchPercentWithout");
        writeCols("ChronoDiffLaunchTimeWith");
        writeCols("ChronoDiffLaunchPercentWith");

        // Remove last comma and add newline
        tempFile.seekp(-1, std::ios_base::cur);
        tempFile << "\n";

        auto writeVals = [&](std::ofstream &out, const float arr[4]) {
            for (int i = 0; i < 4; i++) {
                out << arr[i] << ",";
            }
        };

        for (const auto &entry : csvData) {
            tempFile << entry.NSTEP << "," << entry.SKIPBY << ",";
            writeVals(tempFile, entry.noneGraphTotalTimeWithout);
            writeVals(tempFile, entry.GraphTotalTimeWithout);
            writeVals(tempFile, entry.noneGraphTotalTimeWith);
            writeVals(tempFile, entry.GraphTotalTimeWith);
            writeVals(tempFile, entry.DiffTotalWithout);
            writeVals(tempFile, entry.DiffPerStepWithout);
            writeVals(tempFile, entry.DiffPercentWithout);
            writeVals(tempFile, entry.DiffTotalWith);
            writeVals(tempFile, entry.DiffPerStepWith);
            writeVals(tempFile, entry.DiffPercentWith);
            writeVals(tempFile, entry.ChronoNoneGraphTotalTimeWithout);
            writeVals(tempFile, entry.ChronoGraphTotalTimeWithout);
            writeVals(tempFile, entry.ChronoNoneGraphTotalLaunchTimeWithout);
            writeVals(tempFile, entry.ChronoGraphTotalLaunchTimeWithout);
            writeVals(tempFile, entry.ChronoNoneGraphTotalTimeWith);
            writeVals(tempFile, entry.ChronoGraphTotalTimeWith);
            writeVals(tempFile, entry.ChronoNoneGraphTotalLaunchTimeWith);
            writeVals(tempFile, entry.ChronoGraphTotalLaunchTimeWith);
            writeVals(tempFile, entry.ChronoDiffTotalTimeWithout);
            writeVals(tempFile, entry.ChronoDiffPerStepWithout);
            writeVals(tempFile, entry.ChronoDiffPercentWithout);
            writeVals(tempFile, entry.ChronoDiffTotalTimeWith);
            writeVals(tempFile, entry.ChronoDiffPerStepWith);
            writeVals(tempFile, entry.ChronoDiffPercentWith);
            writeVals(tempFile, entry.ChronoDiffLaunchTimeWithout);
            writeVals(tempFile, entry.ChronoDiffLaunchPercentWithout);
            writeVals(tempFile, entry.ChronoDiffLaunchTimeWith);
            writeVals(tempFile, entry.ChronoDiffLaunchPercentWith);

            tempFile.seekp(-1, std::ios_base::cur);
            tempFile << "\n";
        }
    }

    std::remove(filename.c_str());
    std::rename(tempFILENAME.c_str(), filename.c_str());
    std::cout << "SUCCESS: ADDED/UPDATED CSV FILE\n";
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
}


void runWithGraph(std::vector<float>& totalTimeWithArr, std::vector<float>& totalTimeWithoutArr,
                  std::vector<float>& chronoTotalTimeWithArr, std::vector<float>& chronoTotalTimeWithoutArr,
                  std::vector<float>& chronoTotalLaunchTimeWithArr, std::vector<float>& chronoTotalLaunchTimeWithoutArr,
                  int nstep, int skipby) {
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
}


int main(int argc, char* argv[]) {
    const int NSTEP = (argc > 1) ? atoi(argv[1]) : DEFAULT_NSTEP;
    const int SKIPBY = (argc > 2) ? atoi(argv[2]) : DEFAULT_SKIPBY;

    std::cout << "==============COMPLEX MULTI STREAM KERNELS TEST==============" << std::endl;

    std::vector<int> nsteps = generateSequence(NSTEP);
    const int NUM_RUNS = 4;
    std::vector<CSVData> newDatas(nsteps.size());
    for (auto &newData : newDatas) {
        for (int r = 0; r < NUM_RUNS; r++) {
            newData.noneGraphTotalTimeWithout[r] = 0;
            newData.GraphTotalTimeWithout[r] = 0;
            newData.noneGraphTotalTimeWith[r] = 0;
            newData.GraphTotalTimeWith[r] = 0;
            newData.DiffTotalWithout[r] = 0;
            newData.DiffPerStepWithout[r] = 0;
            newData.DiffPercentWithout[r] = 0;
            newData.DiffTotalWith[r] = 0;
            newData.DiffPerStepWith[r] = 0;
            newData.DiffPercentWith[r] = 0;
            newData.ChronoNoneGraphTotalTimeWithout[r] = 0;
            newData.ChronoGraphTotalTimeWithout[r] = 0;
            newData.ChronoNoneGraphTotalLaunchTimeWithout[r] = 0;
            newData.ChronoGraphTotalLaunchTimeWithout[r] = 0;
            newData.ChronoNoneGraphTotalTimeWith[r] = 0;
            newData.ChronoGraphTotalTimeWith[r] = 0;
            newData.ChronoNoneGraphTotalLaunchTimeWith[r] = 0;
            newData.ChronoGraphTotalLaunchTimeWith[r] = 0;
            newData.ChronoDiffTotalTimeWithout[r] = 0;
            newData.ChronoDiffPerStepWithout[r] = 0;
            newData.ChronoDiffPercentWithout[r] = 0;
            newData.ChronoDiffTotalTimeWith[r] = 0;
            newData.ChronoDiffPerStepWith[r] = 0;
            newData.ChronoDiffPercentWith[r] = 0;
            newData.ChronoDiffLaunchTimeWithout[r] = 0;
            newData.ChronoDiffLaunchPercentWithout[r] = 0;
            newData.ChronoDiffLaunchTimeWith[r] = 0;
            newData.ChronoDiffLaunchPercentWith[r] = 0;
        }
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

    const std::string FILENAME = "complex_multi_stream_kernels.csv";
    for (const auto &newData : newDatas) {
        updateOrAppendCSV(FILENAME, newData);
    }

    return 0;
}
