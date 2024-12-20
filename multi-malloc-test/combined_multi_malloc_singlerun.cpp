// Standard headers
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>    // For sqrt in standard deviation calculation
#include <algorithm>
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

__global__ void kernelD(float* arrayD, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayD[x] = sinf(arrayD[x]); }
}

__global__ void kernelE(int* arrayE, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayE[x] += 5; }
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

            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.noneGraphTotalTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.GraphTotalTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.noneGraphTotalTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.GraphTotalTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.DiffTotalWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.DiffPerStepWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.DiffPercentWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.DiffTotalWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.DiffPerStepWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.DiffPercentWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoNoneGraphTotalTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoGraphTotalTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoNoneGraphTotalLaunchTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoGraphTotalLaunchTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoNoneGraphTotalTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoGraphTotalTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoNoneGraphTotalLaunchTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoGraphTotalLaunchTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffTotalTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffPerStepWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffPercentWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffTotalTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffPerStepWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffPercentWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffLaunchTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffLaunchPercentWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffLaunchTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffLaunchPercentWith[i])) break;
            }

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

        if (!false) {
            tempFile << "NSTEP,SKIPBY,";

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

            tempFile.seekp(-1, std::ios_base::cur);
            tempFile << "\n";
        }

        auto writeVals = [&](const float arr[4]) {
            for (int i = 0; i < 4; i++) {
                tempFile << arr[i] << ",";
            }
        };

        for (const auto &entry : csvData) {
            tempFile << entry.NSTEP << "," << entry.SKIPBY << ",";
            writeVals(entry.noneGraphTotalTimeWithout);
            writeVals(entry.GraphTotalTimeWithout);
            writeVals(entry.noneGraphTotalTimeWith);
            writeVals(entry.GraphTotalTimeWith);
            writeVals(entry.DiffTotalWithout);
            writeVals(entry.DiffPerStepWithout);
            writeVals(entry.DiffPercentWithout);
            writeVals(entry.DiffTotalWith);
            writeVals(entry.DiffPerStepWith);
            writeVals(entry.DiffPercentWith);
            writeVals(entry.ChronoNoneGraphTotalTimeWithout);
            writeVals(entry.ChronoGraphTotalTimeWithout);
            writeVals(entry.ChronoNoneGraphTotalLaunchTimeWithout);
            writeVals(entry.ChronoGraphTotalLaunchTimeWithout);
            writeVals(entry.ChronoNoneGraphTotalTimeWith);
            writeVals(entry.ChronoGraphTotalTimeWith);
            writeVals(entry.ChronoNoneGraphTotalLaunchTimeWith);
            writeVals(entry.ChronoGraphTotalLaunchTimeWith);
            writeVals(entry.ChronoDiffTotalTimeWithout);
            writeVals(entry.ChronoDiffPerStepWithout);
            writeVals(entry.ChronoDiffPercentWithout);
            writeVals(entry.ChronoDiffTotalTimeWith);
            writeVals(entry.ChronoDiffPerStepWith);
            writeVals(entry.ChronoDiffPercentWith);
            writeVals(entry.ChronoDiffLaunchTimeWithout);
            writeVals(entry.ChronoDiffLaunchPercentWithout);
            writeVals(entry.ChronoDiffLaunchTimeWith);
            writeVals(entry.ChronoDiffLaunchPercentWith);

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

void runWithoutGraph(std::vector<float>& totalTimeWithArr, std::vector<float>& totalTimeWithoutArr,
                     std::vector<float>& chronoTotalTimeWithArr, std::vector<float>& chronoTotalTimeWithoutArr,
                     std::vector<float>& chronoTotalLaunchTimeWithArr, std::vector<float>& chronoTotalLaunchTimeWithoutArr,
                     int nstep, int skipby) {
    const int NSTEP = nstep;
    const int SKIPBY = skipby;

    constexpr size_t arraySizeA = 1U << 20;
    constexpr size_t arraySizeB = 1U << 18;
    constexpr size_t arraySizeC = 1U << 16;
    constexpr size_t arraySizeD = 1U << 17;
    constexpr size_t arraySizeE = 1U << 19;

    constexpr int threadsPerBlock = 256;

    const int numBlocksA = (int)((arraySizeA + threadsPerBlock - 1) / threadsPerBlock);
    const int numBlocksB = (int)((arraySizeB + threadsPerBlock - 1) / threadsPerBlock);
    const int numBlocksC = (int)((arraySizeC + threadsPerBlock - 1) / threadsPerBlock);
    const int numBlocksD = (int)((arraySizeD + threadsPerBlock - 1) / threadsPerBlock);
    const int numBlocksE = (int)((arraySizeE + threadsPerBlock - 1) / threadsPerBlock);

    constexpr double initValue = 2.0;

    double* d_arrayA;
    int* d_arrayB;
    float* d_arrayD;
    int* d_arrayE;

    double* h_arrayA;
    int* h_arrayB;
    float* h_arrayD;
    int* h_arrayE;
    HIP_CHECK(hipHostMalloc((void**)&h_arrayA, arraySizeA * sizeof(double)));
    HIP_CHECK(hipHostMalloc((void**)&h_arrayB, arraySizeB * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void**)&h_arrayD, arraySizeD * sizeof(float)));
    HIP_CHECK(hipHostMalloc((void**)&h_arrayE, arraySizeE * sizeof(int)));

    for (size_t i = 0; i < arraySizeA; i++) h_arrayA[i] = initValue;
    for (size_t i = 0; i < arraySizeB; i++) h_arrayB[i] = 1;
    for (size_t i = 0; i < arraySizeD; i++) h_arrayD[i] = static_cast<float>(i)*0.01f;
    for (size_t i = 0; i < arraySizeE; i++) h_arrayE[i] = 1;

    hipEvent_t firstCreateStart, firstCreateStop;
    float firstCreateTime = 0.0f;
    HIP_CHECK(hipEventCreate(&firstCreateStart));
    HIP_CHECK(hipEventCreate(&firstCreateStop));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // HIP_CHECK(hipMallocAsync(&d_arrayA, arraySizeA*sizeof(double), stream));
    // HIP_CHECK(hipMallocAsync(&d_arrayB, arraySizeB*sizeof(int), stream));
    // HIP_CHECK(hipMallocAsync(&d_arrayD, arraySizeD*sizeof(float), stream));
    // HIP_CHECK(hipMallocAsync(&d_arrayE, arraySizeE*sizeof(int), stream));

    HIP_CHECK(hipEventRecord(firstCreateStart, stream));
    const auto graphStart = std::chrono::steady_clock::now();

    HIP_CHECK(hipMallocAsync(&d_arrayA, arraySizeA*sizeof(double), stream));
    HIP_CHECK(hipMallocAsync(&d_arrayB, arraySizeB*sizeof(int), stream));
    HIP_CHECK(hipMallocAsync(&d_arrayD, arraySizeD*sizeof(float), stream));
    HIP_CHECK(hipMallocAsync(&d_arrayE, arraySizeE*sizeof(int), stream));

    HIP_CHECK(hipMemcpyAsync(d_arrayA, h_arrayA, arraySizeA*sizeof(double), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_arrayB, h_arrayB, arraySizeB*sizeof(int), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_arrayD, h_arrayD, arraySizeD*sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_arrayE, h_arrayE, arraySizeE*sizeof(int), hipMemcpyHostToDevice, stream));

    hipLaunchKernelGGL(kernelA, dim3(numBlocksA), dim3(threadsPerBlock), 0, stream, d_arrayA, arraySizeA);
    hipLaunchKernelGGL(kernelB, dim3(numBlocksB), dim3(threadsPerBlock), 0, stream, d_arrayB, arraySizeB);
    hipLaunchKernelGGL(kernelC, dim3(numBlocksC), dim3(threadsPerBlock), 0, stream, d_arrayA, d_arrayB, arraySizeC);
    hipLaunchKernelGGL(kernelD, dim3(numBlocksD), dim3(threadsPerBlock), 0, stream, d_arrayD, arraySizeD);
    hipLaunchKernelGGL(kernelE, dim3(numBlocksE), dim3(threadsPerBlock), 0, stream, d_arrayE, arraySizeE);

    HIP_CHECK(hipMemcpyAsync(h_arrayA, d_arrayA, arraySizeA*sizeof(double), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(h_arrayD, d_arrayD, arraySizeD*sizeof(float), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(h_arrayE, d_arrayE, arraySizeE*sizeof(int), hipMemcpyDeviceToHost, stream));

    HIP_CHECK(hipFreeAsync(d_arrayA, stream));
    HIP_CHECK(hipFreeAsync(d_arrayB, stream));
    HIP_CHECK(hipFreeAsync(d_arrayD, stream));
    HIP_CHECK(hipFreeAsync(d_arrayE, stream));

    const auto graphEnd = std::chrono::steady_clock::now();
    HIP_CHECK(hipEventRecord(firstCreateStop, stream));
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

    for (int i = 1; i <= NSTEP; i++) {
        for (size_t j = 0; j < arraySizeA; j++) h_arrayA[j] = initValue;
        for (size_t j = 0; j < arraySizeB; j++) h_arrayB[j] = 1;
        for (size_t j = 0; j < arraySizeD; j++) h_arrayD[j] = static_cast<float>(j)*0.01f;
        for (size_t j = 0; j < arraySizeE; j++) h_arrayE[j] = 1;

        HIP_CHECK(hipEventRecord(execStart, stream));
        const auto start = std::chrono::steady_clock::now();

        HIP_CHECK(hipMallocAsync(&d_arrayA, arraySizeA*sizeof(double), stream));
        HIP_CHECK(hipMallocAsync(&d_arrayB, arraySizeB*sizeof(int), stream));
        HIP_CHECK(hipMallocAsync(&d_arrayD, arraySizeD*sizeof(float), stream));
        HIP_CHECK(hipMallocAsync(&d_arrayE, arraySizeE*sizeof(int), stream));

        HIP_CHECK(hipMemcpyAsync(d_arrayA, h_arrayA, arraySizeA*sizeof(double), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_arrayB, h_arrayB, arraySizeB*sizeof(int), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_arrayD, h_arrayD, arraySizeD*sizeof(float), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_arrayE, h_arrayE, arraySizeE*sizeof(int), hipMemcpyHostToDevice, stream));

        hipLaunchKernelGGL(kernelA, dim3(numBlocksA), dim3(threadsPerBlock), 0, stream, d_arrayA, arraySizeA);
        hipLaunchKernelGGL(kernelB, dim3(numBlocksB), dim3(threadsPerBlock), 0, stream, d_arrayB, arraySizeB);
        hipLaunchKernelGGL(kernelC, dim3(numBlocksC), dim3(threadsPerBlock), 0, stream, d_arrayA, d_arrayB, arraySizeC);
        hipLaunchKernelGGL(kernelD, dim3(numBlocksD), dim3(threadsPerBlock), 0, stream, d_arrayD, arraySizeD);
        hipLaunchKernelGGL(kernelE, dim3(numBlocksE), dim3(threadsPerBlock), 0, stream, d_arrayE, arraySizeE);

        HIP_CHECK(hipMemcpyAsync(h_arrayA, d_arrayA, arraySizeA*sizeof(double), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(h_arrayD, d_arrayD, arraySizeD*sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(h_arrayE, d_arrayE, arraySizeE*sizeof(int), hipMemcpyDeviceToHost, stream));

        HIP_CHECK(hipFreeAsync(d_arrayA, stream));
        HIP_CHECK(hipFreeAsync(d_arrayB, stream));
        HIP_CHECK(hipFreeAsync(d_arrayD, stream));
        HIP_CHECK(hipFreeAsync(d_arrayE, stream));

        const auto end = std::chrono::steady_clock::now();
        HIP_CHECK(hipEventRecord(execStop, stream));
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
                    std::cout << "Kernels: kernelA, kernelB, kernelC, kernelD, kernelE" << std::endl;
                    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
                    std::cout << "Array Sizes: " << arraySizeA << ", " << arraySizeB << ", " << arraySizeC << ", " << arraySizeD << ", " << arraySizeE << std::endl;
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
                    
                    chronoTotalTimeWithArr.push_back((totalTimeWithChrono.count()*1000.0));
                    chronoTotalTimeWithoutArr.push_back(totalTimeChrono.count()*1000.0);
                    chronoTotalLaunchTimeWithArr.push_back((totalLunchTimeWithChrono.count()*1000.0));
                    chronoTotalLaunchTimeWithoutArr.push_back(totalLunchTimeChrono.count()*1000.0);
                    totalTimeWithArr.push_back(totalTimeWith);
                    totalTimeWithoutArr.push_back(totalTime);
                }
            }
        }
    }

    constexpr double expectedA = initValue * 2.0 + 3;
    bool passed = true;
    for (size_t i = 0; i < arraySizeA; i++) {
        if (h_arrayA[i] != expectedA) {
            passed = false;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    // HIP_CHECK(hipFreeAsync(d_arrayA, stream));
    // HIP_CHECK(hipFreeAsync(d_arrayB, stream));
    // HIP_CHECK(hipFreeAsync(d_arrayD, stream));
    // HIP_CHECK(hipFreeAsync(d_arrayE, stream));

    HIP_CHECK(hipEventDestroy(execStart));
    HIP_CHECK(hipEventDestroy(execStop));
    HIP_CHECK(hipEventDestroy(firstCreateStart));
    HIP_CHECK(hipEventDestroy(firstCreateStop));
    HIP_CHECK(hipStreamDestroy(stream));

    HIP_CHECK(hipHostFree(h_arrayA));
    HIP_CHECK(hipHostFree(h_arrayB));
    HIP_CHECK(hipHostFree(h_arrayD));
    HIP_CHECK(hipHostFree(h_arrayE));
}

void runWithGraph(std::vector<float>& totalTimeWithArr, std::vector<float>& totalTimeWithoutArr,
                  std::vector<float>& chronoTotalTimeWithArr, std::vector<float>& chronoTotalTimeWithoutArr,
                  std::vector<float>& chronoTotalLaunchTimeWithArr, std::vector<float>& chronoTotalLaunchTimeWithoutArr,
                  int nstep, int skipby) {
    const int NSTEP = nstep;
    const int SKIPBY = skipby;

    constexpr size_t arraySizeA = 1U << 20;
    constexpr size_t arraySizeB = 1U << 18;
    constexpr size_t arraySizeC = 1U << 16;
    constexpr size_t arraySizeD = 1U << 17;
    constexpr size_t arraySizeE = 1U << 19;
    constexpr int threadsPerBlock = 256;

    const int numBlocksA = (int)((arraySizeA + threadsPerBlock - 1)/threadsPerBlock);
    const int numBlocksB = (int)((arraySizeB + threadsPerBlock - 1)/threadsPerBlock);
    const int numBlocksC = (int)((arraySizeC + threadsPerBlock - 1)/threadsPerBlock);
    const int numBlocksD = (int)((arraySizeD + threadsPerBlock - 1)/threadsPerBlock);
    const int numBlocksE = (int)((arraySizeE + threadsPerBlock - 1)/threadsPerBlock);

    constexpr double initValue = 2.0;

    double* d_arrayA;
    int* d_arrayB;
    float* d_arrayD;
    int* d_arrayE;

    double* h_arrayA;
    int* h_arrayB;
    float* h_arrayD;
    int* h_arrayE;

    HIP_CHECK(hipHostMalloc((void**)&h_arrayA, arraySizeA*sizeof(double)));
    HIP_CHECK(hipHostMalloc((void**)&h_arrayB, arraySizeB*sizeof(int)));
    HIP_CHECK(hipHostMalloc((void**)&h_arrayD, arraySizeD*sizeof(float)));
    HIP_CHECK(hipHostMalloc((void**)&h_arrayE, arraySizeE*sizeof(int)));

    for (size_t i = 0; i < arraySizeA; i++) h_arrayA[i] = initValue;
    for (size_t i = 0; i < arraySizeB; i++) h_arrayB[i] = 1;
    for (size_t i = 0; i < arraySizeD; i++) h_arrayD[i] = static_cast<float>(i)*0.01f;
    for (size_t i = 0; i < arraySizeE; i++) h_arrayE[i] = 1;

    hipStream_t captureStream;
    HIP_CHECK(hipStreamCreate(&captureStream));

    hipEvent_t graphCreateStart, graphCreateStop;
    float graphCreateTime = 0.0f;
    HIP_CHECK(hipEventCreate(&graphCreateStart));
    HIP_CHECK(hipEventCreate(&graphCreateStop));

    // HIP_CHECK(hipMallocAsync(&d_arrayA, arraySizeA*sizeof(double), captureStream));
    // HIP_CHECK(hipMallocAsync(&d_arrayB, arraySizeB*sizeof(int), captureStream));
    // HIP_CHECK(hipMallocAsync(&d_arrayD, arraySizeD*sizeof(float), captureStream));
    // HIP_CHECK(hipMallocAsync(&d_arrayE, arraySizeE*sizeof(int), captureStream));

    HIP_CHECK(hipEventRecord(graphCreateStart, captureStream));
    const auto graphStart = std::chrono::steady_clock::now();

    HIP_CHECK(hipMallocAsync(&d_arrayA, arraySizeA*sizeof(double), captureStream));
    HIP_CHECK(hipMallocAsync(&d_arrayB, arraySizeB*sizeof(int), captureStream));
    HIP_CHECK(hipMallocAsync(&d_arrayD, arraySizeD*sizeof(float), captureStream));
    HIP_CHECK(hipMallocAsync(&d_arrayE, arraySizeE*sizeof(int), captureStream));

    HIP_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));

    HIP_CHECK(hipMemcpyAsync(d_arrayA, h_arrayA, arraySizeA*sizeof(double), hipMemcpyHostToDevice, captureStream));
    HIP_CHECK(hipMemcpyAsync(d_arrayB, h_arrayB, arraySizeB*sizeof(int), hipMemcpyHostToDevice, captureStream));
    HIP_CHECK(hipMemcpyAsync(d_arrayD, h_arrayD, arraySizeD*sizeof(float), hipMemcpyHostToDevice, captureStream));
    HIP_CHECK(hipMemcpyAsync(d_arrayE, h_arrayE, arraySizeE*sizeof(int), hipMemcpyHostToDevice, captureStream));

    hipLaunchKernelGGL(kernelA, dim3(numBlocksA), dim3(threadsPerBlock), 0, captureStream, d_arrayA, arraySizeA);
    hipLaunchKernelGGL(kernelB, dim3(numBlocksB), dim3(threadsPerBlock), 0, captureStream, d_arrayB, arraySizeB);
    hipLaunchKernelGGL(kernelC, dim3(numBlocksC), dim3(threadsPerBlock), 0, captureStream, d_arrayA, d_arrayB, arraySizeC);
    hipLaunchKernelGGL(kernelD, dim3(numBlocksD), dim3(threadsPerBlock), 0, captureStream, d_arrayD, arraySizeD);
    hipLaunchKernelGGL(kernelE, dim3(numBlocksE), dim3(threadsPerBlock), 0, captureStream, d_arrayE, arraySizeE);

    HIP_CHECK(hipMemcpyAsync(h_arrayA, d_arrayA, arraySizeA*sizeof(double), hipMemcpyDeviceToHost, captureStream));
    HIP_CHECK(hipMemcpyAsync(h_arrayD, d_arrayD, arraySizeD*sizeof(float), hipMemcpyDeviceToHost, captureStream));
    HIP_CHECK(hipMemcpyAsync(h_arrayE, d_arrayE, arraySizeE*sizeof(int), hipMemcpyDeviceToHost, captureStream));

    HIP_CHECK(hipFreeAsync(d_arrayA, captureStream));
    HIP_CHECK(hipFreeAsync(d_arrayB, captureStream));
    HIP_CHECK(hipFreeAsync(d_arrayD, captureStream));
    HIP_CHECK(hipFreeAsync(d_arrayE, captureStream));

    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(captureStream, &graph));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));

    const auto graphEnd = std::chrono::steady_clock::now();
    HIP_CHECK(hipEventRecord(graphCreateStop, captureStream));
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
    std::chrono::duration<double> totalTimeChrono(0.0);
    std::chrono::duration<double> totalLunchTimeChrono(0.0);
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    std::vector<int> nsteps = generateSequence(NSTEP);

    for (int i = 1; i <= NSTEP; i++) {
        for (size_t j = 0; j < arraySizeA; j++) h_arrayA[j] = initValue;
        for (size_t j = 0; j < arraySizeB; j++) h_arrayB[j] = 1;
        for (size_t j = 0; j < arraySizeD; j++) h_arrayD[j] = static_cast<float>(j)*0.01f;
        for (size_t j = 0; j < arraySizeE; j++) h_arrayE[j] = 1;

        HIP_CHECK(hipEventRecord(execStart, captureStream));
        const auto start = std::chrono::steady_clock::now();

        HIP_CHECK(hipGraphLaunch(graphExec, captureStream));

        const auto end = std::chrono::steady_clock::now();
        HIP_CHECK(hipEventRecord(execStop, captureStream));
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

            for (auto num : nsteps) {
                if (num == i) {
                    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
                    double stdDevTime = sqrt(varianceTime);

                    std::cout << "=======Setup (With Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "Iterations: " << i << std::endl;
                    std::cout << "Skip By: " << SKIPBY << std::endl;
                    std::cout << "Kernels: kernelA, kernelB, kernelC, kernelD, kernelE" << std::endl;
                    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
                    std::cout << "Array Sizes: " << arraySizeA << ", " << arraySizeB << ", " << arraySizeC << ", " << arraySizeD << ", " << arraySizeE << std::endl;
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

                    chronoTotalTimeWithArr.push_back((totalTimeWithChrono.count()*1000.0));
                    chronoTotalTimeWithoutArr.push_back(totalTimeChrono.count()*1000.0);
                    chronoTotalLaunchTimeWithArr.push_back((totalLunchTimeWithChrono.count()*1000.0));
                    chronoTotalLaunchTimeWithoutArr.push_back(totalLunchTimeChrono.count()*1000.0);
                    totalTimeWithArr.push_back(totalTimeWith);
                    totalTimeWithoutArr.push_back(totalTime);
                }
            }
        }
    }

    constexpr double expectedA = initValue * 2.0 + 3;
    bool passed = true;
    for (size_t i = 0; i < arraySizeA; i++) {
        if (h_arrayA[i] != expectedA) {
            passed = false;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    // HIP_CHECK(hipFreeAsync(d_arrayA, captureStream));
    // HIP_CHECK(hipFreeAsync(d_arrayB, captureStream));
    // HIP_CHECK(hipFreeAsync(d_arrayD, captureStream));
    // HIP_CHECK(hipFreeAsync(d_arrayE, captureStream));

    HIP_CHECK(hipEventDestroy(execStart));
    HIP_CHECK(hipEventDestroy(execStop));
    HIP_CHECK(hipEventDestroy(graphCreateStart));
    HIP_CHECK(hipEventDestroy(graphCreateStop));
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipStreamDestroy(captureStream));

    HIP_CHECK(hipHostFree(h_arrayA));
    HIP_CHECK(hipHostFree(h_arrayB));
    HIP_CHECK(hipHostFree(h_arrayD));
    HIP_CHECK(hipHostFree(h_arrayE));
}


int main(int argc, char* argv[]) {
    const int NSTEP = (argc > 1) ? atoi(argv[1]) : DEFAULT_NSTEP;
    const int SKIPBY = (argc > 2) ? atoi(argv[2]) : DEFAULT_SKIPBY;

    std::cout << "==============COMPLEX MULTIPLE MALLOCAYSNC/FREEASYNC TEST==============" << std::endl;
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

            float chronoDiffPerStepWith = chronoDiffTotalTimeWith / (nsteps[i] + 1); 
            float chronoDiffPercentWith = (chronoDiffTotalTimeWith / chronoNoneGraphTotalTimeWithArr[i]) * 100;

            float chronoDiffPerStepWithout = chronoDiffTotalTimeWithout / (nsteps[i]); 
            float chronoDiffPercentWithout = (chronoDiffTotalTimeWithout / chronoNoneGraphTotalTimeWithoutArr[i]) * 100;

            float chronoDiffLaunchTimeWith = chronoNoneGraphTotalLaunchTimeWithArr[i] - chronoGraphTotalLaunchTimeWithArr[i];
            float chronoDiffLaunchTimeWithout = chronoNoneGraphTotalLaunchTimeWithoutArr[i] - chronoGraphTotalLaunchTimeWithoutArr[i];

            float chronoDiffLaunchPercentWithout = (chronoDiffLaunchTimeWithout / chronoNoneGraphTotalLaunchTimeWithoutArr[i]) * 100;
            float chronoDiffLaunchPercentWith = (chronoDiffLaunchTimeWith / chronoNoneGraphTotalLaunchTimeWithArr[i]) * 100;

            std::cout << "==============For NSTEP "<< nsteps[i] << "==============" << std::endl;
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

    const std::string FILENAME = "complex_multi_malloc.csv";
    for (const auto &newData : newDatas) {
        updateOrAppendCSV(FILENAME, newData);
    }

    return 0;
}