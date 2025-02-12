#!/usr/bin/env bash

# Ensure the script stops on error
set -e

# Set the architecture variable here
ARCH="gfx1100"

# Set the test parameters here
NSTEPS=10000
SKIPBY=0
NUM_RUNS=4

# CSV Filenames
Test1_Filename="complex_3_different_kernels_AMD.csv"
Test2_Filename="complex_different_sizes_kernels_AMD.csv"
Test3_Filename="complex_multi_mallocasync_AMD.csv"
Test4_Filename="complex_multi_stream_kernels_AMD.csv"

# Set Compiler and Flags
COMPILER="hipcc"
FLAGS="--offload-arch=${ARCH} -std=c++20"
COMPILE="${COMPILER} ${FLAGS}"


# Utility files
CSV_UTIL="util/csv_util.cpp"


# Command
COMMAND="${COMPILE} ${CSV_UTIL}"

echo "Compiling for OFFLOAD_ARCH="${OFFLOAD_ARCH}" target AMD GPU architecture"

echo "Compiling combined_3diff_kernels_singlerun.cpp"
rm -f complex-3diff-kernels-test/combined_3diff_kernels_singlerun
${COMMAND} complex-3diff-kernels-test/combined_3diff_kernels_singlerun.cpp -o complex-3diff-kernels-test/combined_3diff_kernels_singlerun

echo "Entering complex-3diff-kernels-test directory"
cd complex-3diff-kernels-test/ || exit 1
echo "Running combined_3diff_kernels_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_3diff_kernels_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test1_Filename}
cd ..

echo "Compiling combined_diffsize_kernels_singlerun.cpp"
rm -f complex-diffsize-kernels-test/combined_diffsize_kernels_singlerun
${COMMAND} complex-diffsize-kernels-test/combined_diffsize_kernels_singlerun.cpp -o complex-diffsize-kernels-test/combined_diffsize_kernels_singlerun

echo "Entering complex-diffsize-kernels-test directory"
cd complex-diffsize-kernels-test/ || exit 1
echo "Running combined_diffsize_kernels_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_diffsize_kernels_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test2_Filename}
cd ..

echo "Compiling combined_multi_mallocasync_singlerun.cpp"
rm -f complex-multi-mallocasync-test/combined_multi_mallocasync_singlerun
${COMMAND} complex-multi-mallocasync-test/combined_multi_mallocasync_singlerun.cpp -o complex-multi-malloc-test/combined_multi_mallocasync_singlerun

echo "Entering complex-multi-mallocasync-test directory"
cd complex-multi-mallocasync-test/ || exit 1
echo "Running combined_multi_mallocasync_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_multi_mallocasync_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test3_Filename}
cd ..

echo "Compiling combined_multi_stream_singlerun.cpp"
rm -f complex-multi-stream-test/combined_multi_stream_singlerun
${COMMAND} complex-multi-stream-test/combined_multi_stream_singlerun.cpp -o complex-multi-stream-test/combined_multi_stream_singlerun

echo "Entering complex-multi-stream-test directory"
cd complex-multi-stream-test/ || exit 1
echo "Running combined_multi_stream_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_multi_stream_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test4_Filename}
cd ..

echo "Running generate_plots.sh with NUM_RUNS=$NUM_RUNS"
bash generate_plots.sh "${NUM_RUNS}"

echo "All steps completed successfully."
