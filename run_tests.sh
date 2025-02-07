#!/usr/bin/env bash

# Set the architecture variable here
OFFLOAD_ARCH="gfx1100"

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
FLAGS="--offload-arch=${OFFLOAD_ARCH}"
COMPILE="${COMPILER} ${FLAGS}"


# Utility files
CSV_UTIL="util/csv_util.cpp"


# Command
COMMAND="${COMPILE} ${CSV_UTIL}"

echo "Compiling combined_sample3_nomalloc.cpp with OFFLOAD_ARCH=${OFFLOAD_ARCH}"
rm -f complex-kernels-test/combined_3diffkernels_singlerun
${COMMAND} complex-kernels-test/combined_sample3_nomalloc.cpp -o complex-kernels-test/combined_3diffkernels_singlerun

echo "Entering complex-kernels-test directory"
cd complex-kernels-test/ || exit 1
echo "Running combined_sample3_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_3diffkernels_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test1_Filename}
cd ..

echo "Compiling combined_diffsize_kernels_nomalloc.cpp with OFFLOAD_ARCH=${OFFLOAD_ARCH}"
rm -f diffsize-kernels-test/combined_diffsize_kernels_singlerun
${COMMAND} diffsize-kernels-test/combined_diffsize_kernels_nomalloc.cpp -o diffsize-kernels-test/combined_diffsize_kernels_singlerun

echo "Entering diffsize-kernels-test directory"
cd diffsize-kernels-test/ || exit 1
echo "Running combined_diffsize_kernels_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_diffsize_kernels_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test2_Filename}
cd ..

echo "Compiling combined_multi_malloc_singlerun.cpp with OFFLOAD_ARCH=${OFFLOAD_ARCH}"
rm -f multi-malloc-test/combined_multi_malloc_singlerun
${COMMAND} multi-malloc-test/combined_multi_malloc_singlerun.cpp -o multi-malloc-test/combined_multi_malloc_singlerun

echo "Entering multi-malloc-test directory"
cd multi-malloc-test/ || exit 1
echo "Running combined_multi_malloc_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_multi_malloc_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test3_Filename}
cd ..

echo "Compiling combined_multi_stream2_nomalloc.cpp with OFFLOAD_ARCH=${OFFLOAD_ARCH}"
rm -f multi-stream-test/combined_multi_stream_singlerun
${COMMAND} multi-stream-test/combined_multi_stream2_nomalloc.cpp -o multi-stream-test/combined_multi_stream_singlerun

echo "Entering multi-stream-test directory"
cd multi-stream-test/ || exit 1
echo "Running combined_multi_stream_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_multi_stream_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test4_Filename}
cd ..

echo "Running generate_plots.sh with NUM_RUNS=$NUM_RUNS"
bash generate_plots.sh "${NUM_RUNS}"
# bash generate_plots_multitests.sh "${NUM_RUNS}"

echo "All steps completed successfully."
