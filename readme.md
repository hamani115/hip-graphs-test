# CUDA Graph Tests

This repository contains tests and examples for HIP Graphs. HIP Graphs provide a mechanism to capture and replay a sequence of GPU operations, which can help optimize performance by reducing CPU overhead and improving GPU utilization.


## Getting Started

### Prerequisites

- HIP CC
- C++ compiler (e.g., GCC)

### Building the Project

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/cuda-graph-tests.git
    cd cuda-graph-tests
    ```

2. Navigate to one of the test directories
    ```sh
    cd multi-kernel-test
    ```
    Available tests are `memcpy-device-test`, `memcpy-host-test`, and `multi-kernel-test`.

3. Compile the files with prefered naming using `nvcc` compiler:
    ```sh
    hipcc graph_matrix multiply.cpp -o graph_matrix multiply
    ```

4. Run the compiled file and check the result output:
    ```sh
    ./graph_matrix multiply
    ```
### Code Variables for Testing

1. multi-kernel-test
    - modify `skipBy` value to change the number of graph launches time excluded from "Time Spread" at the start (`default` 0 skips).
    - modify `NKERNEL` value to change the number of kernel launches in the graph instance.
    - modify `NSTEP` value to change the number of graph launches.
    - modify `N` value to change the number of dimensions of the matrix.
2. memcpy-device-test
    - modify `skipBy` value to change the number graph launches to skip at the start (`default` 100 skips).
    - modify `NSTEP` value to change the number of graph launches.
    - modify `N` value to change the number of elements in the array.
3. memcpy-host-test
    - modify `skipBy` value to change the number graph launches to skip at the start (`default` 100 skips).
    - modify `NSTEP` value to change the number of graph launches.
    - modify `N` value to change the number of elements in the array.

