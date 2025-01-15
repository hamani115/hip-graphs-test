# HIP Graph Tests

This repository contains various HIP Graph tests and corresponding scripts to compile, run, and generate plots from the results. HIP Graphs can help optimize performance by reducing CPU overhead and improving GPU utilization on AMD hardware.

## Getting Started

### Prerequisites
- **ROCm toolkit** (ensuring `hipcc` is in your PATH).
- A **C++ compiler** supporting HIP (e.g., GCC on a ROCm-enabled system).
- **Python 3** and any plotting libraries required by your Python scripts (e.g., `matplotlib`, `pandas`).

### Repository Structure (Simplified)
- **`complex-kernels-test/`**  
  Contains the source file `combined_sample3_nomalloc.cpp` and outputs a CSV file (e.g., `complex_3_different_kernels.csv`).
- **`diffsize-kernels-test/`**  
  Contains the source file `combined_diffsize_kernels_nomalloc.cpp` and outputs a CSV file (e.g., `complex_different_sizes_kernels.csv`).
- **`multi-malloc-test/`**  
  Contains the source file `combined_multi_malloc_singlerun.cpp` and outputs a CSV file (e.g., `complex_multi_malloc.csv`).
- **`multi-stream-test/`**  
  Contains the source file `combined_multi_stream2_nomalloc.cpp` and outputs a CSV file (e.g., `complex_multi_stream_kernels.csv`).
- **`plot_generator.py`**  
  A Python script that reads CSV files and generates plots in a designated output directory.
- **`run_tests.sh`**  
  A bash script that compiles each test and runs them sequentially, producing CSV output files.
- **`generate_plots.sh`**  
  A bash script that invokes `plot_generator.py` using the relevant CSV files, saving plots in an output directory.

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/hamani115/hip-graph-tests.git
    cd hip-graph-tests
    ```

2. **Run the tests and generate plots**:
    ```bash
    bash run_tests.sh
    ```
   This single script will:
   - Compile each of the HIP test programs (using `hipcc --offload-arch=<gfx_target>`).
   - Execute them with predefined arguments (generating CSV files).
   - Call `generate_plots.sh`, which runs `plot_generator.py` to create plots from the CSV results.

3. **Check the results**:
   - **CSV files** are generated in each test’s directory (e.g., `complex_3_different_kernels.csv`).
   - **Plots** are stored in a default output directory (e.g., `./output_plots`).

## Customizing the Tests or Plots

- **Modifying Test Parameters**  
  Each test directory (e.g., `complex-kernels-test/`, `diffsize-kernels-test/`) has a `.cpp` file. Look for variables such as problem sizes or steps (e.g., `N`, `NSTEP`) and modify them as needed before recompiling.

- **Adding/Removing CSV Files**  
  If you add new tests producing additional CSV files or remove existing ones, update the `CSV_FILES` array in `generate_plots.sh` accordingly. Any CSV path listed there will be passed to `plot_generator.py`.

- **Using a Different Python Script**  
  If you have multiple Python scripts for plotting, modify `PYTHON_SCRIPT` in `generate_plots.sh` to point to your desired script. Ensure it accepts the same arguments (CSV paths, output directory) or adjust the script accordingly.

- **Changing the Output Directory**  
  By default, plots are saved to `./output_plots`. You can change the `OUTPUT_DIR` variable in `generate_plots.sh` if you want them placed elsewhere.

## Notes

- The **`OFFLOAD_ARCH`** (e.g., `gfx1100`) is set in `run_tests.sh`. Update it to match your target AMD GPU architecture if needed.
- Make sure your environment has the required Python packages to run the plotting scripts.
- For more detailed control, you may compile and run each `.cpp` file separately with `hipcc` and then manually call `plot_generator.py`.

---

*Feel free to open an issue or submit a pull request if you run into problems or wish to contribute improvements.*Below is an example **README** for your HIP code repository. It closely mirrors the CUDA version but adapts instructions for HIP/ROCm, using `hipcc` and `gfx1100`. Adjust paths, filenames, and other specifics as needed.

---

# HIP Graph Tests

This repository contains various HIP Graph tests and corresponding scripts to compile, run, and generate plots from the results. HIP Graphs can help optimize performance by reducing CPU overhead and improving GPU utilization on AMD hardware.

## Getting Started

### Prerequisites
- **ROCm toolkit** (ensuring `hipcc` is in your PATH).
- A **C++ compiler** supporting HIP (e.g., GCC on a ROCm-enabled system).
- **Python 3** and any plotting libraries required by your Python scripts (e.g., `matplotlib`, `pandas`).

### Repository Structure (Simplified)
- **`complex-kernels-test/`**  
  Contains the source file `combined_sample3_nomalloc.cpp` and outputs a CSV file (e.g., `complex_3_different_kernels.csv`).
- **`diffsize-kernels-test/`**  
  Contains the source file `combined_diffsize_kernels_nomalloc.cpp` and outputs a CSV file (e.g., `complex_different_sizes_kernels.csv`).
- **`multi-malloc-test/`**  
  Contains the source file `combined_multi_malloc_singlerun.cpp` and outputs a CSV file (e.g., `complex_multi_malloc.csv`).
- **`multi-stream-test/`**  
  Contains the source file `combined_multi_stream2_nomalloc.cpp` and outputs a CSV file (e.g., `complex_multi_stream_kernels.csv`).
- **`plot_generator.py`**  
  A Python script that reads CSV files and generates plots in a designated output directory.
- **`run_tests.sh`**  
  A bash script that compiles each test and runs them sequentially, producing CSV output files.
- **`generate_plots.sh`**  
  A bash script that invokes `plot_generator.py` using the relevant CSV files, saving plots in an output directory.

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/hip-graph-tests.git
    cd hip-graph-tests
    ```

2. **Run the tests and generate plots**:
    ```bash
    bash run_tests.sh
    ```
   This single script will:
   - Compile each of the HIP test programs (using `hipcc --offload-arch=<gfx_target>`).
   - Execute them with predefined arguments (generating CSV files).
   - Call `generate_plots.sh`, which runs `plot_generator.py` to create plots from the CSV results.

3. **Check the results**:
   - **CSV files** are generated in each test’s directory (e.g., `complex_3_different_kernels.csv`).
   - **Plots** are stored in a default output directory (e.g., `./output_plots`).

## Customizing the Tests or Plots

- **Modifying Test Parameters**  
  Each test directory (e.g., `complex-kernels-test/`, `diffsize-kernels-test/`) has a `.cpp` file. Look for variables such as problem sizes or steps (e.g., `N`, `NSTEP`) and modify them as needed before recompiling.

- **Adding/Removing CSV Files**  
  If you add new tests producing additional CSV files or remove existing ones, update the `CSV_FILES` array in `generate_plots.sh` accordingly. Any CSV path listed there will be passed to `plot_generator.py`.

- **Using a Different Python Script**  
  If you have multiple Python scripts for plotting, modify `PYTHON_SCRIPT` in `generate_plots.sh` to point to your desired script. Ensure it accepts the same arguments (CSV paths, output directory) or adjust the script accordingly.

- **Changing the Output Directory**  
  By default, plots are saved to `./output_plots`. You can change the `OUTPUT_DIR` variable in `generate_plots.sh` if you want them placed elsewhere.

## Notes

- The **`OFFLOAD_ARCH`** (e.g., `gfx1100`) is set in `run_tests.sh`. Update it to match your target AMD GPU architecture if needed.
- Make sure your environment has the required Python packages to run the plotting scripts.
- For more detailed control, you may compile and run each `.cpp` file separately with `hipcc` and then manually call `plot_generator.py`.

---

*Feel free to open an issue or submit a pull request if you run into problems or wish to contribute improvements.*