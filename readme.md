# HIP Graph Tests

This repository contains various HIP Graph tests and corresponding scripts to compile, run, and generate plots from the tests results. 

"HIP graphs are an alternative way of executing tasks on a GPU that can provide performance benefits over launching kernels using the standard method via streams. A HIP graph is made up of nodes and edges. The nodes of a HIP graph represent the operations performed, while the edges mark dependencies between those operations." [AMD HIP Docs](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/how-to/hip_runtime_api/hipgraph.html)

HIP Graphs can help optimize performance by reducing CPU overhead and improving GPU utilization on AMD hardware.

## Getting Started

### Prerequisites
- Machine with **AMD GPU** that support **ROCm** (refer to [Supported GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/reference/system-requirements.html))
- **ROCm toolkit** (ensuring `hipcc` is in your PATH. Refer to [Install HIP Guide](https://rocm.docs.amd.com/projects/HIP/en/develop/install/install.html)).
- A **C++ compiler** supporting HIP (e.g., GCC on a ROCm-enabled system).
- **Python 3** and following plotting libraries: `matplotlib`, `pandas`, & `numpy` (for calculation and plotting).

### Repository Structure (Simplified)
- **`complex-3diff-kernels-test/`**  
  Contains the source file `combined_3diff_kernels_singlerun.cpp` and outputs a CSV file (e.g., `complex_3_different_kernels_AMD.csv`).
- **`compelx-diffsize-kernels-test/`**  
  Contains the source file `combined_diffsize_kernels_singlerun.cpp` and outputs a CSV file (e.g., `complex_different_sizes_kernels_AMD.csv`).
- **`complex-multi-malloc-test/`**  
  Contains the source file `combined_multi_malloc_singlerun.cpp` and outputs a CSV file (e.g., `complex_multi_malloc_AMD.csv`).
- **`complex-multi-stream-test/`**  
  Contains the source file `combined_multi_stream_singlerun.cpp` and outputs a CSV file (e.g., `complex_multi_stream_kernels_AMD.csv`).
- **`run_tests.sh`**  
  A bash script that compiles each test and runs them sequentially, producing CSV output files. Additionally, execute `generate_plots.sh` script at the end.
- **`generate_plots.sh`**  
  A bash script that invokes `plot_generator.py` using the relevant CSV files, saving plots in an output directory.
- **`plot_generator.py`**  
  A Python script that reads CSV files and generates plots in a designated output directory.

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/hamani115/hip-graphs-tests.git
    cd hip-graphs-tests
    ```

2. **Run the tests and generate plots**:
  ```bash
  bash run_tests.sh
  ```
  This single script will:
  - Compile each of the HIP test programs (located in its respective directory) using `hipcc --offload-arch=<gfx_target>`.
  - Execute the compiled files with predefined arguments, generating CSV files.
  - Call `generate_plots.sh` with the required arguments, which then execute `plot_generator.py` script to create plots from the CSV results.

3. **Examine the results**:
  - **CSV files** are generated in each test’s directory (e.g., `complex_3_different_kernels_AMD.csv`).
  - **Plots** are saved to the `./output_plots_AMD/` folder by default.

## Customizing the Tests or Plots

- ### **Modifying Test Parameters**  
  In the **`run_test.sh`** bash script, there are variables for the test paramters that are passed as arguments to the executable file of each test.
  1. **`NSTEPS`**: The number of initial iterations that the HIP Graph will be launched (default).
  2. **`SKIPBY`**: The number of initial iterations to skip from timing measurements.
  3. **`NUM_RUNS`**: The number of times the test will be repeated. This affects the average and standard deviation calculations, which in turn influence the plotted values and error bars.
  4. **`Test{N}_Filename`**: The name of the CSV file where the test results will be written. There exist such variable for each test and numbered in place of `{N}` (e.g. `Test2_Filename="complex_different_sizes_kernels_AMD.csv"`).

  **Note**: Each test directory (e.g., `complex-kernels-test/`, `diffsize-kernels-test/`) contains a `.cpp` source file and a corresponding executable (generated after compilation).

- ### **Modifying Compiler Flags**
  in the **`run_test.sh`** bash script, there are variables for the `hipcc` compiler flags that can be modified (currently only one variable).
  1. **`ARCH`**: Specifies the AMD GPU target architecture to compile for using `--offload-arch=<gfx_target>` flag (e.g., `gfx1100` for AMD Radeon Pro W7800 GPU).  

- ### **Adding/Removing Tests** 
  If new tests are added that produce additional CSV files or existing ones wanted remove, add/remove compilation and executing instructions and their corresponding parameters variable from the `run_tests.sh` script.

  Example of compilation and executing instructions from `run_tests.sh`: 
  ```bash
  echo "Compiling combined_3diff_kernels_singlerun.cu"
  rm -f complex-3diff-kernels-test/combined_3diff_kernels_singlerun
  ${COMMAND} complex-3diff-kernels-test/combined_3diff_kernels_singlerun.cu -o complex-3diff-kernels-test/combined_3diff_kernels_singlerun

  echo "Entering complex-kernels-test directory"
  cd complex-3diff-kernels-test/ || exit 1
  echo "Running combined_3diff_kernels_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
  ./combined_3diff_kernels_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test1_Filename}
  cd ..
  ```

- ### **Adding/Removing CSV Files**  
  To produce plots for new CSV files or remove existing ones, update the `CSV_FILES` array in `generate_plots.sh` accordingly. Each CSV file listed there will be passed to `plot_generator.py` to generate the corresponding plots.

- ### **Using a Different Python Script**  
  If you have multiple Python scripts for plotting, modify **`PYTHON_SCRIPT`** in **`generate_plots.sh`** to point to your desired script. Ensure it accepts the same arguments (CSV paths, output directory) or adjust the script accordingly.

  There exists other python scripts other than the default script which can used:
  - `plot_generator.py`(default): Produce set plot for every csv file (test result) passed seperatly.
  - `plot_multitests_generator.py`: Produce set plots for every set of csv files for same test but different GPUs.
  - `plot_combined_generator.py`: Produce set plots for every set of csv files for different test but same GPUs.

  All python scripts have the same arguments passed:
  1. **`csv_files`**: Path(s) to the input CSV file(s). 
  2. **`-o`**, **`--output`**: Directory to save the output plots. Defaults to "./plots".
  3. **`--num_runs`**: Number of runs for each measurement column (default: 4).
  
  **Note**: For csv files passed to `plot_multigpus_generator.py` and `plot_multitests_generator.py` via **`csv_files`** parameter, the scripts depends on the text between the last underscore (`_`) and the file extension `.csv` in each csv filename to distinguish the csv files into different sets. for example, the python script will identify the `complex_different_sizes_kernels_AMD.csv` file as csv file that belongs to "AMD" set so line plots for this set will have a different color. Note, name of the set and color can be changed in the python script variable **`GPU_COLOR`**.


- ### **Changing the Output Directory**  
  By default, plots are saved to `./output_plots_AMD`. You can change the `OUTPUT_DIR` variable in `generate_plots.sh` if you want them placed elsewhere.

## Notes

- The **`--offload-arch`** flag (e.g., `gfx1100`) is set in `run_tests.sh` via **`ARCH`** variable. Update it to match your target AMD GPU architecture if needed.
- Ensure that your environment has the required Python packages to run the plotting scripts.
- For more detailed control, you may compile and run each `.cpp` file separately with `hipcc` and then manually call `plot_generator.py` with the produce CSV file.

---

*Feel free to open a GitHub issue or submit a PR if you encounter problems or wish to contribute improvements.*