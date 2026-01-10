# K-Means Color Quantization Benchmark (CPU vs Parallel)

This project implements a parallelized K-means clustering algorithm for image color quantization using Python and Numba. It compares the performance of a sequential CPU implementation against a parallelized CPU version.

## 📂 Project Structure

- `main.py`: Entry point for the benchmark. Handles configuration, execution, and results logging.
- `parallel.py`: Contains the parallel K-means implementation using `numba` for CPU optimization.
- `sequencial.py`: Contains the standard sequential K-means implementation using `opencv`.
- `data_sample/`: A subset of images for testing purposes.
- `archive/`: (Ignored) Full dataset directory.

## 🚀 Build & Run Instructions

### Prerequisites
- Python 3.12.10+
- Required libraries:
  ```bash
  pip install numpy opencv-python numba
  ```

### Running the Benchmark
1. Clone the repository.
   ```bash
   git clone https://github.com/PereLlauradoAdeva/ParallelComputing-Final.git
   cd ParallelComputing-Final
   ```
2. Run the main script:
   ```bash
   python main.py
   ```
3. Follow the interactive prompts:
   - Enter the number of images to process (e.g., `5` or `-1` for all).
   - The script will execute benchmarks for different K values (8, 16, 32).

## 📊 Parameters

| Parameter | Description | Default / Options |
| :--- | :--- | :--- |
| `limit` | Number of images to process | Input by user (default: 5) |
| `k_values` | Number of color clusters | `[8, 16, 32]` |
| `max_iters` | Maximum K-means iterations | `20` (Parallel), `10` (Sequential) |
| `tol` | Convergence tolerance | `1e-4` |

## 🖥️ System Configuration

To ensure reproducibility and accurate benchmarking, it is recommended to document the hardware specifications:

| Component | Specification |
| :--- | :--- |
| **CPU** | Intel(R) Core(TM) i7-8550U @ 1.80GHz |
| **Cores/Threads** | 4 Physical Cores / 8 Logical Processors |
| **RAM** | 16.0 GB DDR4 |
| **OS** | Windows 10/11 |
| **Python Version** | 3.12.10 |
| **Compiler / JIT** | Numba 0.63.1 (LLVM 14 via llvmlite 0.46.0) |

> **⚠️ Benchmark Best Practices:** 
> * **High-Resolution Timing:** This project uses `time.perf_counter()` to measure execution time with maximum precision, excluding I/O operations.
> * **Warm-up:** A warm-up phase is included to eliminate JIT compilation overhead (Numba) and disk caching effects.
> * **Environment:** Close unnecessary background processes and disable CPU frequency scaling if possible to ensure consistent results.

### ⚠️ Note on Validation Failures

If you see a `FAILED` status with a small difference (e.g., 0.4%), don't worry. This happens because parallel calculations sum numbers in a different order than sequential ones. Tiny rounding differences accumulate and can lead K-Means to a slightly different (but equally valid) solution. It's not a bug, just how floating-point math works.

## 📚 Dataset Citations
The dataset utilized in this project is the **Landscape Pictures** collection, available on Kaggle. Curated by Arnaud58, this dataset features High-Definition images of various landscapes. The rich color diversity and high resolution of these images make them an ideal benchmark for evaluating the efficiency and accuracy of color quantization algorithms.

**Source:** 
*Dataset 1 (Big images):*[Kaggle - Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures?resource=download)
*Dataset 2 (Small images):*[Kaggle - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification?resource=download)

## 👤 Author
**Pere Llauradó Adeva**

---
*UniFi - Parallel Computing for Machine Learning*
