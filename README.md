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
- Python 3.8+
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

## 📚 Dataset Citations
The dataset utilized in this project is the **Landscape Pictures** collection, available on Kaggle. Curated by Arnaud58, this dataset features High-Definition images of various landscapes. The rich color diversity and high resolution of these images make them an ideal benchmark for evaluating the efficiency and accuracy of color quantization algorithms.

**Source:** [Kaggle - Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures?resource=download)

## 👤 Author
**Pere Llauradó Adeva**

---
*UniFi - Parallel Computing for Machine Learning*
