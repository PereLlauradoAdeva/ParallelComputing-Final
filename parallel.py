import numpy as np
import cv2
import csv
import time
import numba
from numba import jit, prange

# ------------------------------------------------------------------------------
# PARALLEL KERNEL (Numba Magic)
# ------------------------------------------------------------------------------
# @jit: Compiles function to machine code (very fast)
# nopython=True: Do not use slow Python objects
# parallel=True: Activates automatic parallelism (Multithreading without GIL)
# ------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def kmeans_assign_labels(pixels, centroids):
    """
    Step 1 (Assignment): Calculate the nearest cluster for each pixel.
    Runs in parallel on all CPU cores.
    """
    n_pixels = pixels.shape[0]
    n_centroids = centroids.shape[0]
    labels = np.empty(n_pixels, dtype=np.int32)
    
    # [False Sharing Documentation]
    # Using `prange` distributes indices `i` among threads.
    # Since `labels` is a contiguous array and each thread writes to a contiguous range
    # (automatic chunking by Numba), "False Sharing" could only occur
    # at chunk boundaries (minimal impact).
    # Therefore, writing to `labels[i]` is safe and efficient.
    
    # prange (Parallel Range): Numba splits this loop among threads
    for i in prange(n_pixels):
        r = pixels[i, 0]
        g = pixels[i, 1]
        b = pixels[i, 2]
        
        min_dist = 1e30
        best_k = -1
        
        # Check distance to each centroid
        for k in range(n_centroids):
            c_r = centroids[k, 0]
            c_g = centroids[k, 1]
            c_b = centroids[k, 2]
            
            # Squared Euclidean distance
            dist = (r - c_r)**2 + (g - c_g)**2 + (b - c_b)**2
            
            if dist < min_dist:
                min_dist = dist
                best_k = k
        
        labels[i] = best_k
        
    return labels

@jit(nopython=True, parallel=True)
def compute_new_centroids(pixels, labels, k, n_threads):
    """
    Step 2 (Update): Recalculate centroid positions in PARALLEL.
    Uses private arrays per thread to avoid race conditions and false sharing.
    """
    n_pixels = pixels.shape[0]
    
    # Calculate padding to ensure each row occupies 64 bytes (8 floats)
    # If k=8 it already takes 64 bytes. If k < 8, force row to be 8.
    pad_k = max(k, 8) 
    
    # Private structures with padding (float64 so each takes 8 bytes)
    priv_sum_r = np.zeros((n_threads, pad_k), dtype=np.float64)
    priv_sum_g = np.zeros((n_threads, pad_k), dtype=np.float64)
    priv_sum_b = np.zeros((n_threads, pad_k), dtype=np.float64)
    priv_counts = np.zeros((n_threads, pad_k), dtype=np.int32)
    
    # Check work manually to use thread index (t)
    chunk_size = (n_pixels + n_threads - 1) // n_threads
    
    for t in prange(n_threads):
        start = t * chunk_size
        end = min(start + chunk_size, n_pixels)
        if start < end:
            for i in range(start, end):
                idx = labels[i]
                priv_sum_r[t, idx] += pixels[i, 0]
                priv_sum_g[t, idx] += pixels[i, 1]
                priv_sum_b[t, idx] += pixels[i, 2]
                priv_counts[t, idx] += 1
                
    # Final reduction (sum partial results from each thread)
    new_centroids = np.zeros((k, 3), dtype=np.float32)
    
    for j in range(k):
        # Sum contributions from all threads
        total_r = np.sum(priv_sum_r[:, j])
        total_g = np.sum(priv_sum_g[:, j])
        total_b = np.sum(priv_sum_b[:, j])
        total_c = np.sum(priv_counts[:, j])
            
        if total_c > 0:
            new_centroids[j, 0] = total_r / total_c
            new_centroids[j, 1] = total_g / total_c
            new_centroids[j, 2] = total_b / total_c
            
    return new_centroids

@jit(nopython=True, parallel=True)
def reconstruct_image(pixels, centroids, labels):
    """Reconstructs the final image in parallel."""
    n_pixels = pixels.shape[0]
    output = np.empty_like(pixels)
    
    for i in prange(n_pixels):
        k = labels[i]
        output[i, 0] = centroids[k, 0]
        output[i, 1] = centroids[k, 1]
        output[i, 2] = centroids[k, 2]
        
    return output

# ------------------------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------------------------
import os

def run_kmeans_parallel_cpu_folder(input_dir, output_dir, k=16, max_iters=20, tol=1e-4, limit=-1, n_threads=None):
    print(f"--- Starting K-Means with Numba (Multicore CPU) ---")
    
    if n_threads is not None:
        numba.set_num_threads(n_threads)
        print(f"   [CONFIG] Forcing {n_threads} threads.")
    else:
        # If not specified, use default (all cores)
        n_threads = numba.get_num_threads()
        print(f"   [CONFIG] Using {n_threads} threads (Default/Env).")
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("No images found.")
        return
        
    if limit != -1:
        image_files = image_files[:limit]
        
    print(f"Processing {len(image_files)} images in PARALLEL (CPU)...")
    
    csv_file = "parallel_results.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Filename", "MeanTime_s", "StdDev_s", "K", "MaxIters", "Resolution", "Pixels"])

    # Initial JIT Compilation (Warm-up complete)
    print("Pre-compiling Numba functions (Warm-up)...")
    dummy_pixels = np.random.rand(100, 3).astype(np.float32)
    dummy_centroids = np.random.rand(k, 3).astype(np.float32)
    dummy_labels = np.zeros(100, dtype=np.int32)
    
    # Run all JIT functions to guarantee compilation
    _ = kmeans_assign_labels(dummy_pixels, dummy_centroids)
    # Pass a dummy n_threads (e.g., 1 or 4) for warm-up or the real one
    _ = compute_new_centroids(dummy_pixels, dummy_labels, k, n_threads if n_threads else 4)
    _ = reconstruct_image(dummy_pixels, dummy_centroids, dummy_labels)
    print("Warm-up complete.")
    
    accumulated_processing_time = 0.0
    accumulated_variance = 0.0
    processed_count = 0
    centroids_dict = {}

    # Determine number of active Numba threads
    # We already have n_threads defined at function start

    
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        # Modify name to include K and Threads to avoid overwriting
        name, ext = os.path.splitext(filename)
        if n_threads is not None:
             output_filename = f"{name}_k{k}_t{n_threads}{ext}"
        else:
             output_filename = f"{name}_k{k}{ext}"
             
        output_path = os.path.join(output_dir, output_filename)
        
        # 1. Load Image (I/O outside timer)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading {filename}")
            continue
            
        pixels = img.reshape(-1, 3).astype(np.float32)
        h, w, c = img.shape
        
        times = []
        final_image = None
        
        # Iterations for statistical stability
        n_iterations = 5
        
        for idx_iter in range(n_iterations):
            # Initialization (part of the algorithm or setup)
            # Usually centroid selection is part of K-Means.
            # We fix seed for consistency.
            # Using 'perf_counter' for high precision.
            
            start_run = time.perf_counter()
            
            np.random.seed(42)
            centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]
            
            # 2. Main K-Means Loop
            labels = None
            for i in range(max_iters):
                # --- PARALLEL STEP (Assignment) ---
                labels = kmeans_assign_labels(pixels, centroids)
                
                # --- UPDATE STEP (NOW PARALLEL) ---
                new_centroids = compute_new_centroids(pixels, labels, k, n_threads)
                
                # Convergence
                shift = np.linalg.norm(new_centroids - centroids)
                centroids = new_centroids
                
                if shift < tol:
                    break
                    
            # 3. Generate Result (Reconstruction is also parallel and part of the process)
            final_pixels = reconstruct_image(pixels, centroids, labels)
            
            end_run = time.perf_counter()
            times.append(end_run - start_run)
            
            # Format image only at the end to save it
            if idx_iter == n_iterations - 1:
                final_image = final_pixels.reshape(h, w, c).astype(np.uint8)
                
                # Calculate INERTIA (parallel/vectorized in numpy)
                # final_pixels and centroids[labels] should be aligned
                # But 'labels' is 1D and 'centroids' has 3 coords.
                # Reconstruct ideal pixels (color of their center)
                # This is already done by 'reconstruct_image' -> final_pixels in float32
                
                diff = pixels - final_pixels # (N, 3)
                inertia = np.sum(diff**2)
                
                centroids_dict[filename] = {"centroids": centroids, "inertia": inertia}

        # Statistical calculations
        mean_time = np.mean(times)
        std_time = np.std(times)
        accumulated_processing_time += mean_time
        accumulated_variance += std_time ** 2

        print(f"  {filename}: {mean_time:.4f}s (+/- {std_time:.4f})")
        
        # Save to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, f"{mean_time:.5f}", f"{std_time:.5f}", k, max_iters, f"{w}x{h}", h*w])

        # Save image (I/O)
        cv2.imwrite(output_path, final_image)
        processed_count += 1
            
    print(f"\n--- Parallel Summary (CPU) ---")
    print(f"Processed images: {processed_count}")
    print(f"Accumulated PROCESS time: {accumulated_processing_time:.4f} seconds")
    if processed_count > 0:
        print(f"Average process time per image: {accumulated_processing_time/processed_count:.4f} seconds")
        
    return accumulated_processing_time, np.sqrt(accumulated_variance), processed_count, centroids_dict

if __name__ == "__main__":
    try:
        user_input = input("Introdueix el nombre d'imatges a processar (-1 per totes): ")
        limit = int(user_input)
    except ValueError:
        limit = -1
        
    IMAGE_PATH = "archive"
    OUTPUT_PATH = "output_img_parallel"
    
    run_kmeans_parallel_cpu_folder(IMAGE_PATH, OUTPUT_PATH, k=16, limit=limit)