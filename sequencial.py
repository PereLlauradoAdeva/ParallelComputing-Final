import os
import time
import cv2
import csv
import numpy as np

def kmeans_cpu(image, k=16, max_iters=20, tol=1e-4):
    """
    Sequential K-Means implementation using NumPy.
    
    Args:
        image: Input image (H, W, 3)
        k: Number of clusters (colors)
        max_iters: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        quantized_image: Resulting image with K colors
        centers: Final K centroids (representative colors)
    """
    # Flatten image to a list of pixels (N, 3)
    pixels = image.reshape((-1, 3)).astype(np.float32)
    N = pixels.shape[0]
    
    # 1. Initialization: Select K random points as initial centroids
    np.random.seed(42) # For reproducibility
    random_indices = np.random.choice(N, k, replace=False)
    centroids = pixels[random_indices]
    
    for i in range(max_iters):
        # 2. Assignment: Calculate distance from each pixel to each centroid
        
        # Instead of np.linalg.norm, we compute squared distance manually.
        # This is much faster and makes comparison with Parallel fair.
        diff = pixels[:, np.newaxis] - centroids
        distances = np.sum(diff**2, axis=2) # Squared distance (no root)
        labels = np.argmin(distances, axis=1)
        
        # 3. Update: Calculate new centroids as the mean of assigned points
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            # Optimization: Boolean indexing
            points_in_cluster = pixels[labels == j]
            if len(points_in_cluster) > 0:
                new_centroids[j] = points_in_cluster.mean(axis=0)
            else:
                # Consistency with Parallel: Keep centroid at 0 if empty.
                # This prevents validation discrepancies (PASSED).
                pass
        
        # Check convergence
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            print(f"   Converged at iteration {i+1} with shift {shift:.6f}")
            centroids = new_centroids
            break
            
        centroids = new_centroids
        
    # 4. Image Reconstruction
    # Replace each pixel with its centroid (representative color)
    quantized_pixels = centroids[labels].astype(np.uint8)
    quantized_image = quantized_pixels.reshape(image.shape)
    
    # Calculate INERTIA (Sum of Squared Errors) for robust validation
    # This is the sum of squared distance from each pixel to its centroid.
    # We already have 'labels' and 'pixels', so we can do it efficiently.
    # Recalculate exact final distances:
    final_diff = pixels - centroids[labels]
    inertia = np.sum(final_diff**2)
    
    return quantized_image, centroids, inertia

def process_images(input_dir, output_dir, limit=-1, k=16):
    """
    Process images in the directory.
    limit: Max number of images (-1 for all).
    k: Number of clusters.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("No images found.")
        return
        
    # Apply limit if not -1
    if limit != -1:
        image_files = image_files[:limit]

    print(f"Starting sequential processing of {len(image_files)} images...")
    
    csv_file = "sequential_results.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Filename", "MeanTime_s", "StdDev_s", "K", "MaxIters", "Resolution", "Pixels"])

    # --- WARM-UP (System Cache) ---
    print("[Sequential] Warming up...")
    dummy_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    _ = kmeans_cpu(dummy_img, k=k, max_iters=2)
    print("[Sequential] Warm-up complete.")

    accumulated_processing_time = 0.0
    accumulated_variance = 0.0
    processed_count = 0
    centroids_dict = {}
    
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        # Modify name to include K to avoid overwriting
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_k{k}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing: {filename}...")
        
        # 1. Read Image (I/O) - Outside timer
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Error reading {filename}")
            continue

        # 2. Statistical Execution (Multiple Iterations)
        times = []
        quantized_img = None
        
        # Iterations for statistical stability
        n_iterations = 5 
        
        for i in range(n_iterations):
            # Total Isolation: Time ONLY the algorithm
            start_time = time.perf_counter() # High precision
            
            # --- K-MEANS ALGORITHM ---
            res_img, centers, inertia = kmeans_cpu(img, k=k, max_iters=20)
            # -------------------------
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
            # Save the last iteration as visual result
            if i == n_iterations - 1:
                quantized_img = res_img
                # Save final centroids AND inertia
                centroids_dict[filename] = {"centroids": centers, "inertia": inertia}


        # Statistical calculations
        mean_time = np.mean(times)
        std_time = np.std(times)
        accumulated_processing_time += mean_time
        accumulated_variance += std_time ** 2
        
        print(f"  K-Means Time (Mean 5 iters): {mean_time:.4f}s (+/- {std_time:.4f})")
        
        # Save to CSV
        h, w, c = img.shape
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, f"{mean_time:.5f}", f"{std_time:.5f}", k, 20, f"{w}x{h}", h*w])

        # 3. Save Image (I/O) - Outside timer
        cv2.imwrite(output_path, quantized_img)
        processed_count += 1

    print(f"\n--- Sequential Summary ---")
    print(f"Processed images: {processed_count}")
    print(f"Accumulated PROCESS time: {accumulated_processing_time:.4f} seconds")
    if processed_count > 0:
        print(f"Average process time per image: {accumulated_processing_time/processed_count:.4f} seconds")
        
    # Return pure processing time and accumulated deviation
    return accumulated_processing_time, np.sqrt(accumulated_variance), processed_count, centroids_dict

if __name__ == "__main__":
    try:
        user_input = input("Introdueix el nombre d'imatges a processar (-1 per totes): ")
        limit = int(user_input)
    except ValueError:
        print("Entrada no vàlida. Es processaran totes les imatges per defecte.")
        limit = -1
        
    IMAGE_PATH = "archive"
    OUTPUT_PATH = "output_img"
    
    process_images(IMAGE_PATH, OUTPUT_PATH, limit)
