import numpy as np
import cv2
import csv
import time
from numba import jit, prange

# ------------------------------------------------------------------------------
# NUCLI PARAL·LEL (La màgia de Numba)
# ------------------------------------------------------------------------------
# @jit: Compila la funció a codi màquina (molt ràpid)
# nopython=True: No utilitzis objectes lents de Python
# parallel=True: Activa el paral·lelisme automàtic (Multithreading sense GIL)
# ------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def kmeans_assign_labels(pixels, centroids):
    """
    Pas 1 (Assignació): Calcula el clúster més proper per a cada píxel.
    S'executa en paral·lel a tots els nuclis de la CPU.
    """
    n_pixels = pixels.shape[0]
    n_centroids = centroids.shape[0]
    labels = np.empty(n_pixels, dtype=np.int32)
    
    # prange (Parallel Range): Numba reparteix aquest bucle entre els fils
    for i in prange(n_pixels):
        r = pixels[i, 0]
        g = pixels[i, 1]
        b = pixels[i, 2]
        
        min_dist = 1e30
        best_k = -1
        
        # Comprovem la distància a cada centroide
        for k in range(n_centroids):
            c_r = centroids[k, 0]
            c_g = centroids[k, 1]
            c_b = centroids[k, 2]
            
            # Distància euclidiana al quadrat
            dist = (r - c_r)**2 + (g - c_g)**2 + (b - c_b)**2
            
            if dist < min_dist:
                min_dist = dist
                best_k = k
        
        labels[i] = best_k
        
    return labels

@jit(nopython=True)
def compute_new_centroids(pixels, labels, k):
    """
    Pas 2 (Actualització): Recalcula la posició dels centres.
    Això és molt ràpid, ho fem serial (o Numba ho optimitza sol).
    """
    sum_r = np.zeros(k, dtype=np.float32)
    sum_g = np.zeros(k, dtype=np.float32)
    sum_b = np.zeros(k, dtype=np.float32)
    counts = np.zeros(k, dtype=np.float32)
    
    n_pixels = pixels.shape[0]
    
    for i in range(n_pixels):
        idx = labels[i]
        sum_r[idx] += pixels[i, 0]
        sum_g[idx] += pixels[i, 1]
        sum_b[idx] += pixels[i, 2]
        counts[idx] += 1
        
    new_centroids = np.zeros((k, 3), dtype=np.float32)
    for j in range(k):
        if counts[j] > 0:
            new_centroids[j, 0] = sum_r[j] / counts[j]
            new_centroids[j, 1] = sum_g[j] / counts[j]
            new_centroids[j, 2] = sum_b[j] / counts[j]
            
    return new_centroids

@jit(nopython=True, parallel=True)
def reconstruct_image(pixels, centroids, labels):
    """Reconstrueix la imatge final en paral·lel."""
    n_pixels = pixels.shape[0]
    output = np.empty_like(pixels)
    
    for i in prange(n_pixels):
        k = labels[i]
        output[i, 0] = centroids[k, 0]
        output[i, 1] = centroids[k, 1]
        output[i, 2] = centroids[k, 2]
        
    return output

# ------------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ------------------------------------------------------------------------------
import os

def run_kmeans_parallel_cpu_folder(input_dir, output_dir, k=16, max_iters=20, tol=1e-4, limit=-1):
    print(f"--- Iniciant K-Means amb Numba (CPU Multicore) ---")
    
    if not os.path.exists(input_dir):
        print(f"Error: El directori '{input_dir}' no existeix.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("No s'han trobat imatges.")
        return
        
    if limit != -1:
        image_files = image_files[:limit]
        
    print(f"Processant {len(image_files)} imatges en PARAL·LEL (CPU)...")
    
    csv_file = "parallel_results.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Filename", "ExecutionTime", "K", "MaxIters", "Resolution", "Pixels"])

    # Compilació JIT inicial (Dummy run)
    print("Pre-compilant funcions Numba...")
    dummy_pixels = np.random.rand(100, 3).astype(np.float32)
    dummy_centroids = np.random.rand(k, 3).astype(np.float32)
    _ = kmeans_assign_labels(dummy_pixels, dummy_centroids)
    
    total_start_time = time.time()
    processed_count = 0
    
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 1. Carregar imatge
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error llegint {filename}")
            continue
            
        pixels = img.reshape(-1, 3).astype(np.float32)
        h, w, c = img.shape
        
        # Inicialització
        start_image = time.time()
        np.random.seed(42)
        centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]
        
        # 2. Bucle Principal
        for i in range(max_iters):
            # --- PAS PARAL·LEL (Assignació) ---
            labels = kmeans_assign_labels(pixels, centroids)
            
            # --- PAS ACTUALITZACIÓ ---
            new_centroids = compute_new_centroids(pixels, labels, k)
            
            # Convergència
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            
            if shift < tol:
                break
                
        # 3. Generar Resultat
        final_pixels = reconstruct_image(pixels, centroids, labels)
        final_image = final_pixels.reshape(h, w, c).astype(np.uint8)
        
        elapsed = time.time() - start_image
        print(f"  {filename}: {elapsed:.4f}s")
        
        # Guardar al CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, f"{elapsed:.4f}", k, max_iters, f"{w}x{h}", h*w])

        cv2.imwrite(output_path, final_image)
        processed_count += 1
            
    total_time = time.time() - total_start_time
    print(f"\n--- Resum Paral·lel (CPU) ---")
    print(f"Imatges processades: {processed_count}")
    print(f"Temps total: {total_time:.4f} segons")
    if processed_count > 0:
        print(f"Temps mitjà per imatge: {total_time/processed_count:.4f} segons")
        
    return total_time, processed_count

if __name__ == "__main__":
    try:
        user_input = input("Introdueix el nombre d'imatges a processar (-1 per totes): ")
        limit = int(user_input)
    except ValueError:
        limit = -1
        
    IMAGE_PATH = "archive"
    OUTPUT_PATH = "output_img_parallel"
    
    run_kmeans_parallel_cpu_folder(IMAGE_PATH, OUTPUT_PATH, k=16, limit=limit)