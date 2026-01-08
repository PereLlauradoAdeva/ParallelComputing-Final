import numpy as np
import cv2
import csv
import time
import numba
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
    
    # [Documentació False Sharing]
    # L'ús de `prange` reparteix els índexs `i` entre els fils.
    # Com que `labels` és un array contigu i cada fil escriu en un rang contigu d'índexs
    # (chunking automàtic de Numba), el "False Sharing" només es podria produir 
    # a les fronteres dels chunks (mínim impacte).
    # Per tant, escriure a `labels[i]` és segur i eficient.
    
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

@jit(nopython=True, parallel=True)
def compute_new_centroids(pixels, labels, k, n_threads):
    """
    Pas 2 (Actualització): Recalcula la posició dels centres en PARAL·LEL.
    Utilitza arrays privats per fil per evitar condicions de carrera i false sharing.
    """
    n_pixels = pixels.shape[0]
    
    # Calculem el padding per assegurar que cada fila ocupi 64 bytes (8 floats)
    # Si k=8 ja ocupa 64 bytes. Si k < 8, forcem que la fila sigui de 8.
    pad_k = max(k, 8) 
    
    # Estructures privades amb padding (float64 per ocupar 8 bytes cadascun)
    priv_sum_r = np.zeros((n_threads, pad_k), dtype=np.float64)
    priv_sum_g = np.zeros((n_threads, pad_k), dtype=np.float64)
    priv_sum_b = np.zeros((n_threads, pad_k), dtype=np.float64)
    priv_counts = np.zeros((n_threads, pad_k), dtype=np.int32)
    
    # Repartim la feina manualment per utilitzar l'índex de fil (t)
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
                
    # Reducció final (suma dels resultats parcials de cada fil)
    new_centroids = np.zeros((k, 3), dtype=np.float32)
    
    for j in range(k):
        # Sumar aportacions de tots els fils
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

def run_kmeans_parallel_cpu_folder(input_dir, output_dir, k=16, max_iters=20, tol=1e-4, limit=-1, n_threads=None):
    print(f"--- Iniciant K-Means amb Numba (CPU Multicore) ---")
    
    if n_threads is not None:
        numba.set_num_threads(n_threads)
        print(f"   [CONFIG] Forçant {n_threads} fils.")
    else:
        # Si no s'especifica, agafar el defecte (tots els cores)
        n_threads = numba.get_num_threads()
        print(f"   [CONFIG] Utilitzant {n_threads} fils (Defecte/Env).")
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
            writer.writerow(["Filename", "MeanTime_s", "StdDev_s", "K", "MaxIters", "Resolution", "Pixels"])

    # Compilació JIT inicial (Warm-up complet)
    print("Pre-compilant funcions Numba (Warm-up)...")
    dummy_pixels = np.random.rand(100, 3).astype(np.float32)
    dummy_centroids = np.random.rand(k, 3).astype(np.float32)
    dummy_labels = np.zeros(100, dtype=np.int32)
    
    # Executar totes les funcions JIT per garantir compilació
    _ = kmeans_assign_labels(dummy_pixels, dummy_centroids)
    # Passem un n_threads fictici (per exemple 1 o 4) pel warm-up o el real
    _ = compute_new_centroids(dummy_pixels, dummy_labels, k, n_threads if n_threads else 4)
    _ = reconstruct_image(dummy_pixels, dummy_centroids, dummy_labels)
    print("Warm-up complet.")
    
    accumulated_processing_time = 0.0
    accumulated_variance = 0.0
    processed_count = 0
    centroids_dict = {}

    # Determinar nombre de fils Numba actius
    # Ja tenim n_threads definit a l'inici de la funció

    
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 1. Carregar imatge (I/O fora del cronòmetre)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error llegint {filename}")
            continue
            
        pixels = img.reshape(-1, 3).astype(np.float32)
        h, w, c = img.shape
        
        times = []
        final_image = None
        
        # Iteracions per estabilitat estadística
        n_iterations = 5
        
        for idx_iter in range(n_iterations):
            # Inicialització (part de l'algorisme o setup)
            # Normalment la tria de centroides és part de K-Means, així que la cronometrem o 
            # utilitzem la mateixa 'seed' per consistència. La deixarem dins per ser justos amb sequencial.
            # NO: Si volem veure estabilitat de la PARAL·LELERITZACIÓ, potser millor fixar la seed sempre igual.
            # Utilitzem 'perf_counter' per alta precisió.
            
            start_run = time.perf_counter()
            
            np.random.seed(42)
            centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]
            
            # 2. Bucle Principal K-Means
            labels = None
            for i in range(max_iters):
                # --- PAS PARAL·LEL (Assignació) ---
                labels = kmeans_assign_labels(pixels, centroids)
                
                # --- PAS ACTUALITZACIÓ (ARA PARAL·LEL) ---
                new_centroids = compute_new_centroids(pixels, labels, k, n_threads)
                
                # Convergència
                shift = np.linalg.norm(new_centroids - centroids)
                centroids = new_centroids
                
                if shift < tol:
                    break
                    
            # 3. Generar Resultat (Reconstrucció també és paral·lela i part del procés)
            final_pixels = reconstruct_image(pixels, centroids, labels)
            
            end_run = time.perf_counter()
            times.append(end_run - start_run)
            
            # Formatem la imatge només al final per guardar-la
            if idx_iter == n_iterations - 1:
                final_image = final_pixels.reshape(h, w, c).astype(np.uint8)
                centroids_dict[filename] = centroids

        # Càlculs estadístics
        mean_time = np.mean(times)
        std_time = np.std(times)
        accumulated_processing_time += mean_time
        accumulated_variance += std_time ** 2

        print(f"  {filename}: {mean_time:.4f}s (+/- {std_time:.4f})")
        
        # Guardar al CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, f"{mean_time:.5f}", f"{std_time:.5f}", k, max_iters, f"{w}x{h}", h*w])

        # Guardar imatge (I/O)
        cv2.imwrite(output_path, final_image)
        processed_count += 1
            
    print(f"\n--- Resum Paral·lel (CPU) ---")
    print(f"Imatges processades: {processed_count}")
    print(f"Temps de PROCÉS acumulat (suma de mitjanes): {accumulated_processing_time:.4f} segons")
    if processed_count > 0:
        print(f"Temps mitjà de procés per imatge: {accumulated_processing_time/processed_count:.4f} segons")
        
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