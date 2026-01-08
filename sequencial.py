import os
import time
import cv2
import csv
import numpy as np

def kmeans_cpu(image, k=16, max_iters=20, tol=1e-4):
    """
    Implementació seqüencial de K-Means utilitzant NumPy.
    
    Args:
        image: Imatge d'entrada (H, W, 3)
        k: Nombre de clústers (colors)
        max_iters: Màxim d'iteracions
        tol: Tolerància per convergència
    
    Returns:
        quantized_image: Imatge resultant amb K colors
        centers: Els K centroides finals (colors representatius)
    """
    # Aplanar la imatge a una llista de píxels (N, 3)
    pixels = image.reshape((-1, 3)).astype(np.float32)
    N = pixels.shape[0]
    
    # 1. Inicialització: Seleccionar K punts aleatoris com a centroides inicials
    np.random.seed(42) # Per reproductibilitat
    random_indices = np.random.choice(N, k, replace=False)
    centroids = pixels[random_indices]
    
    for i in range(max_iters):
        # 2. Assignació: Calcular distància de cada píxel a cada centroide
        # 2. Assignació: Calcular distància de cada píxel a cada centroide
        
        # En lloc de np.linalg.norm, fem la distància al quadrat manualment
        # Això és molt més ràpid i fa que la comparació amb Parallel sigui justa.
        diff = pixels[:, np.newaxis] - centroids
        distances = np.sum(diff**2, axis=2) # Distància al quadrat (sense arrel)
        labels = np.argmin(distances, axis=1)
        
        # 3. Actualització: Calcular nous centroides com la mitjana dels punts assignats
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            # Optimització: Boolean indexing
            points_in_cluster = pixels[labels == j]
            if len(points_in_cluster) > 0:
                new_centroids[j] = points_in_cluster.mean(axis=0)
            else:
                # Si un clúster es queda buit, re-inicialitzar aleatòriament (opcional)
                new_centroids[j] = pixels[np.random.choice(N)]
        
        # Comprovar convergència
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            print(f"   Converged at iteration {i+1} with shift {shift:.6f}")
            centroids = new_centroids
            break
            
        centroids = new_centroids
        
    # 4. Reconstrucció de la imatge
    # Substituir cada píxel pel seu centroide (color representatiu)
    quantized_pixels = centroids[labels].astype(np.uint8)
    quantized_image = quantized_pixels.reshape(image.shape)
    
    return quantized_image, centroids

def process_images(input_dir, output_dir, limit=-1, k=16):
    """
    Processa les imatges del directori.
    limit: Nombre màxim d'imatges a processar (-1 per totes).
    k: Nombre de clústers.
    """
    if not os.path.exists(input_dir):
        print(f"Error: El directori '{input_dir}' no existeix.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("No s'han trobat imatges.")
        return
        
    # Aplicar el límit si és diferent de -1
    if limit != -1:
        image_files = image_files[:limit]

    print(f"Iniciant processament seqüencial de {len(image_files)} imatges...")
    
    csv_file = "sequential_results.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Filename", "MeanTime_s", "StdDev_s", "K", "MaxIters", "Resolution", "Pixels"])

    # --- WARM-UP (Cache del Sistema) ---
    print("[Seuencial] Fent Warm-up...")
    dummy_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    _ = kmeans_cpu(dummy_img, k=k, max_iters=2)
    print("[Sequencial] Warm-up complet.")

    accumulated_processing_time = 0.0
    accumulated_variance = 0.0
    processed_count = 0
    centroids_dict = {}
    
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Processant: {filename}...")
        
        # 1. Llegir imatge (I/O) - Fora del cronòmetre
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Error llegint {filename}")
            continue

        # 2. Execució Estadística (Multiles Iteracions)
        times = []
        quantized_img = None
        
        # Iteracions per estabilitat estadística
        n_iterations = 5 
        
        for i in range(n_iterations):
            # Aïllament total: Cronometrar NOMÉS l'algorisme
            start_time = time.perf_counter() # Alta precisió
            
            # --- ALGORISME K-MEANS ---
            res_img, centers = kmeans_cpu(img, k=k, max_iters=20)
            # -------------------------
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
            # Guardem l'última iteració com a resultat visual
            if i == n_iterations - 1:
                quantized_img = res_img
                # Save final centroids
                centroids_dict[filename] = centers


        # Càlculs estadístics
        mean_time = np.mean(times)
        std_time = np.std(times)
        accumulated_processing_time += mean_time
        accumulated_variance += std_time ** 2
        
        print(f"  Temps K-Means (Mitjana 5 iters): {mean_time:.4f}s (+/- {std_time:.4f})")
        
        # Guardar al CSV
        h, w, c = img.shape
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, f"{mean_time:.5f}", f"{std_time:.5f}", k, 20, f"{w}x{h}", h*w])

        # 3. Guardar imatge (I/O) - Fora del cronòmetre
        cv2.imwrite(output_path, quantized_img)
        processed_count += 1

    print(f"\n--- Resum Sequencial ---")
    print(f"Imatges processades: {processed_count}")
    print(f"Temps de PROCÉS acumulat (suma de mitjanes): {accumulated_processing_time:.4f} segons")
    if processed_count > 0:
        print(f"Temps mitjà de procés per imatge: {accumulated_processing_time/processed_count:.4f} segons")
        
    # Returnem el temps pur de processament i la desviació acumulada
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
