import os
import time
import cv2
import csv
import numpy as np

def kmeans_cpu(image, k=16, max_iters=10, tol=1e-4):
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
        
        # Implementació eficient en memòria:
        # En lloc de crear una matriu (N, K, 3), calculem distàncies iterant per clústers.
        distances = np.zeros((N, k), dtype=np.float32)
        
        for j in range(k):
             # Distància Euclidiana: || pixel - centroid ||
             # axis=1 calcula la norma per cada fila (píxel)
             diff = pixels - centroids[j]
             distances[:, j] = np.linalg.norm(diff, axis=1)
        
        # Assignar al clúster més proper (índex del mínim)
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
    
    return quantized_image

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
            writer.writerow(["Filename", "ExecutionTime", "K", "MaxIters", "Resolution", "Pixels"])

    total_start_time = time.time()
    
    processed_count = 0
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Processant: {filename}...")
        
        # Llegir imatge
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Error llegint {filename}")
            continue

        # Cronometrar només el procés de K-Means
        start_time = time.time()
        
        # --- K-MEANS ---
        quantized_img = kmeans_cpu(img, k=k, max_iters=10) # Utilizar K dinàmic
        # ---------------
        
        elapsed = time.time() - start_time
        print(f"  Temps K-Means: {elapsed:.4f}s")
        
        # Guardar al CSV
        h, w, c = img.shape
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, f"{elapsed:.4f}", k, 10, f"{w}x{h}", h*w]) # max_iters hardcoded to 10 in logic call

        # Guardar resultat
        cv2.imwrite(output_path, quantized_img)
        processed_count += 1

    total_time = time.time() - total_start_time
    
    print(f"\n--- Resum ---")
    print(f"Imatges processades: {processed_count}")
    print(f"Temps total d'execució: {total_time:.4f} segons")
    if processed_count > 0:
        print(f"Temps mitjà per imatge: {total_time/processed_count:.4f} segons")
        
    return total_time, processed_count

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
