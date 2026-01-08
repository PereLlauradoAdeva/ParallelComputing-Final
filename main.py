import os
import time
import csv
import numba
import numpy as np
from sequencial import process_images as run_sequential
from parallel import run_kmeans_parallel_cpu_folder as run_parallel

def main():
    print("=================================================================")
    print("      BENCHMARK K-MEANS COLOR QUANTIZATION (CPU vs PARALLEL)     ")
    print("=================================================================")
    
    # 1. Configuració de l'experiment (DATASETS)
    DATASETS = [
        {"name": "Full_Res", "path": "archive"},
        {"name": "Small_Res", "path": "archive2/seg_pred/seg_pred"}
    ]
    
    OUTPUT_DIR_SEQ = "output_seq_bench"
    OUTPUT_DIR_PAR = "output_par_bench"
    
    k_values = [8, 16, 32] 
    
    # Detectar el màxim de threads disponibles per evitar errors de Numba
    max_threads = numba.config.NUMBA_NUM_THREADS
    all_thread_counts = [1, 2, 4, 8, 16]
    thread_counts = [t for t in all_thread_counts if t <= max_threads]
    
    print(f"[INFO] Màxim threads detectats: {max_threads}")
    print(f"[INFO] Thread counts efectius: {thread_counts}")

    print(f"\n[INFO] Iniciant Benchmark Multi-Dataset")

    # --- BUCLE DE DATASETS ---
    for dataset in DATASETS:
        d_name = dataset["name"]
        d_path = dataset["path"]
        
        print(f"\n=================================================================")
        print(f"   DATASET: {d_name}  (Path: {d_path})")
        print(f"=================================================================")
        
        if not os.path.exists(d_path):
            print(f"[ALERTA] El directori '{d_path}' no existeix. Saltant...")
            continue

        # Obtenir el nombre d'imatges a processar PER AQUEST DATASET
        try:
            limit_input = input(f"\n[CONFIG] Nombre d'imatges per '{d_name}' (-1 = totes): ")
            limit = int(limit_input)
        except ValueError:
            limit = 5 
            print(f"   [WARN] Entrada invàlida. Usant default {limit}.")
            
        # Reiniciem resultats per cada dataset per tenir CSVs nets i separats
        results = []
        
        INPUT_DIR = d_path # El codi original fa servir INPUT_DIR variable

        print(f"\n[INFO] Processant '{d_name}' amb limit={limit}")
        
        # 2. Bucle de Tests
        for k in k_values:
            print(f"\n-----------------------------------------------------------------")
            print(f" EXPERIMENT: K (Clusters) = {k}")
            print(f"-----------------------------------------------------------------")
            
            # --- EXECUCIÓ SEQÜENCIAL ---
            print(f"\n>> Executant SEQÜENCIAL (K={k})...")
            t_seq, std_seq, n_seq, centroids_seq_dict = run_sequential(INPUT_DIR, OUTPUT_DIR_SEQ, limit=limit, k=k)
            
            # --- Bucle de FILS (SCALABILITY) ---
            for threads in thread_counts:
                 print(f"\n-----------------------------------------------------------------")
                 print(f" EXPERIMENT: K={k} | THREADS={threads}")
                 print(f"-----------------------------------------------------------------")
                 
                 # --- EXECUCIÓ PARAL·LELA ---
                 print(f"\n>> Executant PARAL·LEL (K={k}, T={threads})...")
                 t_par, std_par, n_par, centroids_par_dict = run_parallel(INPUT_DIR, OUTPUT_DIR_PAR, k=k, limit=limit, n_threads=threads)
                 
                 # --- CÀLCUL DE SPEEDUP I EFICIÈNCIA ---
                 if t_par > 0:
                     speedup = t_seq / t_par
                 else:
                     speedup = 0
                     
                 if threads > 0:
                     efficiency = speedup / threads
                 else:
                     efficiency = 0
                     
                 print(f"\n>> RESULTATS K={k} T={threads}:")
                 print(f"   - Temps Seqüencial: {t_seq:.4f} s (+/- {std_seq:.4f})")
                 print(f"   - Temps Paral·lel:  {t_par:.4f} s (+/- {std_par:.4f})")
                 print(f"   - SPEEDUP:          {speedup:.2f}x")
                 print(f"   - EFICIÈNCIA:       {efficiency:.2f}") # (1.0 = 100% ideal lineal)
                 
                 # --- VALIDACIÓ DE CORRECCIÓ ---
                 print(f"\n   [VALIDACIÓ] Comparant resultats...")
                 correct_count = 0
                 total_checked = 0
                 
                 common_files = set(centroids_seq_dict.keys()).intersection(set(centroids_par_dict.keys()))
                 for filename in common_files:
                     c_seq = centroids_seq_dict[filename]
                     c_par = centroids_par_dict[filename]
                     
                     passed = False
                     
                     # Check 1: Direct Centroid Match
                     if np.allclose(c_seq, c_par, atol=1e-1):
                         passed = True
                     
                     # Check 2: Sorted Centroid Match (Robustness)
                     if not passed:
                         c_seq_sorted = c_seq[np.argsort(c_seq[:, 0])]
                         c_par_sorted = c_par[np.argsort(c_par[:, 0])]
                         if np.allclose(c_seq_sorted, c_par_sorted, atol=1e-1):
                             passed = True

                     if passed:
                         correct_count += 1
                     else:
                         # pass
                         pass
                     total_checked += 1

                 if total_checked > 0 and correct_count == total_checked:
                      validation_msg = "PASSED"
                      print(f"   [RESULTAT] CORRECTE ({correct_count}/{total_checked})")
                 else:
                      validation_msg = "FAILED"
                      print(f"   [RESULTAT] FALLIDA ({total_checked - correct_count} errors)")

                 results.append({
                     "k": k,
                     "threads": threads,
                     "seq": t_seq,
                     "std_seq": std_seq,
                     "par": t_par,
                     "std_par": std_par,
                     "speedup": speedup,
                     "efficiency": efficiency,
                     "validation": validation_msg
                 })

        # 3. Resum Final per Dataset
        print(f"\n=========================================================================================")
        print(f"                       RESUM FINAL ({d_name})                    ")
        print(f"=========================================================================================")
        print(f"{'K':<5} | {'Threads':<8} | {'T. Seq':<10} | {'S.Dev':<8} | {'T. Par':<10} | {'S.Dev':<8} | {'Spd':<8} | {'Eff':<8} | {'Valid'}")
        print("-" * 105)
        for r in results:
            val = r.get('validation', 'N/A')
            print(f"{r['k']:<5} | {r['threads']:<8} | {r['seq']:<10.4f} | {r['std_seq']:<8.4f} | {r['par']:<10.4f} | {r['std_par']:<8.4f} | {r['speedup']:<8.2f}x | {r['efficiency']:<8.2f} | {val}")
        print("=========================================================================================")
        
        # 4. Guardar resultats en CSV SEPARAT
        csv_filename = f"benchmark_results_{d_name}.csv"
        print(f"\n[INFO] Guardant resultats de '{d_name}' a '{csv_filename}'...")
        
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Capçaleres
            # Capçaleres
            writer.writerow(["K", "Threads", "T. Seq", "Std. Seq", "T. Par", "Std. Par", "Spd", "Eff", "Valid"])
            
            for r in results:
                val = r.get('validation', 'N/A')
                writer.writerow([r['k'], r['threads'], f"{r['seq']:.4f}", f"{r['std_seq']:.4f}", f"{r['par']:.4f}", f"{r['std_par']:.4f}", f"{r['speedup']:.4f}", f"{r['efficiency']:.4f}", val])
                
        print("Arxiu guardat correctament.")
    
    print("\nBenchmark Multi-Dataset finalitzat.")

if __name__ == "__main__":
    main()
