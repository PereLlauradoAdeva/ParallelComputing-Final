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
    
    # 1. Experiment Configuration (DATASETS)
    DATASETS = [
        {"name": "Full_Res", "path": "archive"},
        {"name": "Small_Res", "path": "archive2/seg_pred/seg_pred"}
    ]
    
    OUTPUT_DIR_SEQ = "output_seq_bench"
    OUTPUT_DIR_PAR = "output_par_bench"
    
    k_values = [8, 16, 32] 
    
    # Detect max available threads to avoid Numba errors
    max_threads = numba.config.NUMBA_NUM_THREADS
    all_thread_counts = [1, 2, 4, 8, 16]
    thread_counts = [t for t in all_thread_counts if t <= max_threads]
    
    print(f"[INFO] Max threads detected: {max_threads}")
    print(f"[INFO] Effective thread counts: {thread_counts}")

    print(f"\n[INFO] Starting Multi-Dataset Benchmark")

    # --- DATASET LOOP ---
    for dataset in DATASETS:
        d_name = dataset["name"]
        d_path = dataset["path"]
        
        print(f"\n=================================================================")
        print(f"   DATASET: {d_name}  (Path: {d_path})")
        print(f"=================================================================")
        
        if not os.path.exists(d_path):
            print(f"[ALERT] Directory '{d_path}' does not exist. Skipping...")
            continue

        # Get number of images to process FOR THIS DATASET
        try:
            limit_input = input(f"\n[CONFIG] Number of images for '{d_name}' (-1 = all): ")
            limit = int(limit_input)
        except ValueError:
            limit = 5 
            print(f"   [WARN] Invalid input. Using default {limit}.")
            
        # Reset results for each dataset to have clean separate CSVs
        results = []
        
        INPUT_DIR = d_path 

        print(f"\n[INFO] Processing '{d_name}' with limit={limit}")
        
        # 2. Test Loop
        for k in k_values:
            print(f"\n-----------------------------------------------------------------")
            print(f" EXPERIMENT: K (Clusters) = {k}")
            print(f"-----------------------------------------------------------------")
            
            # --- SEQUENTIAL EXECUTION ---
            print(f"\n>> Running SEQUENTIAL (K={k})...")
            t_seq, std_seq, n_seq, centroids_seq_dict = run_sequential(INPUT_DIR, OUTPUT_DIR_SEQ, limit=limit, k=k)
            
            # --- THREADS LOOP (SCALABILITY) ---
            for threads in thread_counts:
                 print(f"\n-----------------------------------------------------------------")
                 print(f" EXPERIMENT: K={k} | THREADS={threads}")
                 print(f"-----------------------------------------------------------------")
                 
                 # --- PARALLEL EXECUTION ---
                 print(f"\n>> Running PARALLEL (K={k}, T={threads})...")
                 t_par, std_par, n_par, centroids_par_dict = run_parallel(INPUT_DIR, OUTPUT_DIR_PAR, k=k, limit=limit, n_threads=threads)
                 
                 # --- CALCULATE SPEEDUP AND EFFICIENCY ---
                 if t_par > 0:
                     speedup = t_seq / t_par
                 else:
                     speedup = 0
                     
                 if threads > 0:
                     efficiency = speedup / threads
                 else:
                     efficiency = 0
                     
                 print(f"\n>> RESULTS K={k} T={threads}:")
                 print(f"   - Sequential Time: {t_seq:.4f} s (+/- {std_seq:.4f})")
                 print(f"   - Parallel Time:   {t_par:.4f} s (+/- {std_par:.4f})")
                 print(f"   - SPEEDUP:         {speedup:.2f}x")
                 print(f"   - EFFICIENCY:      {efficiency:.2f}") # (1.0 = 100% ideal linear)
                 
                 # --- CORRECTNESS VALIDATION ---
                 print(f"\n   [VALIDATION] Comparing results (Inertia/Error)...")
                 correct_count = 0
                 total_checked = 0
                 
                 common_files = set(centroids_seq_dict.keys()).intersection(set(centroids_par_dict.keys()))
                 for filename in common_files:
                     # Retrieve data: Now it is a dictionary {"centroids": ..., "inertia": ...}
                     data_seq = centroids_seq_dict[filename]
                     data_par = centroids_par_dict[filename]
                     
                     if isinstance(data_seq, dict) and "inertia" in data_seq:
                         inertia_seq = data_seq["inertia"]
                         inertia_par = data_par["inertia"]
                     else:
                         # Fallback if something went wrong or old versions
                         inertia_seq = 1.0 
                         inertia_par = 1.0 # Avoid crash
                         print(f"   [WARN] Old data format detected for {filename}")

                     # Calculate Relative Error Percentage
                     # abs(seq - par) / seq * 100
                     if inertia_seq == 0:
                         diff_rel = 0.0
                     else:
                         diff_rel = abs(inertia_seq - inertia_par) / inertia_seq * 100.0
                     
                     # Acceptable Tolerance: < 0.1% difference in total error
                     # This accepts different solutions (Butterfly Effect) as long as they are equally good.
                     TOLERANCE_PERCENT = 0.1 
                     
                     if diff_rel < TOLERANCE_PERCENT:
                         correct_count += 1
                     else:
                         print(f"     FAIL {filename}: Diff={diff_rel:.4f}% (Seq={inertia_seq:.1f}, Par={inertia_par:.1f})")
                     
                     total_checked += 1

                 if total_checked > 0 and correct_count == total_checked:
                      validation_msg = "PASSED"
                      print(f"   [RESULT] CORRECT ({correct_count}/{total_checked}) - All within {TOLERANCE_PERCENT}% relative error.")
                 else:
                      validation_msg = "FAILED"
                      print(f"   [RESULT] FAILED ({total_checked - correct_count} errors)")

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

        # 3. Final Summary per Dataset
        print(f"\n=========================================================================================")
        print(f"                       FINAL SUMMARY ({d_name})                    ")
        print(f"=========================================================================================")
        print(f"{'K':<5} | {'Threads':<8} | {'T. Seq':<10} | {'S.Dev':<8} | {'T. Par':<10} | {'S.Dev':<8} | {'Spd':<8} | {'Eff':<8} | {'Valid'}")
        print("-" * 105)
        for r in results:
            val = r.get('validation', 'N/A')
            print(f"{r['k']:<5} | {r['threads']:<8} | {r['seq']:<10.4f} | {r['std_seq']:<8.4f} | {r['par']:<10.4f} | {r['std_par']:<8.4f} | {r['speedup']:<8.2f}x | {r['efficiency']:<8.2f} | {val}")
        print("=========================================================================================")
        
        # 4. Save results to SEPARATE CSV
        csv_filename = f"benchmark_results_{d_name}.csv"
        print(f"\n[INFO] Saving results of '{d_name}' to '{csv_filename}'...")
        
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Headers
            # Headers
            writer.writerow(["K", "Threads", "T. Seq", "Std. Seq", "T. Par", "Std. Par", "Spd", "Eff", "Valid"])
            
            for r in results:
                val = r.get('validation', 'N/A')
                writer.writerow([r['k'], r['threads'], f"{r['seq']:.4f}", f"{r['std_seq']:.4f}", f"{r['par']:.4f}", f"{r['std_par']:.4f}", f"{r['speedup']:.4f}", f"{r['efficiency']:.4f}", val])
                
        print("Arxiu guardat correctament.")
    
    print("\nBenchmark Multi-Dataset finalitzat.")

if __name__ == "__main__":
    main()
