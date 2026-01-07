import os
import time
import csv
import numba
from sequencial import process_images as run_sequential
from parallel import run_kmeans_parallel_cpu_folder as run_parallel

def main():
    print("=================================================================")
    print("      BENCHMARK K-MEANS COLOR QUANTIZATION (CPU vs PARALLEL)     ")
    print("=================================================================")
    
    # 1. Configuració de l'experiment
    INPUT_DIR = "archive"
    OUTPUT_DIR_SEQ = "output_seq_bench"
    OUTPUT_DIR_PAR = "output_par_bench"
    
    # Obtenir el nombre d'imatges a processar
    try:
        limit_input = input("\n[CONFIG] Nombre d'imatges per test (-1 = totes): ")
        limit = int(limit_input)
    except ValueError:
        limit = 5 # Default segur si error
        
    k_values = [8, 16, 32] # Diferents complexitats de colors
    
    results = []

    print(f"\n[INFO] Iniciant bateria de tests. Imatges: {limit}")
    
    # 2. Bucle de Tests
    for k in k_values:
        print(f"\n-----------------------------------------------------------------")
        print(f" EXPERIMENT: K (Clusters) = {k}")
        print(f"-----------------------------------------------------------------")
        
        # --- EXECUCIÓ SEQÜENCIAL ---
        print(f"\n>> Executant SEQÜENCIAL (K={k})...")
        t_seq, n_seq = run_sequential(INPUT_DIR, OUTPUT_DIR_SEQ, limit=limit) # Nota: He d'adaptar sequencial per acceptar K si no ho fa
        
        # --- EXECUCIÓ PARAL·LELA ---
        # Numba utilitza automàticament tots els cores disponibles.
        print(f"\n>> Executant PARAL·LEL (K={k})...")
        t_par, n_par = run_parallel(INPUT_DIR, OUTPUT_DIR_PAR, k=k, limit=limit)
        
        # --- CÀLCUL DE SPEEDUP ---
        if t_par > 0:
            speedup = t_seq / t_par
        else:
            speedup = 0
            
        print(f"\n>> RESULTATS K={k}:")
        print(f"   - Temps Seqüencial: {t_seq:.4f} s")
        print(f"   - Temps Paral·lel:  {t_par:.4f} s")
        print(f"   - SPEEDUP:          {speedup:.2f}x")
        
        results.append({
            "k": k,
            "seq": t_seq,
            "par": t_par,
            "speedup": speedup
        })

    # 3. Resum Final
    print(f"\n=================================================================")
    print(f"                       RESUM FINAL                           ")
    print(f"=================================================================")
    print(f"{'K (Clusters)':<15} | {'T. Seq (s)':<15} | {'T. Par (s)':<15} | {'Speedup':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['k']:<15} | {r['seq']:<15.4f} | {r['par']:<15.4f} | {r['speedup']:.2f}x")
    print("=================================================================")
    print("Benchmark finalitzat.")

    # 4. Guardar resultats en CSV
    csv_filename = "benchmark_results.csv"
    print(f"\n[INFO] Guardant resultats a '{csv_filename}'...")
    
    # Obtenir el nombre de fils utilitzats per Numba
    threads = numba.config.NUMBA_NUM_THREADS
    
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Capçaleres demanades: Threads,Sequential_Time_ms,Parallel_Time_ms,Speedup,Efficiency
        writer.writerow(["Threads", "Sequential_Time_ms", "Parallel_Time_ms", "Speedup", "Efficiency"])
        
        for r in results:
            seq_ms = r['seq'] * 1000
            par_ms = r['par'] * 1000
            speedup = r['speedup']
            efficiency = speedup / threads if threads > 0 else 0
            
            writer.writerow([threads, f"{seq_ms:.4f}", f"{par_ms:.4f}", f"{speedup:.4f}", f"{efficiency:.4f}"])
            
    print("Arxiu guardat correctament.")

if __name__ == "__main__":
    main()
