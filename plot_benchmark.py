import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_speedup_per_dataset(csv_file, output_name, dataset_name):
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) < 10:
        print(f"[WARN] Skipping {dataset_name} (File empty or missing)")
        return

    try:
        df = pd.read_csv(csv_file)
        
        # Strip whitespace from columns
        df.columns = df.columns.str.strip()
        
        # Get unique K values
        k_values = sorted(df['K'].unique())
        
        plt.figure(figsize=(10, 6))
        
        colors = {8: 'green', 16: 'orange', 32: 'blue'}
        
        for k in k_values:
            subset = df[df['K'] == k]
            if subset.empty:
                continue
                
            subset = subset.sort_values(by='Threads')
            threads = subset['Threads']
            speedup = subset['Spd']
            
            plt.plot(threads, speedup, marker='o', linestyle='-', linewidth=2, 
                     color=colors.get(k, 'black'), label=f'K={k}')

        # Ideal Speedup Line (Linear)
        # Use threads from the last subset as reference (assuming they are same for all K)
        if 'threads' in locals():
             plt.plot(threads, threads, linestyle='--', color='gray', alpha=0.7, label='Ideal Speedup')
        
        plt.title(f'Speedup vs Threads - {dataset_name}')
        plt.xlabel('Threads')
        plt.ylabel('Speedup')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        if 'threads' in locals():
            plt.xticks(threads) 
        
        plt.savefig(output_name)
        print(f"[SUCCESS] Saved plot to {output_name}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Could not plot {dataset_name}: {e}")

def plot_efficiency_per_dataset(csv_file, output_name, dataset_name):
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) < 10:
        print(f"[WARN] Skipping {dataset_name} (File empty or missing)")
        return

    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        k_values = sorted(df['K'].unique())
        
        plt.figure(figsize=(10, 6))
        
        colors = {8: 'green', 16: 'orange', 32: 'blue'}
        
        for k in k_values:
            subset = df[df['K'] == k]
            if subset.empty:
                continue
                
            subset = subset.sort_values(by='Threads')
            threads = subset['Threads']
            efficiency = subset['Eff']
            
            plt.plot(threads, efficiency, marker='o', linestyle='-', linewidth=2, 
                     color=colors.get(k, 'black'), label=f'K={k}')

        plt.title(f'Efficiency vs Threads - {dataset_name}')
        plt.xlabel('Threads')
        plt.ylabel('Efficiency (Speedup / Threads)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        if 'threads' in locals():
            plt.xticks(threads) 
        
        plt.savefig(output_name)
        print(f"[SUCCESS] Saved plot to {output_name}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Could not plot Efficiency {dataset_name}: {e}")

def plot_time_comparison(csv_file, output_name, dataset_name, k_val=32):
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) < 10:
        print(f"[WARN] Skipping {dataset_name} (File empty or missing)")
        return

    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        
        subset = df[df['K'] == k_val]
        if subset.empty:
            print(f"[WARN] No data for K={k_val} in {dataset_name}")
            return
            
        subset = subset.sort_values(by='Threads')
        threads = subset['Threads']
        t_par = subset['T. Par']
        t_seq = subset['T. Seq'].iloc[0] # Sequential time is constant
        
        plt.figure(figsize=(10, 6))
        
        # Parallel Times as Bars
        bars = plt.bar(threads.astype(str), t_par, color='skyblue', label='Parallel Time', alpha=0.8)
        
        # Sequential Time as Horizontal Line
        plt.axhline(y=t_seq, color='red', linestyle='--', linewidth=2, label=f'Sequential Time ({t_seq:.2f}s)')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}s',
                     ha='center', va='bottom')

        plt.title(f'Execution Time Comparison (K={k_val}) - {dataset_name}')
        plt.xlabel('Threads')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        plt.savefig(output_name)
        print(f"[SUCCESS] Saved plot to {output_name}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Could not plot Time Comparison {dataset_name}: {e}")

def plot_speedup_vs_k_comparison(dataset_files, output_name, fixed_threads=8):
    plt.figure(figsize=(10, 6))
    
    for dataset_name, csv_file in dataset_files.items():
        if not os.path.exists(csv_file) or os.path.getsize(csv_file) < 10:
            print(f"[WARN] Skipping {dataset_name} for K plot (File missing)")
            continue
            
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            
            # Filter for fixed threads
            subset = df[df['Threads'] == fixed_threads]
            
            if subset.empty:
                print(f"[WARN] No data for Threads={fixed_threads} in {dataset_name}")
                continue
            
            subset = subset.sort_values(by='K')
            k_vals = subset['K']
            speedup = subset['Spd']
            
            # Stylistic choice: Solid for Full, Dashed for Small? Or just different colors.
            # User mentioned "Full_Res (solid) ... Small_Res (dashed)" in previous request but let's stick to consistent colors or markers
            # Let's use different markers and colors
            if "Small" in dataset_name:
                style = '--'
                marker = 'x'
            else:
                style = '-'
                marker = 'o'
                
            plt.plot(k_vals, speedup, marker=marker, linestyle=style, linewidth=2, label=f'{dataset_name} (T={fixed_threads})')
            
        except Exception as e:
            print(f"[ERROR] Could not process {dataset_name} for K plot: {e}")
            
    plt.title(f'Speedup vs Cluster Size (K) at {fixed_threads} Threads')
    plt.xlabel('Cluster Size (K)')
    plt.ylabel('Speedup')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks([8, 16, 32]) # Constant K values
    
    plt.savefig(output_name)
    print(f"[SUCCESS] Saved plot to {output_name}")
    plt.close()

if __name__ == "__main__":
    # Speedup Plots
    plot_speedup_per_dataset("benchmark_results_Full_Res.csv", "Speedup_Full_Res_All_K.png", "Full_Res")
    plot_speedup_per_dataset("benchmark_results_Small_Res.csv", "Speedup_Small_Res_All_K.png", "Small_Res")
    
    # Efficiency Plots
    plot_efficiency_per_dataset("benchmark_results_Full_Res.csv", "Efficiency_Full_Res_All_K.png", "Full_Res")
    plot_efficiency_per_dataset("benchmark_results_Small_Res.csv", "Efficiency_Small_Res_All_K.png", "Small_Res")
    
    # Time Comparison Plots (K=32)
    # Re-generating these to ensure script is complete
    plot_time_comparison("benchmark_results_Full_Res.csv", "Time_Comparison_K32_Full_Res.png", "Full_Res", k_val=32)
    plot_time_comparison("benchmark_results_Small_Res.csv", "Time_Comparison_K32_Small_Res.png", "Small_Res", k_val=32)
    
    # Speedup vs K Comparison (Threads=8)
    datasets = {
        "Full_Res": "benchmark_results_Full_Res.csv",
        "Small_Res": "benchmark_results_Small_Res.csv"
    }
    plot_speedup_vs_k_comparison(datasets, "Speedup_vs_K_Threads8.png", fixed_threads=8)
