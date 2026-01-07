"""
1-comparison.py – run all variants and compare speed-ups.
"""

from __future__ import annotations

import os
import sysconfig
import sys
from textwrap import dedent

from seq_primes import run_sequential
from threads_primes_manual import run_threads
from threads_primes_pool import run_threads_pool
from mp_primes_pool import run_multiprocessing


def main() -> None:
    print("=== Environment info ===")
    print(f"Python: {sys.version.split()[0]}{getattr(sys, "abiflags", "")}")
    print(f"Executable: {sys.executable}")
    print(f"Py_GIL_DISABLED: {bool(sysconfig.get_config_var("Py_GIL_DISABLED"))}")
    print(f"Py_GIL_DISABLED={os.environ.get('Py_GIL_DISABLED')}")
    print()

    t_seq = run_sequential()
    print()

    import multiprocessing
    cores = multiprocessing.cpu_count()
    num_workers = max(2, min(cores, 8))
    print(f"Using {num_workers} workers for multiprocessing...")

    t_thr = run_threads(num_threads=num_workers)
    print()

    t_thr_pool = run_threads_pool(num_workers=num_workers)
    print()

    t_mp = run_multiprocessing(num_workers=num_workers)
    print()

    def speedup(base: float, other: float) -> float:
        return base / other if other > 0 else float("inf")

    print("=== Summary (lower is better) ===")
    print(f"sequential:      {t_seq:.3f} s   (baseline)")
    print(f"threads:         {t_thr:.3f} s   (speed-up x{speedup(t_seq, t_thr):.2f})")
    print(f"threads-pool:    {t_thr_pool:.3f} s   (speed-up x{speedup(t_seq, t_thr_pool):.2f})")
    print(f"multiprocessing: {t_mp:.3f} s   (speed-up x{speedup(t_seq, t_mp):.2f})")
    print()
    print(dedent(
        """
        Interpret the results:
          • On classic CPython with the GIL:
              - threads ≈ sequential
              - multiprocessing faster for CPU-bound work
          • On free-threaded Python 3.14 with the GIL disabled:
              - threads and thread pools should show clear speed-ups
              - multiprocessing may still help, but has more overhead
        """
    ))


if __name__ == "__main__":
    main()
