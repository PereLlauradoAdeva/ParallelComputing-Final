"""
threads_primes_manual.py – manual threading using threading.Thread.
"""

from __future__ import annotations

import threading
import time
from typing import List

from primes import count_primes


def run_threads(num_threads: int = 4) -> float:
    START, END = 1, 3_000_00
    step = (END - START) // num_threads

    results: List[int] = [0] * num_threads

    def worker(i: int, s: int, e: int) -> None:
        results[i] = count_primes(s, e)

    threads: list[threading.Thread] = []
    t0 = time.perf_counter()
    for i in range(num_threads):
        s = START + i * step
        e = END if i == num_threads - 1 else s + step
        t = threading.Thread(target=worker, args=(i, s, e))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    t1 = time.perf_counter()

    total = sum(results)
    elapsed = t1 - t0
    print(f"[threads] workers={num_threads}, primes={total}, time={elapsed:.3f} s")
    return elapsed


if __name__ == "__main__":
    run_threads()
