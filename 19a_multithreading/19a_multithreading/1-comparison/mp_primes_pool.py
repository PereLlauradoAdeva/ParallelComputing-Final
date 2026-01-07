"""
mp_primes_pool.py – multiprocessing.Pool version of the prime benchmark.
"""

from __future__ import annotations

import time
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

from primes import count_primes


def _make_ranges(start: int, end: int, chunks: int) -> List[Tuple[int, int]]:
    step = (end - start) // chunks
    ranges: List[Tuple[int, int]] = []
    for i in range(chunks):
        s = start + i * step
        e = end if i == chunks - 1 else s + step
        ranges.append((s, e))
    return ranges


def run_multiprocessing(num_workers: int | None = None) -> float:
    if num_workers is None:
        num_workers = cpu_count()

    START, END = 1, 3_000_00
    ranges = _make_ranges(START, END, num_workers)

    t0 = time.perf_counter()
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(count_primes, ranges)
    t1 = time.perf_counter()

    total = sum(results)
    elapsed = t1 - t0
    print(f"[multiprocessing] workers={num_workers}, primes={total}, time={elapsed:.3f} s")
    return elapsed


if __name__ == "__main__":
    run_multiprocessing()
