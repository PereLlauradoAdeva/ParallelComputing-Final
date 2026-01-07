"""
threads_primes_pool.py – threading using ThreadPoolExecutor.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def run_threads_pool(num_workers: int = 4) -> float:
    START, END = 1, 3_000_00
    ranges = _make_ranges(START, END, num_workers)

    t0 = time.perf_counter()
    results: List[int] = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        future_to_range = {
            ex.submit(count_primes, s, e): (s, e) for (s, e) in ranges
        }
        for fut in as_completed(future_to_range):
            results.append(fut.result())
    t1 = time.perf_counter()

    total = sum(results)
    elapsed = t1 - t0
    print(f"[threads-pool] workers={num_workers}, primes={total}, time={elapsed:.3f} s")
    return elapsed


if __name__ == "__main__":
    run_threads_pool()
