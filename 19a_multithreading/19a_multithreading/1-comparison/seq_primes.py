"""
seq_primes.py – sequential baseline for the prime-counting workload.
"""

from __future__ import annotations

import time

from primes import count_primes


def run_sequential() -> float:
    START, END = 1, 3_000_00

    t0 = time.perf_counter()
    total = count_primes(START, END)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f"[sequential] primes={total}, time={elapsed:.3f} s")
    return elapsed


if __name__ == "__main__":
    run_sequential()
