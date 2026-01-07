"""
primes.py – shared CPU-bound workload for threading/multiprocessing demos.
Compatible with both classic CPython (with GIL) and free-threaded builds.
"""


from __future__ import annotations


def is_prime(n: int) -> bool:
    """Simple (inefficient but clear) primality test."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def count_primes(start: int, end: int) -> int:
    """Count primes in [start, end)."""
    return sum(1 for x in range(start, end) if is_prime(x))
