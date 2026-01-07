#!/usr/bin/env python3
"""
Compare behaviour of a shared counter on:

- CPython 3.14 with GIL
- CPython 3.14t (no-GIL / free-threaded)

We run two experiments:

1. Increment a shared counter from many threads *without* any lock
   -> data race
   -> On classic CPython with GIL, this often still prints the "correct"
      result by accident, because only one thread executes Python
      bytecode at a time.
   -> On no-GIL Python, this will almost always give a smaller,
      incorrect result due to lost updates.

2. Increment the same shared counter but use a threading.Lock
   to protect the critical section
   -> correct result on both interpreters, but a bit slower.
"""

import threading
import time
import sys
import sysconfig

# Tune these to make the effect more visible on your machine
NUM_THREADS = 8
INCREMENTS_PER_THREAD = 200_000

# Shared state
counter = 0
lock = threading.Lock()


def increment_no_lock() -> None:
    """Increment shared counter without any synchronization (data race)."""
    global counter
    for _ in range(INCREMENTS_PER_THREAD):
        # Not atomic on a free-threaded interpreter:
        #   1. load counter
        #   2. add 1
        #   3. store result
        counter += 1


def increment_with_lock() -> None:
    """Increment shared counter using a lock (correct but slower)."""
    global counter
    for _ in range(INCREMENTS_PER_THREAD):
        # Only one thread at a time can execute this block
        with lock:
            counter += 1


def run_test(target, label: str) -> None:
    """Run one experiment and print the result."""
    global counter
    counter = 0

    threads = [threading.Thread(target=target) for _ in range(NUM_THREADS)]

    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    expected = NUM_THREADS * INCREMENTS_PER_THREAD
    print(f"\n=== {label} ===")
    print(f"Expected: {expected}")
    print(f"Actual:   {counter}")
    print(f"Elapsed:  {elapsed:.3f} s")


if __name__ == "__main__":
    print(f"Python version: {sys.version.split()[0]}{getattr(sys, "abiflags", "")}")
    print(f"Py_GIL_DISABLED: {bool(sysconfig.get_config_var("Py_GIL_DISABLED"))}")
    print(f"Threads: {NUM_THREADS}, increments per thread: {INCREMENTS_PER_THREAD}")

    run_test(increment_no_lock, "WITHOUT lock (data race)")
    run_test(increment_with_lock, "WITH lock (correct)")
