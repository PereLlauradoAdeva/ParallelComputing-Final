"""
Compare:
- Manual threads: create a new Thread for each task.
- Thread pool: reuse a fixed set of worker threads.

Goal: show when a thread pool can be faster:
- Many small tasks
- Similar level of parallelism
- Minimized print() overhead
"""

import threading
import time
import random
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor, wait


# ==== TUNABLE PARAMETERS ======================================================

# 1) Many small tasks: increase these to amplify the benefit of a pool
NUM_BATCHES = 50          # how many groups of tasks
TASKS_PER_BATCH = 20      # tasks per batch  -> total tasks = NUM_BATCHES * TASKS_PER_BATCH

# 2) Task duration: smaller tasks make thread-creation overhead more visible
BASE_DURATION = 0.002     # ~2ms per task (simulated work)

# 3) Equal / fair parallelism:
#    Set MAX_WORKERS ~= TASKS_PER_BATCH so both approaches can run similar #tasks in parallel.
MAX_WORKERS = TASKS_PER_BATCH

# 4) Printing is very expensive; turn it off for realistic timings.
ENABLE_PRINTS = False     # set True to see per-task logs & thread names


# ==== TASK FUNCTION ===========================================================

def handle_task(task_id: int, batch_id: int, duration: float):
    """
    Simulate a small unit of work, e.g. a short I/O + CPU job.
    Returns the thread name so we can inspect which threads ran the tasks.
    """
    thread_name = threading.current_thread().name

    if ENABLE_PRINTS:
        print(f"[batch {batch_id:02d} task {task_id:02d}] START on {thread_name}")

    time.sleep(duration)

    if ENABLE_PRINTS:
        print(f"[batch {batch_id:02d} task {task_id:02d}] END   on {thread_name}")

    return thread_name


# ==== VERSION 1: MANUAL THREADS ==============================================

def demo_manual_threads():
    """
    Manual threading:
    - For each task, we create a brand new Thread.
    - For each batch, we start and then join all threads.

    Cost model:
    - Per task: work_time + thread_create + thread_teardown
    - With many small tasks, thread_create/teardown overhead becomes significant.
    """
    total_tasks = NUM_BATCHES * TASKS_PER_BATCH

    print("=== Manual threads (1 new Thread per task) ===")
    start = time.perf_counter()

    for batch in range(1, NUM_BATCHES + 1):
        threads = []

        if ENABLE_PRINTS:
            print(f"\n--- Starting batch {batch} with manual threads ---")

        for task in range(1, TASKS_PER_BATCH + 1):
            # Small, slightly jittered duration
            duration = BASE_DURATION * (1.0 + 0.1 * random.random())

            t = threading.Thread(
                target=handle_task,
                args=(task, batch, duration),
            )
            t.start()
            threads.append(t)

        # Wait for all threads in this batch to finish
        for t in threads:
            t.join()

    elapsed = time.perf_counter() - start
    print(f"Manual threads total time: {elapsed:.3f} s "
          f"for {total_tasks} tasks ({total_tasks / elapsed:.1f} tasks/s)")

    print("Note: this approach creates and destroys a Thread object for EVERY task.\n")
    return elapsed, total_tasks


# ==== VERSION 2: THREAD POOL ==================================================

def demo_thread_pool():
    """
    Thread pool:
    - Create a fixed set of worker threads once (MAX_WORKERS).
    - Submit all tasks to the pool; workers pull tasks from a queue.

    Cost model:
    - Per task: work_time + small queue/dispatch overhead
    - Plus ONE-TIME cost of creating MAX_WORKERS threads
    - With many small tasks, amortized cost per task is much smaller.
    """
    total_tasks = NUM_BATCHES * TASKS_PER_BATCH
    all_thread_names = set()

    print(f"=== ThreadPoolExecutor (reusing {MAX_WORKERS} worker threads) ===")
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for batch in range(1, NUM_BATCHES + 1):
            futures = []

            if ENABLE_PRINTS:
                print(f"\n--- Submitting batch {batch} to the pool ---")

            for task in range(1, TASKS_PER_BATCH + 1):
                duration = BASE_DURATION * (1.0 + 0.1 * random.random())
                fut = executor.submit(handle_task, task, batch, duration)
                futures.append(fut)

            # Wait for all tasks in this batch and collect which threads ran them
            done, _ = wait(futures)
            for fut in done:
                all_thread_names.add(fut.result())

    elapsed = time.perf_counter() - start
    print(f"Thread pool total time: {elapsed:.3f} s "
          f"for {total_tasks} tasks ({total_tasks / elapsed:.1f} tasks/s)")

    print("\nPool worker threads actually used (reused across all batches):")
    for name in sorted(all_thread_names):
        print("  ", name)
    print()

    return elapsed, total_tasks, all_thread_names


# ==== MAIN COMPARISON =========================================================

if __name__ == "__main__":
    random.seed(0)

    manual_time, total_tasks_manual = demo_manual_threads()
    pool_time, total_tasks_pool, pool_threads = demo_thread_pool()

    assert total_tasks_manual == total_tasks_pool
    total_tasks = total_tasks_manual

    print("=== Summary ===")
    print(f"Python version: {sys.version.split()[0]}{getattr(sys, "abiflags", "")}")
    print(f"Py_GIL_DISABLED: {bool(sysconfig.get_config_var("Py_GIL_DISABLED"))}")
    print(f"Manual threads: {manual_time:.3f} s "
          f"({total_tasks / manual_time:.1f} tasks/s)")
    print(f"Thread pool:   {pool_time:.3f} s "
          f"({total_tasks / pool_time:.1f} tasks/s)")

    if pool_time > 0:
        speedup = manual_time / pool_time
        print(f"Speedup (manual / pool): {speedup:.2f}x "
              "(>1.0 means pool is faster)")
        print("Speedup should be more visible in multithread (no-GIL) Python")

    print("\nKey considerations:")
    print("  - Many small tasks (NUM_BATCHES, TASKS_PER_BATCH) highlight pool benefits.")
    print("  - Equal parallelism: MAX_WORKERS ~= TASKS_PER_BATCH for a fair comparison.")
    print("  - Printing is disabled by default (ENABLE_PRINTS = False) because "
          "print() is much slower than thread management itself.")
    print("  - Manual threads pay create+teardown cost per task, "
          "while the pool amortizes that over many tasks.")
