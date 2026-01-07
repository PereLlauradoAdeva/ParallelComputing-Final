import threading
import time


# --- 1) Subclass threading.Thread -------------------

class MyThread(threading.Thread):
    def __init__(self, name, delay):
        super().__init__()  # initialize the base Thread
        self.name = name
        self.delay = delay

    def run(self):
        """This method is called when you call .start()."""
        for i in range(3):
            time.sleep(self.delay)
            print(f"[CLASS] {self.name} iteration {i}")


# --- 2) Passing a function as target -----------------------------

def worker_function(name, delay):
    for i in range(3):
        time.sleep(delay)
        print(f"[FUNC ] {name} iteration {i}")


if __name__ == "__main__":
    # Create threads using the CLASS approach
    t1 = MyThread(name="ClassThread-1", delay=0.5)
    t2 = MyThread(name="ClassThread-2", delay=0.7)

    # Create threads using the FUNCTION (target) approach
    t3 = threading.Thread(target=worker_function,
                          args=("FuncThread-1", 0.6))
    t4 = threading.Thread(target=worker_function,
                          args=("FuncThread-2", 0.8))

    # Start all threads
    t1.start()
    t2.start()
    t3.start()
    t4.start()

    # Wait for all to finish
    t1.join()
    t2.join()
    t3.join()
    t4.join()

    print("All threads completed.")
