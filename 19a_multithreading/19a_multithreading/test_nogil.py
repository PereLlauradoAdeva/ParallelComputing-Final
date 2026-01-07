import sys
import sysconfig


def is_python_314t():
    v = sys.version_info
    # Python 3.14 and 't' ABI flag (free-threaded build)
    return (v.major, v.minor) == (3, 14) and getattr(sys, "abiflags", "") == "t"


def is_gil_disabled():
    # This config var is defined for free-threaded builds in 3.13+
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def main():
    print("Python version:", sys.version)
    print("is_gil_disabled:", is_gil_disabled())

    if is_python_314t() and is_gil_disabled():
        print("Running on Python 3.14t without the GIL.")
    else:
        print("Not running on Python 3.14t without the GIL.")


if __name__ == "__main__":
    main()
