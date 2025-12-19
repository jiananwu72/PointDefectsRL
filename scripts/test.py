import ctypes
import sys

def check_driver_version():
    # Try to load the driver library (standard names on Linux)
    lib_names = ['libcuda.so', 'libcuda.so.1']
    cuda = None
    
    for name in lib_names:
        try:
            cuda = ctypes.CDLL(name)
            print(f"Success: Loaded {name}")
            break
        except OSError:
            continue

    if not cuda:
        print("Error: Could not load libcuda.so or libcuda.so.1")
        return

    # 1. Initialize the driver (required before querying version)
    # 0 is the flags argument
    result = cuda.cuInit(0)
    
    if result != 0:
        print(f"cuInit failed with error code: {result}")
        # Error 100 = CUDA_ERROR_NO_DEVICE
        # Error 803 = CUDA_ERROR_SYSTEM_DRIVER_MISMATCH
        # Error 35 = CUDA_ERROR_INSUFFICIENT_DRIVER
        if result == 35:
            print("  -> Confirmed: The driver is too old for the CUDA Runtime you are using.")
        return

    # 2. Query the version
    version = ctypes.c_int()
    result = cuda.cuDriverGetVersion(ctypes.byref(version))

    if result != 0:
        print(f"cuDriverGetVersion failed with error code: {result}")
    else:
        v = version.value
        main_ver = v // 1000
        minor_ver = (v % 1000) // 10
        print(f"\nRAW DRIVER VERSION: {v}")
        print(f"DECODED VERSION:    CUDA {main_ver}.{minor_ver}")

if __name__ == "__main__":
    check_driver_version()