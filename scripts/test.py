import ctypes

try:
    # Load the NVIDIA driver shared library
    cuda = ctypes.CDLL("libcuda.so.1")
    
    # Check the version (this function is part of the driver API)
    version = ctypes.c_int()
    cuda.cuDriverGetVersion(ctypes.byref(version))
    
    print(f"Raw Driver Version Code: {version.value}")
    print(f"Decoded CUDA Version: {version.value // 1000}.{(version.value % 1000) // 10}")

except OSError:
    print("Could not find libcuda.so.1. The driver might not be visible in this environment.")
except Exception as e:
    print(f"An error occurred: {e}")