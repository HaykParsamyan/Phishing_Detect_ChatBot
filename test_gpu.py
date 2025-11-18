import cudf
import cupy as cp
import numpy as np
import os
import sys


def check_cudf_install():
    """
    Checks if cuDF is correctly installed, can access the GPU, and performs
    a basic operation.
    """
    try:
        # --- 1. Basic Import and Version Check ---
        print("--- Checking cuDF Status ---")
        print(f"cuDF version: {cudf.__version__}")

        # --- 2. GPU Accessibility Check ---
        # Checks if Cupy (a key dependency) can see the CUDA device.
        if cp.cuda.is_available():
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"CUDA devices detected: {device_count}")
            print(f"Using GPU: {cp.cuda.Device(0).name}")
        else:
            print("ERROR: CuPy/CUDA device not found.")
            sys.exit(1)

        # --- 3. Operational Test (Creating a DataFrame on GPU) ---
        print("\n--- Running Operational Test ---")
        # Create a small Pandas DataFrame on the CPU
        cpu_data = np.random.rand(5, 3)
        pdf = cudf.pandas.DataFrame(cpu_data, columns=['A', 'B', 'C'])

        # Convert the Pandas DataFrame to a cuDF DataFrame on the GPU
        gdf = cudf.DataFrame.from_pandas(pdf)

        # Perform a GPU-accelerated operation (sum of column A)
        gpu_sum = gdf['A'].sum()

        print(f"cuDF DataFrame created successfully (on GPU): {gdf.shape}")
        print(f"Result of GPU sum (Col 'A'): {gpu_sum}")
        print("\n✅ cuDF is installed and operating correctly on the GPU.")

    except ImportError:
        print("\n❌ ERROR: cuDF or one of its dependencies (like cuPy) is not installed.")
        print("Please ensure you run 'conda activate [your-rapids-env]' before running this script.")
    except Exception as e:
        print(f"\n❌ FATAL ERROR during cuDF operation: {e}")
        print("The libraries are installed but cannot link to your CUDA/NVIDIA driver.")
        print("Try reinstalling the NVIDIA driver or check the environment variables.")


if __name__ == '__main__':
    check_cudf_install()