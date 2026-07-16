from csi_tool.fetch_data import export_numpy
import time

if __name__ == '__main__':
    start_time = time.perf_counter()

    export_numpy()
    
    elapsed = time.perf_counter() - start_time
    print(f"Execution time: {elapsed:.3f} s")