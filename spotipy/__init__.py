import os

def set_omp_num_threads():
    # set OMP_NUM_THREADS to 1/2 of the number of CPUs by default
    n_cpu = os.cpu_count()
    n_threads = int(os.environ.get("OMP_NUM_THREADS",1))
    n_threads = min(n_threads, n_cpu//2)
    print(f'using {n_threads} thread(s)')
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

set_omp_num_threads()
    