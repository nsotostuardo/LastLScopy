import os
import time
import platform
import numpy as np
import psutil
from numba import njit, prange, get_num_threads, set_num_threads, get_thread_id
import numba

# ============================================================
# 1) ENVIRONMENT INFORMATION
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

print_header("ENVIRONMENT")
print("Python platform            :", platform.platform())
print("Architecture               :", platform.machine())
print("Numba version              :", numba.__version__)
print("os.cpu_count()             :", os.cpu_count())
print("Logical CPUs (psutil)      :", psutil.cpu_count(logical=True))
print("Physical CPUs (psutil)     :", psutil.cpu_count(logical=False))

# Process CPU affinity: very important on clusters / taskset / cgroups
p = psutil.Process(os.getpid())
if hasattr(p, "cpu_affinity"):
    try:
        affinity = p.cpu_affinity()
        print("Process CPU affinity       :", affinity)
        print("CPUs actually available to this process:", len(affinity))
    except Exception as e:
        print("Could not read cpu_affinity():", e)
        affinity = None
else:
    affinity = None
    print("cpu_affinity() not available on this platform")

print_header("RELEVANT ENVIRONMENT VARIABLES")
env_keys = [
    "NUMBA_NUM_THREADS",
    "NUMBA_THREADING_LAYER",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_JOB_ID",
]
for k in env_keys:
    print(f"{k:24s}: {os.environ.get(k)}")

# Decide how many threads Numba SHOULD use
if os.environ.get("SLURM_CPUS_PER_TASK"):
    target_threads = int(os.environ["SLURM_CPUS_PER_TASK"])
elif affinity is not None:
    target_threads = len(affinity)
else:
    target_threads = psutil.cpu_count(logical=True) or os.cpu_count() or 1

set_num_threads(target_threads)

print_header("NUMBA")
print("Numba get_num_threads()    :", get_num_threads())

try:
    print("Numba threading layer      :", numba.threading_layer())
except Exception as e:
    print("threading_layer() not available yet before compiling something:", e)

# ============================================================
# 2) TEST KERNELS
# ============================================================

@njit(parallel=True)
def parallel_sum_test(a, repeats):
    total = 0.0
    for i in prange(a.shape[0]):
        x = a[i]
        acc = 0.0
        for _ in range(repeats):
            acc += np.sin(x) * np.cos(x) + np.sqrt(x + 1.0)
            x += 1e-12
        total += acc
    return total

@njit(parallel=True)
def mark_used_threads(n):
    used = np.zeros(get_num_threads(), dtype=np.int32)
    for i in prange(n):
        tid = get_thread_id()
        if 0 <= tid < used.shape[0]:
            used[tid] = 1
    return used

@njit
def serial_sum_test(a, repeats):
    total = 0.0
    for i in range(a.shape[0]):
        x = a[i]
        acc = 0.0
        for _ in range(repeats):
            acc += np.sin(x) * np.cos(x) + np.sqrt(x + 1.0)
            x += 1e-12
        total += acc
    return total

# ============================================================
# 3) DATA
# ============================================================

N = 8_000_000
REPEATS = 20
a = np.linspace(1.0, 100.0, N, dtype=np.float64)

print_header("WARMUP / JIT COMPILATION")
_ = serial_sum_test(a[:10000], 2)
_ = parallel_sum_test(a[:10000], 2)
_ = mark_used_threads(100000)

try:
    print("Numba threading layer      :", numba.threading_layer())
except Exception as e:
    print("Could not get threading_layer():", e)

# ============================================================
# 4) PARALLELIZATION DIAGNOSTICS
# ============================================================

print_header("PARALLELIZATION DIAGNOSTICS")
parallel_sum_test.parallel_diagnostics(level=4)

# ============================================================
# 5) HOW MANY THREADS ACTUALLY PARTICIPATE?
# ============================================================

print_header("THREADS ACTUALLY USED BY THE KERNEL")
used = mark_used_threads(3_000_000)
used_count = int(used.sum())
print("Threads configured in Numba:", get_num_threads())
print("Threads that participated   :", used_count)
print("Participation map           :", used)

# ============================================================
# 6) MEASURE ACTUAL PROCESS CPU USAGE
# ============================================================

print_header("PROCESS CPU MONITORING")

# First call to stabilize psutil
p.cpu_percent(interval=None)

samples = []

t0 = time.perf_counter()
res = parallel_sum_test(a, REPEATS)
t1 = time.perf_counter()

# Run a longer monitoring phase
p.cpu_percent(interval=None)
a2 = np.linspace(1.0, 100.0, N, dtype=np.float64)
start = time.perf_counter()
while time.perf_counter() - start < 4.0:
    _ = parallel_sum_test(a2, 8)
    samples.append(p.cpu_percent(interval=0.2))

avg_cpu = sum(samples) / len(samples) if samples else 0.0
max_cpu = max(samples) if samples else 0.0

logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
allowed = len(affinity) if affinity is not None else logical
theoretical_max = allowed * 100.0

print("Dummy result                :", res)
print("Main execution time         :", round(t1 - t0, 3), "s")
print("Average process CPU usage   :", round(avg_cpu, 1), "%")
print("Peak process CPU usage      :", round(max_cpu, 1), "%")
print("Theoretical max by affinity :", round(theoretical_max, 1), "%")

if theoretical_max > 0:
    print("Approx. average utilization :", round(100.0 * avg_cpu / theoretical_max, 1), "%")
    print("Approx. peak utilization    :", round(100.0 * max_cpu / theoretical_max, 1), "%")

# ============================================================
# 7) SERIAL VS PARALLEL COMPARISON
# ============================================================

print_header("SERIAL VS PARALLEL COMPARISON")

t2 = time.perf_counter()
res_serial = serial_sum_test(a, REPEATS)
t3 = time.perf_counter()

t4 = time.perf_counter()
res_parallel = parallel_sum_test(a, REPEATS)
t5 = time.perf_counter()

print("Serial time                 :", round(t3 - t2, 3), "s")
print("Parallel time               :", round(t5 - t4, 3), "s")
if (t5 - t4) > 0:
    print("Approx. speedup             :", round((t3 - t2) / (t5 - t4), 2), "x")
print("Absolute result difference  :", abs(res_serial - res_parallel))

# ============================================================
# 8) SIMPLE AUTOMATIC VERDICT
# ============================================================

print_header("AUTOMATIC VERDICT")

problems = []

if used_count <= 1:
    problems.append("Numba is not using multiple threads in practice.")

if theoretical_max > 0 and avg_cpu < 0.5 * theoretical_max:
    problems.append("The process is not using a high fraction of the CPUs available to it.")

if affinity is not None and len(affinity) < (psutil.cpu_count(logical=True) or 999999):
    problems.append("The process has restricted CPU affinity: it cannot see all CPUs on the node.")

if os.environ.get("SLURM_CPUS_PER_TASK") and get_num_threads() != int(os.environ["SLURM_CPUS_PER_TASK"]):
    problems.append("Numba thread count does not match SLURM_CPUS_PER_TASK.")

if (t5 - t4) >= (t3 - t2):
    problems.append("The parallel version does not improve over the serial version in this test.")

if problems:
    print("Possible problems detected:")
    for i, msg in enumerate(problems, 1):
        print(f"{i}. {msg}")
else:
    print("No obvious blocking issues detected. Numba appears to compile, parallelize, and use the CPUs available to the process.")