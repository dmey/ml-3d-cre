import os
import sys
import time
import subprocess
from multiprocessing import cpu_count
import xarray as xr
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = f'{THIS_DIR}/..'

# Common
TMP_DIR = 'tmp'
INPUT_DATA = f'{ROOT_DIR}/data/saf/nwp_saf_profiles_in.nc'
NITER = 10
NREPEAT = 10
BATCH_SIZE = 50000
MAX_THREADS = int(os.environ.get('NCORES', cpu_count()))
MEASURE_PARALLEL = False
print(f'Using {MAX_THREADS} threads maximum')

# ecrad
ECRAD_DIR = f'{ROOT_DIR}/ecrad'
NML_PATCH_SCRIPT = f'{THIS_DIR}/change_namelist.sh'
SPARTA_NML = f'{ROOT_DIR}/data/saf/configCY47R1_spartacus.nam'
TRIPLE_NML = f'{ROOT_DIR}/data/saf/configCY47R1_tripleclouds.nam'
OUTPUT_DATA = f'{TMP_DIR}/output.nc'
PATCHED_NML = f'{TMP_DIR}/config.nam'

# NN
NN_RESULTS_DIR = f'{ROOT_DIR}/results'
NN_CASE_NAMES = ['model_lw', 'model_sw']
NN_HELPER_SCRIPT = f'{THIS_DIR}/run_benchmarks_nn_helper.py'

def get_num_profiles():
    ds = xr.open_dataset(INPUT_DATA)
    return ds.dims['column']
NUM_PROFILES = get_num_profiles()

def compute_stats(arr):
    return {
        'mean': np.mean(arr),
        'med': np.median(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr)
    }

def run_ecrad(scheme_name, nml, num_threads):
    os.makedirs(TMP_DIR, exist_ok=True)

    # Set nrepeat so that the simulation itself consumes the bulk of the
    # runtime while the NetCDF loading and writing at the start and end
    # remains negligible. Scales by num_threads to avoid excessive execution time.
    nrepeat = min(num_threads, NREPEAT)
    patches = [f'nrepeat={nrepeat}']
    subprocess.run(['bash', NML_PATCH_SCRIPT, nml, PATCHED_NML] + patches,
                   check=True, capture_output=True)

    print(f'Running ecrad: {num_threads} threads, nrepeat={nrepeat}, {os.path.basename(nml)}')
    durations = []
    for _ in range(NITER):
        t0 = time.time()
        subprocess.run(
            ['bin/ecrad', os.path.abspath(PATCHED_NML),
            os.path.abspath(INPUT_DATA), os.path.abspath(OUTPUT_DATA)],
            cwd=ECRAD_DIR, check=True, env={'OMP_NUM_THREADS': str(num_threads)})
        durations.append((time.time() - t0) * 1000)
    
    durations = np.array(durations) / nrepeat
    speed = (durations * num_threads) / NUM_PROFILES

    return {
        'label': f'{scheme_name} [cpu, omp={num_threads}]',
        'total': compute_stats(durations),
        'speed': compute_stats(speed),
        'durations': durations,
        'num_threads': num_threads
    }

def run_model(device: str, omp_threads, intra_threads, inter_threads, batch_size):
    print(f'Running NN: {device}, {intra_threads} intra threads, '
          f'{inter_threads} inter threads, {omp_threads} OMP threads, {batch_size} batch size ')
    nrepeat = NREPEAT
    durations = []
    for _ in range(NITER):
        duration = 0
        for case_name in NN_CASE_NAMES:
            subprocess.run([sys.executable, NN_HELPER_SCRIPT, 
                NN_RESULTS_DIR, case_name, str(NUM_PROFILES), device,
                str(intra_threads), str(inter_threads), str(nrepeat), str(batch_size)
            ], check=True, env={'OMP_NUM_THREADS': str(omp_threads)})

            with open(TMP_DIR + '/bench.txt') as f:
                duration += float(f.read()) * 1000 # ms
        durations.append(duration)
    
    durations = np.array(durations) / nrepeat
    num_threads = max(omp_threads, intra_threads)
    speed = (durations * num_threads) / NUM_PROFILES

    return {
        'label': f'NN [{device}, intra={intra_threads}, inter={inter_threads}, omp={omp_threads}]',
        'total': compute_stats(durations),
        'speed': compute_stats(speed),
        'durations': durations,
        'num_threads': num_threads
    }

def add_runtimes(r1, r2):
    assert r1['num_threads'] == r2['num_threads']
    durations = r1['durations'] + r2['durations']
    speed = (durations * r1['num_threads']) / NUM_PROFILES
    return {
        'label': r1['label'] + ' + ' + r2['label'],
        'total': compute_stats(durations),
        'speed': compute_stats(speed),
        'durations': durations,
        'num_threads': r1['num_threads']
    }

# ECRAD
runtime_triple_serial = run_ecrad('Tripleclouds', TRIPLE_NML, 1)
runtime_sparta_serial = run_ecrad('SPARTACUS', SPARTA_NML, 1)
if MEASURE_PARALLEL:
    runtime_triple_parallel = run_ecrad('Tripleclouds', TRIPLE_NML, MAX_THREADS)
    runtime_sparta_parallel = run_ecrad('SPARTACUS', SPARTA_NML, MAX_THREADS)

# Neural network
runtime_nn_serial = run_model('cpu', 1, 1, 1, BATCH_SIZE)
if MEASURE_PARALLEL:
    runtime_nn_parallel_cpu_intra_omp = run_model('cpu', MAX_THREADS, MAX_THREADS, 1, BATCH_SIZE)
    # inter=2 after Intel rec: https://software.intel.com/content/www/us/en/develop/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html
    # Though note that our model does not have independent ops, so it shouldn't make a difference.
    runtime_nn_parallel_cpu_intra_inter_2_omp = run_model('cpu', MAX_THREADS, MAX_THREADS, 2, BATCH_SIZE)
    runtime_nn_parallel_cpu_intra = run_model('cpu', 1, MAX_THREADS, 1, BATCH_SIZE)
    runtime_nn_parallel_cpu_omp = run_model('cpu', MAX_THREADS, 1, 1, BATCH_SIZE)

if os.environ.get('GPU') == '1':
    runtime_nn_serial_gpu = run_model('gpu', 1, 1, 1, BATCH_SIZE)
    if MEASURE_PARALLEL:
        runtime_nn_parallel_gpu_intra_omp = run_model('gpu', MAX_THREADS, MAX_THREADS, 1, BATCH_SIZE)
        # inter=2 after Intel rec: https://software.intel.com/content/www/us/en/develop/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html
        # Though note that our model does not have independent ops, so it shouldn't make a difference.
        runtime_nn_parallel_gpu_intra_inter_2_omp = run_model('gpu', MAX_THREADS, MAX_THREADS, 2, BATCH_SIZE)
        runtime_nn_parallel_gpu_intra = run_model('gpu', 1, MAX_THREADS, 1, BATCH_SIZE)
        runtime_nn_parallel_gpu_omp = run_model('gpu', MAX_THREADS, 1, 1, BATCH_SIZE)

# Summary
def print_stats(stats):
    print(stats['label'] + ':')
    print(f"  total: {stats['total']['mean']} ± {stats['total']['std']} ms")
    print(f"    median: {stats['total']['med']}")
    print(f"    min: {stats['total']['min']}")
    print(f"    max: {stats['total']['max']}")
    print(f"  speed: {stats['speed']['mean']} ± {stats['speed']['std']} ms")
    print(f"    median: {stats['speed']['med']}")
    print(f"    min: {stats['speed']['min']}")
    print(f"    max: {stats['speed']['max']}")

print(f'#profiles={NUM_PROFILES}, NN batch size: {BATCH_SIZE}')
print()
print_stats(runtime_triple_serial)
print_stats(runtime_sparta_serial)
if MEASURE_PARALLEL:
    print_stats(runtime_triple_parallel)
    print_stats(runtime_sparta_parallel)

print_stats(runtime_nn_serial)
if MEASURE_PARALLEL:
    print_stats(runtime_nn_parallel_cpu_intra_omp)
    print_stats(runtime_nn_parallel_cpu_intra_inter_2_omp)
    print_stats(runtime_nn_parallel_cpu_omp)
    print_stats(runtime_nn_parallel_cpu_intra)

print_stats(add_runtimes(runtime_triple_serial, runtime_nn_serial))
if MEASURE_PARALLEL:
    print_stats(add_runtimes(runtime_triple_parallel, runtime_nn_parallel_cpu_intra_omp))
    print_stats(add_runtimes(runtime_triple_parallel, runtime_nn_parallel_cpu_intra_inter_2_omp))
    print_stats(add_runtimes(runtime_triple_parallel, runtime_nn_parallel_cpu_omp))
    print_stats(add_runtimes(runtime_triple_parallel, runtime_nn_parallel_cpu_intra))

if os.environ.get('GPU') == '1':
    print('GPU:')
    print_stats(runtime_nn_serial_gpu)
    if MEASURE_PARALLEL:
        print_stats(runtime_nn_parallel_gpu_intra_omp)
        print_stats(runtime_nn_parallel_gpu_intra_inter_2_omp)
        print_stats(runtime_nn_parallel_gpu_omp)
        print_stats(runtime_nn_parallel_gpu_intra)

    print_stats(add_runtimes(runtime_triple_serial, runtime_nn_serial_gpu))
    if MEASURE_PARALLEL:
        print_stats(add_runtimes(runtime_triple_parallel, runtime_nn_parallel_gpu_intra_omp))
        print_stats(add_runtimes(runtime_triple_parallel, runtime_nn_parallel_gpu_intra_inter_2_omp))
        print_stats(add_runtimes(runtime_triple_parallel, runtime_nn_parallel_gpu_omp))
        print_stats(add_runtimes(runtime_triple_parallel, runtime_nn_parallel_gpu_intra))