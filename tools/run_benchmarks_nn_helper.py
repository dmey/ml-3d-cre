import os
import sys
import time
import xarray as xr
import tensorflow as tf

THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, f'{THIS_DIR}/../notebooks')
import utils

results_dir, case_name, num_profiles, device, intra_op_threads, inter_op_threads, nrepeat, batch_size = sys.argv[1:]
num_profiles = int(num_profiles)
batch_size = int(batch_size)
print(f'Batch size: {batch_size}')
nrepeat = int(nrepeat)

TMP_DIR = 'tmp'

tf.config.threading.set_intra_op_parallelism_threads(int(intra_op_threads))
tf.config.threading.set_inter_op_parallelism_threads(int(inter_op_threads))
with tf.device(f'/{device}:0'):
    print('Loading model')
    d = utils.load_training_results(case_name, results_dir)

    print('Preparing data')
    inputs = xr.concat([d.x_train_nn.sel(column=slice(0,1))] * num_profiles * nrepeat, dim='column')
    x = d.model.to_model_input(inputs)
    print('Warming up')
    for _ in range(2):
        d.model.predict(x, batch_size=batch_size)
    print('Starting benchmark')
    t0 = time.time()
    x = d.model.to_model_input(inputs)
    d.model.predict(x, batch_size=batch_size)
    total_duration = time.time() - t0

with open(TMP_DIR + '/bench.txt', 'w') as f:
    f.write(str(total_duration))
