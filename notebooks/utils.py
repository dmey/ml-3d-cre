from typing import Optional, Union, List, Tuple, NamedTuple, Dict, Callable, Iterator, Any
import os
import shutil
import pickle
import json
import tempfile
import io
import base64

import numpy as np
import scipy
import scipy.stats
import pandas as pd
import xarray as xr
import synthia as syn
import pyvinecopulib as pv
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

HyperParameters = dict



def split_train_val_test(ds, path_to_col_index):
    # Split saf_profile into training/validation/test

    column_idx = np.loadtxt(path_to_col_index)

    ds_train_len = int(len(column_idx) *  0.6)
    ds_val_len = int(len(column_idx) * 0.2)
    ds_test_len = int(len(column_idx) * 0.2)

    assert ds_train_len + ds_val_len + ds_test_len == len(ds.column)

    ds_train_rng = slice(0, ds_train_len)
    ds_val_rng = slice(ds_train_len, ds_train_len + ds_val_len)
    ds_test_rng = slice(ds_train_len + ds_val_len, ds_train_len + ds_val_len + ds_test_len)
    ds_train_val_rng = slice(0, ds_train_len + ds_val_len)

    ds_tran_val = ds.sel(column=ds_train_val_rng)
    ds_test = ds.sel(column=ds_test_rng)

    return ds_tran_val, ds_test

def shuffle_dataset(ds: xr.Dataset, dim: str, seed=None) -> xr.Dataset:
    idx = np.arange(ds.dims[dim])
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(idx)
    return ds.isel({dim: idx})

def train_test_split_dataset(ds: xr.Dataset, dim: str, 
        train_size: Optional[Union[float, int]]=None,
        test_size: Optional[Union[float, int]]=None,
        shuffle=True, seed=None) -> tuple:
    if shuffle:
        ds = shuffle_dataset(ds, dim, seed=seed)
    count = ds.dims[dim]
    if train_size is None:
        assert test_size is not None
        if test_size > 1:
            test_count = int(test_size)
            assert test_count < count
        else:
            test_count = int(count * test_size)
            assert test_count >= 0
        train_count = count - test_count
    else:
        assert test_size is None
        if train_size > 1:
            train_count = int(train_size)
            assert train_count <= count
        else:
            train_count = int(count * train_size)
            assert train_count > 0
        test_count = count - train_count
    train = ds.isel({dim: slice(0, train_count)})
    test = ds.isel({dim: slice(train_count, None)})
    return train, test

def compute_augmented_dataset(ds_true, var_to_synth, synth_mul_factor, uniformization_ratio, stretch_factor, copula_type='gaussian', num_threads=None):
    if synth_mul_factor == 0:
        return ds_true.isel(column=slice(0,1))
    
    if copula_type == 'gaussian':
        pyvinecopulib_ctrl = None
    elif copula_type == 'tll':
        pyvinecopulib_ctrl = pv.FitControlsVinecop(family_set=[pv.tll], trunc_lvl=50, num_threads=num_threads)
    elif copula_type == 'parametric':
        pyvinecopulib_ctrl = pv.FitControlsVinecop(family_set=pv.parametric, trunc_lvl=50, num_threads=num_threads)
    else:
        raise RuntimeError('Copula option not supported')

    generator = syn.CopulaDataGenerator(verbose=True)
    parameterizer = None
    if pyvinecopulib_ctrl:
        generator.fit(ds_true[var_to_synth], copula=syn.VineCopula(controls=pyvinecopulib_ctrl), parameterize_by=parameterizer)
    else:
        generator.fit(ds_true[var_to_synth], copula=syn.GaussianCopula(), parameterize_by=parameterizer)

    n_samples = len(ds_true.column) * synth_mul_factor
    ds_synth_ext = generator.generate(n_samples=n_samples,
        uniformization_ratio=uniformization_ratio, stretch_factor=stretch_factor)

    # Compute expanded dataset
    ds_true_ext = xr.concat([ds_true] * synth_mul_factor, dim='column')
    # Merge synthetic samples with extended dataset
    ds_true_synth_ext = ds_true_ext.copy()
    for name in ds_synth_ext:
        assert name in list(ds_true_synth_ext)
        ds_true_synth_ext[name] = ds_synth_ext[name]
    return ds_true_synth_ext

def compute_optical_depth(ds: xr.Dataset) -> xr.DataArray:
    p_hl_a = ds['pressure_hl'].sel(half_level=slice(1,None))
    p_hl_b = ds['pressure_hl'].sel(half_level=slice(None,-1))
    delta_pressure = p_hl_b - p_hl_a
    delta_pressure = delta_pressure.rename({'half_level': 'level'})
    g = 9.80665 # m/s²
    rho_liquid = 1000 # kg/m³
    rho_ice = 917 # kg/m³
    optical_depth = (ds['q_liquid'] / (rho_liquid * ds['re_liquid']) +\
                     ds['q_ice'] / (rho_ice * ds['re_ice']) ) * delta_pressure / g
    optical_depth = optical_depth.rename('optical_depth_fl')
    return optical_depth

def compute_dz(ds):
    R = 286.9 # J/kg K\n
    rho_fl = ds['pressure_fl'] / ds['temperature_fl'] / R
    g = 9.81 # m/s2
    dz = -1 * ds['pressure_hl'].diff('half_level').rename(half_level='level') / (g * rho_fl)
    # The last level (TOA) is unrepresentative as pressure is 0
    dz[:,-1] = dz[:,-2]
    return dz.rename('dz')

def add_derived_inputs(ds):
    ds = xr.merge([ds, compute_optical_depth(ds), compute_dz(ds)])
    return ds

def add_heating_rates(ds: xr.Dataset, x: xr.Dataset) -> xr.Dataset:
    g_0 = 9.81 # m/s
    c_p = 1004 # J/kg/K
    day_in_s = 86400 # s
    c = day_in_s * (g_0 / c_p)
    
    if 'flux_dn_lw' in ds:
        flux_net_lw = ds.flux_dn_lw - ds.flux_up_lw
        heating_rate_lw = (-1*c*flux_net_lw.diff('half_level')/x.pressure_hl.diff('half_level')).rename({'half_level':'level'})
        ds = ds.assign(heating_rate_lw=heating_rate_lw)
    if 'flux_dn_sw' in ds:
        flux_net_sw = ds.flux_dn_sw - ds.flux_up_sw
        heating_rate_sw = (-1*c*flux_net_sw.diff('half_level')/x.pressure_hl.diff('half_level')).rename({'half_level':'level'})
        ds = ds.assign(heating_rate_sw=heating_rate_sw)
    
    return ds

def compute_norm_stats(ds, root=1):
    ds = ds ** (1/root)
    stats = {
        name : {
            'mean' : ds[name].mean(),
            'std' : ds[name].std()
        } for name in ds
    }
    stats['__root'] = root
    return stats

def normalize_inputs(ds, norm_stats):
    ds_norm = xr.zeros_like(ds)

    # These are already in reasonable scale O(1).
    quantity_no_norm = ['cos_solar_zenith_angle',
                        'sw_albedo', 'lw_emissivity']
    for quantity in list(ds_norm):
        if quantity in quantity_no_norm:
            ds_norm[quantity] = ds[quantity]
            print(f'Skipping normalization for: {quantity}')
        else:
            stats = norm_stats[quantity]
            ds_norm[quantity] = (ds[quantity] ** (1/norm_stats['__root']) - stats['mean']) / stats['std']
    return ds_norm

def normalize_outputs(ds, norm_stats):
    ds_norm = xr.zeros_like(ds)
    for quantity in ds:
        stats = norm_stats[quantity]
        ds_norm[quantity] = (ds[quantity] ** (1/norm_stats['__root']) - stats['mean']) / stats['std']
    return ds_norm

def unnormalize_outputs(ds, norm_stats):
    ds_unnorm = xr.zeros_like(ds)
    for quantity in ds:
        stats = norm_stats[quantity]
        ds_unnorm[quantity] = (ds[quantity] * stats['std'] + stats['mean']) ** norm_stats['__root']
    return ds_unnorm

def reverse_levels(ds: xr.Dataset) -> xr.Dataset:
    for dim in ['level', 'half_level']:
        if dim in ds.dims:
            try:
                ds = ds.drop(dim)
            except:
                pass
            ds = ds.reindex({dim: ds[dim][::-1]}) # 0 = ground
            ds = ds.sortby(dim)
            ds = ds.drop(dim)
    return ds

def load_scheme_inputs(path, only_relevant=True, only_sw=False, only_lw=False, with_pressure_diff=False) -> xr.Dataset:

    ds = xr.open_dataset(path)

    ds['optical_depth'] = compute_optical_depth(ds)

    for name in ['pressure', 'temperature']:
        if f'{name}_fl' not in ds:
            # copied from ecrad/practical/ecradplot/io.py
            ds[f'{name}_fl'] = xr.DataArray(ds[f'{name}_hl'][:,:-1] + 0.5*ds[f'{name}_hl'].diff('half_level'),
                                            coords={'column':ds.column, 'level':ds.level}, 
                                            dims=['column', 'level'])

    # We add pressure difference as it can be useful to tell depth of grid box.
    pressure_diff  = ds.pressure_hl.diff('half_level').rename('pressure_diff').rename(half_level='level')
    ds = xr.merge([ds, pressure_diff])

    if only_relevant:
        ds = ds[['cloud_fraction', 'optical_depth', 'pressure_fl', 'temperature_fl',
                 'cos_solar_zenith_angle', 'sw_albedo',
                 'skin_temperature', 'lw_emissivity',
                 #'o3_vmr',
                 'q',
                 'pressure_diff']]
        if only_sw:
            ds = ds.drop(['skin_temperature', 'lw_emissivity'])
        if only_lw:
            ds = ds.drop(['cos_solar_zenith_angle', 'sw_albedo'])
    else:
        # Does not have 'column' dimension
        ds = ds.drop('o2_vmr')

    if not with_pressure_diff:
        ds = ds.drop('pressure_diff')

    ds = reverse_levels(ds)
    return ds

def load_scheme_outputs(path, only_relevant=True) -> xr.Dataset:
    ds = xr.open_dataset(path)
    if only_relevant:
        ds = ds[['flux_up_lw', 'flux_dn_lw', 'flux_up_sw', 'flux_dn_sw', 'flux_dn_direct_sw']]
    ds = reverse_levels(ds)
    return ds

class StackInfoVar(NamedTuple):
    name: str
    dims: Tuple[str]
    shape: Tuple[int]

StackInfo = List[StackInfoVar]

# Specialized stacking/unstacking functions (as opposed to
# using xarray's to_stacked_array/to_unstacked_dataset).
# This allows to have control over the exact stacking behaviour
# which in turn allows to store compact stacking metadata and use it
# to unstack arbitrary arrays not directly related to the input dataset object.
def to_stacked_array(ds: xr.Dataset, var_names=None, new_dim='stacked', name=None) -> Tuple[xr.DataArray, StackInfo]:
    # Sample dimension must be the first dimension in all variables.
    if not var_names:
        var_names = sorted(ds.data_vars)
    stack_info = []
    var_stacked = []
    for var_name in var_names:
        v = ds.data_vars[var_name]
        if len(v.dims) > 1:
            stacked = v.stack({new_dim: v.dims[1:]})
            stacked = stacked.drop(list(stacked.coords.keys()))
        else:
            stacked = v.expand_dims(new_dim, axis=-1)
        stack_info.append(StackInfoVar(var_name, v.dims, v.shape[1:]))
        var_stacked.append(stacked)
    arr = xr.concat(var_stacked, new_dim)
    if name:
        arr = arr.rename(name)
    return arr, stack_info

def to_unstacked_dataset(arr: np.ndarray, stack_info: StackInfo) -> xr.Dataset:
    unstacked = {}
    curr_i = 0
    for var in stack_info:
        feature_len = 1
        unstacked_shape = [arr.shape[0],]
        for dim_len in var.shape:
            feature_len *= dim_len
            unstacked_shape.append(dim_len)
        var_slice = arr[:, curr_i:curr_i+feature_len]
        var_unstacked = var_slice.reshape(unstacked_shape)
        unstacked[var.name] = xr.DataArray(var_unstacked, dims=var.dims)
        curr_i += feature_len
    ds = xr.Dataset(unstacked)
    return ds

class TrainingData(NamedTuple):
    has_test_data: bool

    # Note: This is the original unnormalized input data.
    x_train: xr.Dataset
    x_test: Optional[xr.Dataset]

    # These are normalized and flattened for input to the NN.
    x_train_flat: np.ndarray
    x_test_flat: Optional[np.ndarray]

    y_true_train: xr.Dataset
    y_true_test: Optional[xr.Dataset]

    y_true_train_flat: np.ndarray
    y_true_test_flat: Optional[np.ndarray]
    
    x_flat_dims: int
    y_stack_info: StackInfo
    sample_idx_train: xr.DataArray
    sample_idx_test: Optional[xr.DataArray]

    @property
    def n_features(self):
        assert self.x_train_flat.ndim == 2
        return self.x_train_flat.shape[1]
    
    @property
    def n_labels(self):
        return self.y_true_train_flat.shape[1]

def create_training_data(x: xr.Dataset, y_true: xr.Dataset,
                         test_size=0.33, apply_pca=False, shuffle=True,
                         x_flat_dims=1) -> TrainingData:    
    if x_flat_dims == 1:
        x_flat, _ = to_stacked_array(x)
    elif x_flat_dims == 2:
        assert all(arr.dims == ('column', 'level') for arr in x.values()),\
            'All input variables must be <column>x<level>'
        x_flat = x.to_stacked_array('features', ['column', 'level'])
    else:
        raise ValueError
    y_true_flat, y_stack_info = to_stacked_array(y_true)

    # Make column index explicit so it survives shuffling.
    x_flat = x_flat.assign_coords(column=x_flat.column)

    has_test_data = test_size > 0

    if has_test_data:
        x_train_flat, x_test_flat, y_true_train_flat, y_true_test_flat = train_test_split(
            x_flat, y_true_flat, test_size=test_size, random_state=42, shuffle=shuffle)
    else:
        # No shuffle needed here as shuffling is done during training anyway.
        x_train_flat = x_flat
        y_true_train_flat = y_true_flat
        x_test_flat = None
        y_true_test_flat = None

    sample_idx_train = x_train_flat.column
    sample_idx_test = x_test_flat.column if has_test_data else None

    x_train = x.isel(column=sample_idx_train)
    x_test = x.isel(column=sample_idx_test) if has_test_data else None

    # TensorFlow cannot handle xarray DataArray objects, so convert to numpy arrays.
    def to_numpy(arr):
        try:
            return arr.values
        except:
            return arr
    x_train_flat = to_numpy(x_train_flat)
    y_true_train_flat = to_numpy(y_true_train_flat)
    x_test_flat = to_numpy(x_test_flat)
    y_true_test_flat = to_numpy(y_true_test_flat)

    if apply_pca:
        pca = PCA(.99).fit(x_train_flat)
        x_train_flat = pca.transform(x_train_flat)
        if has_test_data:
            x_test_flat = pca.transform(x_test_flat)

    # Restore original structure for further processing after training.
    y_true_train = to_unstacked_dataset(y_true_train_flat, y_stack_info)
    y_true_train = y_true_train.assign_coords(column=sample_idx_train)
    if has_test_data:
        y_true_test = to_unstacked_dataset(y_true_test_flat, y_stack_info)
        y_true_test = y_true_test.assign_coords(column=sample_idx_test)
    else:
        y_true_test = None

    data = TrainingData(
        has_test_data=has_test_data,
        x_train=x_train, x_test=x_test,
        x_train_flat=x_train_flat, x_test_flat=x_test_flat,
        y_true_train=y_true_train, y_true_test=y_true_test,
        y_true_train_flat=y_true_train_flat, y_true_test_flat=y_true_test_flat,
        x_flat_dims=x_flat_dims,
        y_stack_info=y_stack_info,
        sample_idx_train=sample_idx_train, sample_idx_test=sample_idx_test)
    return data

def rmtree(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def save(data, path):
    if path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        with open(path, 'wb') as f:
            pickle.dump(data, f)

def load(path):
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)

class TrainedModel(NamedTuple):
    model: keras.Model
    hp: HyperParameters
    history: pd.DataFrame
    x_flat_dims: int
    y_stack_info: StackInfo

    def to_model_input(self, x: xr.Dataset) -> np.ndarray:
        # Note: Dataset is assumed to be loaded by load_scheme_inputs().
        if self.x_flat_dims == 1:
            x_flat, _ = to_stacked_array(x)
        elif self.x_flat_dims == 2:
            assert all(arr.dims == ('column', 'level') for arr in x.values()),\
                'All input variables must be <column>x<level>'
            x_flat = x.to_stacked_array('features', ['column', 'level'])
        else:
            raise ValueError
        x_flat = x_flat.values
        return x_flat
    
    def predict(self, x: np.ndarray, **kw) -> xr.Dataset:
        y_pred = self.model.predict(x, **kw)
        y_pred_unstacked = to_unstacked_dataset(y_pred, self.y_stack_info)
        return y_pred_unstacked

def train_model(build_model_fn: Callable[[TrainingData,HyperParameters],keras.Model],
                data: TrainingData, hp: HyperParameters,
                epochs=100, batch_size=32,
                early_stopping=False, early_stopping_patience=50,
                extra_callbacks=None,
                verbose=1, save_best=True,
                ) -> TrainedModel:

    callbacks = []
    if data.has_test_data:
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience)
            callbacks.append(early_stop)
        val_kwargs = dict(validation_data=(data.x_test_flat, data.y_true_test_flat))
    else:
        val_kwargs = dict(validation_split=0)

    if extra_callbacks:
        callbacks += extra_callbacks

    model_fn = lambda hp: build_compiled_model(build_model_fn, data, hp)

    if save_best:
        tmp_dir = tempfile.mkdtemp(prefix='phd-rads')
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=tmp_dir,
            save_best_only=True,
            monitor='val_mse' if data.has_test_data else 'mse',
            verbose=0,
            save_weights_only=False,
            save_freq='epoch'
            )
        callbacks.append(save_callback)

    fit_kwargs = dict(
        x=data.x_train_flat, y=data.y_true_train_flat,
        epochs=epochs, batch_size=batch_size, verbose=verbose,
        callbacks=callbacks,
        **val_kwargs)

    model = model_fn(hp)
    history = model.fit(**fit_kwargs)
    history = pd.DataFrame(history.history)
    history.index.name = 'epoch'

    if save_best:
        model = tf.keras.models.load_model(tmp_dir)
        shutil.rmtree(tmp_dir)
    
    trained_model = TrainedModel(model, hp, history, data.x_flat_dims, data.y_stack_info)
    return trained_model

class DiskTrainingResults(NamedTuple):
    model: TrainedModel
    
    has_test_data: bool

    # Scheme inputs
    x_train: xr.Dataset
    x_test: Optional[xr.Dataset]

    # NN inputs (may have fewer features)
    x_train_nn: xr.Dataset
    x_test_nn: Optional[xr.Dataset]
    
    # Scheme outputs
    y_triple_train: xr.Dataset
    y_triple_test: Optional[xr.Dataset]
    y_sparta_train: xr.Dataset
    y_sparta_test: Optional[xr.Dataset]
    
    # Ground truth for NN (in our case, either y_sparta or y_triple - y_sparta)
    y_true_train: xr.Dataset
    y_true_test: Optional[xr.Dataset]
    
    # Predictions of NN (in our case, either y_sparta or y_triple - y_sparta)
    y_pred_train: xr.Dataset
    y_pred_test: Optional[xr.Dataset]

def save_training_results(name: str, model: TrainedModel, train_data: TrainingData,
                          ds_inputs_all: xr.Dataset,
                          ds_triple: xr.Dataset, ds_sparta: xr.Dataset, skip_save=False,
                          save_dir='results') -> DiskTrainingResults:
    if not skip_save:
        exp_dir = os.path.join(save_dir, name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save keras model in case we need it again.
        model_path = os.path.join(exp_dir, 'model')
        model.model.save(model_path)

    # Save all inputs subset to train/test data.
    x_train = ds_inputs_all.isel(column=train_data.sample_idx_train)
    if train_data.has_test_data:
        x_test = ds_inputs_all.isel(column=train_data.sample_idx_test)
    else:
        x_test = None

    # Save Tripleclouds & Spartacus outputs subset to train/test data.
    y_triple_train = ds_triple.isel(column=train_data.sample_idx_train)
    y_sparta_train = ds_sparta.isel(column=train_data.sample_idx_train)

    if train_data.has_test_data:
        y_triple_test = ds_triple.isel(column=train_data.sample_idx_test)
        y_sparta_test = ds_sparta.isel(column=train_data.sample_idx_test)
    else:
        y_triple_test = None
        y_sparta_test = None

    # Save predictions using training and test data.
    y_pred_train = model.predict(train_data.x_train_flat).assign_coords(column=train_data.sample_idx_train)
    if train_data.has_test_data:
        y_pred_test = model.predict(train_data.x_test_flat).assign_coords(column=train_data.sample_idx_test)
    else:
        y_pred_test = None
    
    if not skip_save:
        disk_trained_model = model._replace(model=None)
    else:
        disk_trained_model = model

    r = DiskTrainingResults(
        model=disk_trained_model,
        has_test_data=train_data.has_test_data,
        x_train=x_train.drop('column'),
        x_test=x_test.drop('column') if train_data.has_test_data else None,
        x_train_nn=train_data.x_train.drop('column'),
        x_test_nn=train_data.x_test.drop('column') if train_data.has_test_data else None,
        y_triple_train=y_triple_train.drop('column'),
        y_triple_test=y_triple_test.drop('column') if train_data.has_test_data else None,
        y_sparta_train=y_sparta_train.drop('column'),
        y_sparta_test=y_sparta_test.drop('column') if train_data.has_test_data else None,
        y_true_train=train_data.y_true_train.drop('column'),
        y_true_test=train_data.y_true_test.drop('column') if train_data.has_test_data else None,
        y_pred_train=y_pred_train.drop('column'),
        y_pred_test=y_pred_test.drop('column') if train_data.has_test_data else None,
        )

    if not skip_save:
        print('Saving results')
        save(r, os.path.join(exp_dir, 'results.pkl'))
        r = r._replace(model=model)

    return AttrDict(r._asdict())

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def load_training_results(name: str, folder='results') -> DiskTrainingResults:
    folder = os.path.join(folder, name)
    results = load(os.path.join(folder, 'results.pkl'))
    model = tf.keras.models.load_model(os.path.join(folder, 'model'))
    results = results._replace(model=results.model._replace(model=model))
    results = AttrDict(results._asdict())
    return results

@tf.keras.utils.register_keras_serializable(package='Custom', name='var_reg')
class VarianceRegularizer(regularizers.Regularizer):
    def __init__(self, var_factor=0.):
        self.var_factor = var_factor

    def __call__(self, x):
        regularization = 0.
        if self.var_factor:
            regularization += self.var_factor * tf.math.reduce_variance(x)
        return regularization

    def get_config(self):
        return {'var_factor': self.var_factor}

@tf.keras.utils.register_keras_serializable(package='Custom', name='combo_reg')
class ComboRegularizer(regularizers.Regularizer):
    def __init__(self, l1_factor=0., l2_factor=0., var_factor=0.):
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.l1_l2 = keras.regularizers.l1_l2(l1_factor, l2_factor)
        self.var = VarianceRegularizer(var_factor)

    def __call__(self, x):
        regularization = 0.
        regularization += self.l1_l2(x)
        regularization += self.var(x)
        return regularization

    def get_config(self):
        return {
            'l1_factor': self.l1_factor,
            'l2_factor': self.l2_factor,
            'var_factor': self.var.var_factor
        }

def create_regularizer(hp: HyperParameters) -> Optional[regularizers.Regularizer]:
    l1_penalty = hp.get('l1_penalty', 0.)
    l2_penalty = hp.get('l2_penalty', 0.)
    var_regularizer_factor = hp.get('var_regularizer_factor', 0.)
    if l1_penalty or l2_penalty or var_regularizer_factor:
        regularizer = ComboRegularizer(l1_penalty, l2_penalty, var_regularizer_factor)
    else:
        regularizer = None
    return regularizer

def build_compiled_model(build_model_fn: Callable[[TrainingData,HyperParameters],keras.Model],
                         data: TrainingData, hp: HyperParameters) -> keras.Model:

    model = build_model_fn(data, hp)

    optimizer = tf.keras.optimizers.Adam(hp['learning_rate'])

    loss = hp.get('loss', 'mse')
    if loss == 'huber':
        loss = tf.keras.losses.Huber()
    
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    model.summary()
    return model

def build_dense_nn_model(data: TrainingData, hp: HyperParameters) -> keras.Model:
    regularizer = create_regularizer(hp)
    model = keras.Sequential()
    model.add(layers.InputLayer([data.n_features]))
    if hp['dropout_ratio_input'] > 0:
        model.add(layers.Dropout(hp['dropout_ratio_input'], kernel_regularizer=regularizer))
    for _ in range(hp['n_hidden_layers']):
        model.add(layers.Dense(hp['hidden_size'], activation=hp['activation'], kernel_regularizer=regularizer))
        if hp['dropout_ratio_hidden'] > 0:
            model.add(layers.Dropout(hp['dropout_ratio_hidden'], kernel_regularizer=regularizer))
    model.add(layers.Dense(data.n_labels, activation='linear', kernel_regularizer=regularizer))
    return model

def build_rnn_model(data: TrainingData, hp: HyperParameters) -> keras.Model:
    assert data.x_train_flat.ndim == 3
    n_levels = data.x_train_flat.shape[1]
    n_quantities = data.x_train_flat.shape[2]

    regularizer = create_regularizer(hp)
    model = keras.Sequential()
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    model.add(layers.InputLayer([n_levels, n_quantities]))

    assert hp['rnn_type'] in ['SimpleRNN', 'GRU', 'LSTM']
    assert hp['rnn_direction'] in ['fwd', 'bwd', 'bi']
    rnn_class = getattr(layers, hp['rnn_type'])
    go_backwards = hp['rnn_direction'] == 'bwd'

    for _ in range(hp['n_hidden_layers'] - 1):
        l = rnn_class(hp['hidden_size'], return_sequences=True,
                go_backwards=go_backwards,
                activation=hp['activation'],
                kernel_regularizer=regularizer)
        if hp['rnn_direction'] == 'bi':
            l = layers.Bidirectional(l)
        model.add(l)
    l = rnn_class(hp['hidden_size'], return_sequences=False, 
        go_backwards=go_backwards,
        activation=hp['activation'],
        kernel_regularizer=regularizer)
    if hp['rnn_direction'] == 'bi':
        l = layers.Bidirectional(l)
    model.add(l)

    model.add(layers.Dense(data.n_labels,
                           activation='linear',
                           # not sure if zeros are a good idea
                           kernel_initializer=tf.initializers.zeros(),
                           kernel_regularizer=regularizer))
    
    return model

def compute_error_metrics_summary(y_true: xr.Dataset, y_pred: xr.Dataset) -> Dict[str,float]:
    error = y_true - y_pred
    error, _ = to_stacked_array(error)
    mse = (error*error).mean()
    y_true_flat = to_stacked_array(y_true)[0].values.ravel()
    y_pred_flat = to_stacked_array(y_pred)[0].values.ravel()
    pearson_coef, pearson_p = scipy.stats.pearsonr(y_true_flat, y_pred_flat)
    return {
        'mbe': error.mean().item(),
        'mae': np.fabs(error).mean().item(),
        'mse': mse.item(),
        'rmse': np.sqrt(mse).item(),
        'std': error.std().item(),
        'pearson_coef': pearson_coef,
        'pearson_p': pearson_p,
    }

def compute_error_metrics_per_quantity(y_true: xr.Dataset, y_pred: xr.Dataset) -> pd.DataFrame:
    error = (y_true - y_pred).to_dataframe()
    mse = (error*error).mean()

    var_names = []
    pearson_coef = []
    pearson_p = []
    for name in y_true:
        var_names.append(name)
        pear_coef, pear_p = scipy.stats.pearsonr(y_true[name].values.ravel(), y_pred[name].values.ravel())
        pearson_coef.append(pear_coef)
        pearson_p.append(pear_p)

    df = pd.DataFrame({
        'mbe': error.mean(),
        'mae': np.fabs(error).mean(),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'std': error.std(),
        'pearson_coef': {name: pear_coef for name, pear_coef in zip(var_names, pearson_coef)},
        'pearson_p': {name: pear_p for name, pear_p in zip(var_names, pearson_p)},
    })
    df.index.name = 'quantity'
    return df

def compute_error_metrics(y_true: xr.Dataset, y_pred: xr.Dataset) -> pd.DataFrame:
    per_quantity = compute_error_metrics_per_quantity(y_true, y_pred)
    summary = compute_error_metrics_summary(y_true, y_pred)
    summary = pd.DataFrame(summary, index=['all'])
    summary.index.name = 'quantity'
    return pd.concat([summary, per_quantity])

def compute_metrics_per_level(y_true: xr.Dataset, quantity: str,
                              percentiles=[10, 25, 75, 90]) -> pd.DataFrame:
    percentiles = np.asanyarray(percentiles)
    vals = y_true[quantity]
    dim = 'column'
    
    percentiles_bias = vals.quantile(percentiles / 100, dim)
    percentiles_abs = np.fabs(vals).quantile(percentiles / 100, dim)

    return pd.DataFrame({
        'mean_bias': vals.mean(dim),
        'std_bias': vals.std(dim),
        'mean_abs': np.fabs(vals).mean(dim),
        **{f'{p}th_percentile_bias': percentiles_bias[i]
           for i, p in enumerate(percentiles)},
        **{f'{p}th_percentile_abs': percentiles_abs[i]
           for i, p in enumerate(percentiles)},
    }, index=vals.level)

def compute_error_metrics_per_level(y_true: xr.Dataset, y_pred: xr.Dataset, quantity: str,
                                    percentiles=[10, 25, 75, 90]) -> pd.DataFrame:
    percentiles = np.asanyarray(percentiles)
    error = y_true[quantity] - y_pred[quantity]
    dim = 'column'
    mse = (error*error).mean(dim)

    pearson_coef = []
    pearson_p = []
    for level in y_true.level:
        pear_coef, pear_p = scipy.stats.pearsonr(
            y_true[quantity].isel(level=level).values.ravel(),
            y_pred[quantity].isel(level=level).values.ravel())
        pearson_coef.append(pear_coef)
        pearson_p.append(pear_p)
    
    percentiles_bias_error = error.quantile(percentiles / 100, dim)
    percentiles_abs_error = np.fabs(error).quantile(percentiles / 100, dim)

    return pd.DataFrame({
        'mean_bias_error': error.mean(dim),
        'std_bias_error': error.std(dim),
        **{f'{p}th_percentile_bias_error': percentiles_bias_error[i]
           for i, p in enumerate(percentiles)},
        **{f'{p}th_percentile_abs_error': percentiles_abs_error[i]
           for i, p in enumerate(percentiles)},
        'mean_abs_error': np.fabs(error).mean(dim),
        'mean_squared_error': mse,
        'root_mean_squared_error': np.sqrt(mse),
        'pearson_coef': pearson_coef,
        'pearson_p': pearson_p,
    }, index=error.level)

def plot_inputs(ds, is_normalized):
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    for quantity in list(ds):
        if len(ds[quantity].shape) == 1: # scalars
            ds[quantity].plot.hist(ax=ax[0], label=quantity, alpha=0.3)
            ax[0].set_ylabel('Count')
            ax[0].set_xlabel('Range')
            ax[0].legend()
        elif len(ds[quantity].shape) == 2: # profiles
            ds[quantity].mean('column').plot(ax=ax[1], label=quantity)
            if is_normalized:
                ax[1].set_ylabel('Normalized range (Z score)')
            else:
                ax[1].set_ylabel('Range')
            ax[1].set_xlabel('Vertical level')
            ax[1].set_title('Mean profiles')
            ax[1].legend()
            ax
        else:
            raise RuntimeError('Number of dims not supported')


def plot_scatter(x, y, quantitiy, ax, x_axis_label=None):
    q_l_map = {'flux_dn_lw' : 0,
            'flux_up_lw' : -1,
            'flux_dn_sw' : 0,
            'flux_dn_direct_sw' : 0,
            'flux_up_sw' : -1}

    x = x.sel(level=q_l_map[quantitiy])
    y = y.sel(level=q_l_map[quantitiy])
    ax.scatter(x, y,
        s=100, facecolors='black', edgecolors='black', alpha=0.1)

    l_name = 'BOA' if q_l_map[quantitiy] == 0 else 'TOA'
    ax.text(0.05, 0.9, l_name, transform=ax.transAxes)

    x_y_lim_min = xr.concat([x,y], dim='column').min()
    x_y_lim_max = xr.concat([x,y], dim='column').max()
    ax.set_xlim(x_y_lim_min, x_y_lim_max)
    ax.set_ylim(x_y_lim_min, x_y_lim_max)

    if x_axis_label is None:
        x_axis_label = '3D signal in W m⁻²'

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(f'3D prediction in W m⁻²')

    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="-", c="r", linewidth=3)


def multi_plot(y_sparta: xr.Dataset, y_triple: xr.Dataset, y_pred: xr.Dataset, x: xr.Dataset,
        y_axis='level', y_axis_label=None, log_y=False,
        x_axis_label=None, variant='fluxes', is_diff=False):

    # https://matplotlib.org/3.3.3/tutorials/introductory/customizing.html
    mpl_params = {
        'font.size': 23,
        'axes.titlesize': 20,
        'axes.labelsize': 30,
        'axes.linewidth': 3,
        'xtick.major.size': 10,
        'xtick.major.width': 3,
        'ytick.major.size': 10,
        'ytick.major.width': 3,
        'ytick.right': True,
        'xtick.top': True,
        'xtick.direction': 'inout',
        'ytick.direction': 'inout',
        'lines.linewidth': 6
    }
   
    if variant == 'fluxes':
        d_names = {
            'flux_dn_lw' : 'Downwelling longwave',
            'flux_up_lw' : 'Upwelling longwave',
            'flux_dn_sw' : 'Total downwelling shortwave',
            'flux_dn_direct_sw' : 'Direct downwelling shortwave',
            'flux_up_sw' : 'Upwelling shortwave',
        }
    elif variant == 'hr':
        d_names = {
            'heating_rate_lw' : 'Longwave heating rate',
            'heating_rate_sw' : 'Shortwave heating rate',
        }
    else:
        assert False

    with plt.rc_context(mpl_params):        

        n_rows, n_cols = len(d_names), 3

        # Row captions (https://stackoverflow.com/a/27430940)
        fig, big_axs = plt.subplots(n_rows, 1, figsize=(8*n_cols, 8*n_rows))
        for i, quantity in enumerate(d_names.values()):
            big_axs[i].set_title(quantity, pad=43, fontsize=40)
            big_axs[i].axis('off')

        # Best effort, may be off by one sometimes
        n_x_ticks = 6

        axs = np.empty((n_rows, n_cols), dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                axs[i,j] = fig.add_subplot(n_rows, n_cols, 1 + i * n_cols + j)

                # Tick count
                axs[i,j].xaxis.set_major_locator(plt.MaxNLocator(n_x_ticks - 1, min_n_ticks=n_x_ticks))

                # Scientific notation
                axs[i,j].ticklabel_format(axis='x', scilimits=(-2, 2))

        show_legend = True

        for i, quantity in enumerate(d_names.keys()):
            x_label = '' if i + 1 < n_rows else x_axis_label
            j = 0

            if variant == 'hr':
                xlim_level = 90
                plot_shaded_error_boxplot(y_sparta + y_sparta, y_sparta, y_pred, x, quantity, y_axis, y_axis_label, log_y, x_label, ax=axs[i,j], y_ticks_and_labels=True, counter=i, show_legend=show_legend, xlim_level=xlim_level, alt_color=True)
                j += 1

            plot_shaded_error_boxplot(y_sparta, y_triple, y_pred, x, quantity, y_axis, y_axis_label, log_y, x_label, ax=axs[i,j], y_ticks_and_labels=variant == 'fluxes', counter=i, show_legend=show_legend, with_pred=True)
            j += 1
            plot_shaded_error_boxplot(y_sparta, y_pred, y_pred, x, quantity, y_axis, y_axis_label, log_y, x_label, ax=axs[i,j], y_ticks_and_labels=False, is_error=True, counter=i, show_legend=show_legend, with_mae=True)
            j += 1
            
            if quantity not in ['heating_rate_sw', 'heating_rate_lw']:
                if is_diff:
                    plot_scatter(y_sparta[quantity] - y_triple[quantity], 
                        y_pred[quantity] - y_triple[quantity], quantity, axs[i,j], x_axis_label=x_label)
                else:
                    plot_scatter(y_sparta[quantity],
                        y_pred[quantity], quantity, axs[i,j], x_axis_label=x_label)
        
        add_subfigure_labels(axs, yloc=1.02)

        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                if variant == 'fluxes' and j == axs.shape[1] - 1:
                    continue
                axs[i,j].axvline(0, linestyle='dotted', linewidth=3, color='black')
        
        h_pad = 0.0
        if variant == 'fluxes':
            # reduce spacing and then push out last plot in each row
            fig.tight_layout(w_pad=-3.0, h_pad=h_pad)
            for i in range(axs.shape[0]):
                ax = axs[i, -1]
                left, bottom, width, height = ax.get_position().bounds
                ax.set_position((left + 0.06, bottom, width, height))
        else:
            fig.tight_layout(h_pad=h_pad)

        fig.align_ylabels(axs[:, -1])

def add_subfigure_labels(axes, xloc=0.0, yloc=1.05, zorder=0, label_list=[], flatten_order='C'):
    if label_list == []:
        import string
        labels = string.ascii_lowercase
    else:
        labels = label_list
        
    for i, ax in enumerate(axes.flatten(order=flatten_order)):
        ax.text(xloc, yloc, "[%s]" %(labels[i]), va='baseline', transform=ax.transAxes, zorder=zorder)
        
def plot_shaded_error_boxplot(
        y_sparta: xr.Dataset, y_triple: xr.Dataset, y_pred: xr.Dataset, x: xr.Dataset, quantity: str,
        y_axis='level', y_axis_label=None, log_y=False,
        x_axis_label=None, log_x=False, stat='bias',
        ax=None, y_ticks_and_labels=True, is_error=False, counter=None, show_legend=True,
        with_mae=False, with_pred=False, alt_color=False,
        xlim_level=None):

    assert y_triple[quantity].ndim == 2, f'{quantity} has ndim={y_triple[quantity].ndim}'

    percentiles = [5, 25, 75, 95]
    color_error = '#595BD4'
    if alt_color:
        color_normal = color_error if is_error else '#FE6100'
    else:
        color_normal = color_error if is_error else '#C375C5'
    color_normal_2 = color_error if is_error else '#00C1AD'
    alphas = [0.15, 0.2]

    if x_axis_label is None:
        if 'heating_rate' in quantity:
            x_axis_label = r'$\dfrac{dT}{dt}$ in K d$^{-1}$'
        else:
            x_axis_label = 'Flux in W m⁻²'
    if y_axis_label is None:
        y_axis_label = y_axis

    stats = compute_error_metrics_per_level(y_sparta, y_triple, quantity, percentiles)
    
    y_coords = x[y_axis]
    if y_coords.ndim == 2:
        y_coords = y_coords.mean(axis=0)
    stats['y_coords'] = y_coords

    if 'heating_rate' in quantity:
        if alt_color:
            legend_label = 'Mean error' if is_error else 'SPARTACUS'
        else:
            legend_label = 'Mean error' if is_error else '3D signal'
            legend_pred_label = 'Mean error' if is_error else '3D prediction'
    else:
        legend_label = 'Mean error' if is_error else '3D signal'
        legend_pred_label = 'Mean error' if is_error else '3D prediction'

    stats.plot(ax=ax, x=f'mean_{stat}_error', y='y_coords',
               logx=log_x, logy=log_y,
               legend=False, color=color_normal, label=legend_label)

    if with_pred:
        stats_pred = compute_metrics_per_level(y_pred - y_triple, quantity, percentiles)
        stats_pred['y_coords'] = y_coords
        stats_pred.plot(ax=ax, x=f'mean_{stat}', y='y_coords',
                logx=log_x, logy=log_y,
                legend=False, color=color_normal_2, label=legend_pred_label)

    if with_mae:
        legend_label = 'Mean absolute error' if is_error else 'MA(B)'
        stats.plot(ax=ax, x='mean_abs_error', y='y_coords',
                logx=log_x, logy=log_y,
                legend=False, color=color_normal_2, ls='dashed',
                label=legend_label
                ) 

    for i in range(len(percentiles) // 2):
        color = color_normal
        left_percentile = stats[f'{percentiles[i]}th_percentile_{stat}_error']
        right_percentile = stats[f'{percentiles[-1-i]}th_percentile_{stat}_error']
        ax.fill_betweenx(y_coords,
                         left_percentile,
                         right_percentile,
                         color=color, alpha=alphas[i],
                         #label=f"{percentiles[i]}-{100-percentiles[i]} %"
                         )

    if with_pred:
        for i in range(len(percentiles) // 2):
            color = color_normal_2
            left_percentile = stats_pred[f'{percentiles[i]}th_percentile_{stat}']
            right_percentile = stats_pred[f'{percentiles[-1-i]}th_percentile_{stat}']
            ax.fill_betweenx(y_coords,
                            left_percentile,
                            right_percentile,
                            color=color, alpha=alphas[i],
                            #label=f"{percentiles[i]}-{100-percentiles[i]} %"
                            )

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    
    if xlim_level:
        i = len(percentiles) - 1
        left_percentile = stats[f'{percentiles[i]}th_percentile_{stat}_error'][:xlim_level]
        right_percentile = stats[f'{percentiles[-1-i]}th_percentile_{stat}_error'][:xlim_level]
        combined = np.concatenate([left_percentile, right_percentile])
        xlim_min = combined.min()
        xlim_max = combined.max()
        padding = abs((xlim_max - xlim_min) * 0.1)
        xlim_min -= padding
        xlim_max += padding
        ax.set_xlim(xlim_min, xlim_max)
        ax.autoscale(axis='y')
    else:
        ax.autoscale()

    ax.invert_yaxis()
    if not y_ticks_and_labels:
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.set_ylabel('')
    if counter == 0 and show_legend:
        ax.legend(loc='upper right')
      
def plt_show_svg(fig_or_path=None):
    from IPython.display import HTML
    if fig_or_path is not None and hasattr(fig_or_path, 'to_image'):
        # plotly
        svg = fig_or_path.to_image(format="svg")
    elif isinstance(fig_or_path, str):
        with open(fig_or_path, 'rb') as f:
            svg = f.read()
    else:
        if fig_or_path is None:
            fig_or_path = plt.gcf()
        f = io.BytesIO()
        fig_or_path.savefig(f, format='svg', bbox_inches='tight')
        plt.close(fig_or_path)
        svg = f.getvalue()
    svg_url = 'data:image/svg+xml;base64,' + base64.b64encode(svg).decode()
    display(HTML(f'<img src="{svg_url}"></img>'))


def rescale(y_pred, pressure_hl, sw_albedo):
    # SECTION 2: postprocess neural-network prediction to obtain
    # upwelling and downwelling flux, and consistent heating rate

    y_pred = reverse_levels(y_pred)
    pressure_hl = reverse_levels(pressure_hl)

    # Work out profile of the divergence of the net flux from the heating
    # rate
    g=9.80665
    Cp=1004.0
    scaling = 3600.0*24.0*g/Cp

    net_flux_div_lw_from_hr = -y_pred['heating_rate_lw'] * np.diff(pressure_hl) / scaling
    net_flux_div_sw_from_hr = -y_pred['heating_rate_sw'] * np.diff(pressure_hl) / scaling

    # Work out the total atmospheric divergence from the heating rate =
    # the sum of the divergence across each layer
    total_div_sw_from_hr = net_flux_div_sw_from_hr.sum(dim='level')
    total_div_lw_from_hr = net_flux_div_lw_from_hr.sum(dim='level')

    # Total divergence from scalar fluxes: in longwave we can assume that
    # the downward flux at TOA is zero, and that the 3D effect on the
    # upward flux at the surface is also zero. Therefore the net flux
    # (=down-up) is equal to minus the scalar flux at TOA, and is equal to
    # the scalar flux at the surface. The total atmospheric flux
    # divergence is the surface net flux minus the TOA net flux.
    # Therefore the total atmospheric flux divergence is the sum of the
    # scalar flux at TOA and surface.
    total_div_lw_from_scalar_flux = y_pred['flux_scalar_lw'].sel(half_level=0) + \
                                    y_pred['flux_scalar_lw'].sel(half_level=-1)

    # Scale the net flux divergence profile from the heating rates to
    # match the total divergence from the scalar fluxes
    lw_scaling = total_div_lw_from_scalar_flux / total_div_lw_from_hr

    # If the heating rates, and hence the denominator in the previous
    # formula, are very small then the scaling may be very large - cap it
    # to lie between 0.5 and 2.
    lw_scaling = np.maximum(0.5, np.minimum(lw_scaling, 2.0))

    # Apply the scaling
    net_flux_div_lw = net_flux_div_lw_from_hr * lw_scaling

    # Compute the net flux profile by integrating its divergence, starting
    # from the known value at the top (=minus the scalar flux )
    net_flux_lw = -y_pred['flux_scalar_lw'].isel(half_level=0).values[:, np.newaxis] + \
        np.concatenate([np.zeros((net_flux_div_lw.shape[0], 1)), net_flux_div_lw.cumsum(dim='level')], axis=1)
    
    # Reconstruct the up and down from the scalar and the net
    flux_dn_lw_3d = 0.5 * (y_pred['flux_scalar_lw'] + net_flux_lw)
    flux_up_lw_3d = 0.5 * (y_pred['flux_scalar_lw'] - net_flux_lw)

    # Compute the new heating rate
    hr_lw_3d = -scaling * net_flux_div_lw / np.diff(pressure_hl)

    y_pred['flux_dn_lw_rescaled'] = flux_dn_lw_3d
    y_pred['flux_up_lw_rescaled'] = flux_up_lw_3d
    y_pred['heating_rate_lw_rescaled'] = hr_lw_3d

    # Shortwave
    # Total divergence from scalar fluxes: in the shortwave we assume the
    # downward flux at TOA has no 3D component so the 3D effect on
    # downward here is zero.  Therefore the 3D effect on net flux here is
    # equal to minus the 3D effect on scalar flux here. At the surface, we
    # know that net=down-up, scalar=down+up and albedo=up/down. Therefore
    # net=scalar*(1-albedo)/(1+albedo). The total atmospheric flux
    # divergence is the surface net flux minus the TOA net flux. Therefore
    # the total atmospheric flux divergence is the sum of the scalar flux
    # at TOA and the net flux (from the formula above) at the surface.
    total_div_sw_from_scalar_flux = y_pred['flux_scalar_sw'].sel(half_level=-1) * \
                                    (1 - sw_albedo) / (1 + sw_albedo) + \
                                    y_pred['flux_scalar_sw'].sel(half_level=0)

    # Scale the net flux divergence profile from the heating rates to
    # match the total divergence from the scalar fluxes
    sw_scaling = total_div_sw_from_scalar_flux / total_div_sw_from_hr

    # If the heating rates, and hence the denominator in the previous
    # formula, are very small then the scaling may be very large - cap it
    # to lie between 0.5 and 2.
    sw_scaling = np.maximum(0.5, np.minimum(sw_scaling, 2.0))

    # Apply the scaling
    net_flux_div_sw = net_flux_div_sw_from_hr * sw_scaling

    # Compute the net flux profile by integrating its divergence, starting
    # from the known value at the top (=minus the scalar flux )
    net_flux_sw = -y_pred['flux_scalar_sw'].isel(half_level=0).values[:, np.newaxis] + \
        np.concatenate([np.zeros((net_flux_div_sw.shape[0], 1)), net_flux_div_sw.cumsum(dim='level')], axis=1)

    # Reconstruct the up and down from the scalar and the net
    flux_dn_sw_3d = 0.5 * (y_pred['flux_scalar_sw'] + net_flux_sw)
    flux_up_sw_3d = 0.5 * (y_pred['flux_scalar_sw'] - net_flux_sw)

    # Compute the new heating rate
    hr_sw_3d = -scaling * net_flux_div_sw / np.diff(pressure_hl)

    y_pred['flux_dn_sw_rescaled'] = flux_dn_sw_3d
    y_pred['flux_up_sw_rescaled'] = flux_up_sw_3d
    y_pred['heating_rate_sw_rescaled'] = hr_sw_3d

    return reverse_levels(y_pred)
