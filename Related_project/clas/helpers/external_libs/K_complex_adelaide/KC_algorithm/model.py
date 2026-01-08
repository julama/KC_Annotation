#Copyright (C) 2020 Bastien Lechat

#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
#published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
#of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.



import numpy as np

import joblib
import os
import sys
from KC_algorithm.utils import EpochData
from KC_algorithm.utils import findN2peaks
from KC_algorithm.Wavelet import wave
import warnings
warnings.filterwarnings('ignore')

import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import sklearn
import torch


def load_joblib_with_sklearn_compat(filename):
    """
    Load a joblib file with compatibility for old sklearn module paths.
    Handles the case where old sklearn used 'sklearn.preprocessing.data'
    which no longer exists in newer versions.
    """
    # Add compatibility mapping to sys.modules before loading
    import sklearn.preprocessing
    # Create a fake module that redirects to the current sklearn.preprocessing
    class FakeModule:
        def __getattr__(self, name):
            # Import from sklearn.preprocessing
            return getattr(sklearn.preprocessing, name)
    
    # Temporarily add the old module path to sys.modules
    old_module = sys.modules.get('sklearn.preprocessing.data', None)
    sys.modules['sklearn.preprocessing.data'] = FakeModule()
    
    try:
        result = joblib.load(filename)
    finally:
        # Restore original module or remove fake one
        if old_module is not None:
            sys.modules['sklearn.preprocessing.data'] = old_module
        elif 'sklearn.preprocessing.data' in sys.modules:
            del sys.modules['sklearn.preprocessing.data']
    
    return result


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim,drop_out =0.5):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(input_dim, 1000, bias=False))
        self.add_module('bn1', torch.nn.BatchNorm1d(1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('dropout1', torch.nn.Dropout(p=drop_out, inplace=False))


        self.add_module('linear2', torch.nn.Linear(1000, 1000,bias=False))
        self.add_module('bn2', torch.nn.BatchNorm1d(1000))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('dropout2', torch.nn.Dropout(p=drop_out, inplace=False))


        self.add_module('linear3', torch.nn.Linear(1000, 500,bias=False))
        self.add_module('bn3', torch.nn.BatchNorm1d(500))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('dropout3', torch.nn.Dropout(p=drop_out, inplace=False))


        self.add_module('linear4', torch.nn.Linear(500, 256,bias=False))
        self.add_module('bn4', torch.nn.BatchNorm1d(256))
        self.add_module('relu4', torch.nn.ReLU())
        self.add_module('dropout4', torch.nn.Dropout(p=drop_out, inplace=False))


        self.add_module('linear6', torch.nn.Linear(256, output_dim,bias=False))

class GaussianProcessLayer(gpytorch.models.AbstractVariationalGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        # Handle gpytorch version compatibility: whiten parameter was removed in newer versions
        try:
            # Try with whiten parameter (older gpytorch versions)
            variational_strategy = VariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=True, whiten=True
            )
        except TypeError:
            # Newer gpytorch versions without whiten parameter
            try:
                variational_strategy = VariationalStrategy(
                    self, inducing_points, variational_distribution,
                    learn_inducing_locations=True
                )
            except TypeError:
                # Last resort: try with minimal parameters
                variational_strategy = VariationalStrategy(
                    self, inducing_points, variational_distribution
                )
        super(GaussianProcessLayer, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DKLModel(gpytorch.Module):
    def __init__(self, inducing_points, feature_extractor, num_features):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(inducing_points)
        self.num_features = num_features

    def forward(self, x):
        #print(x.type())
        projected_x = self.feature_extractor(x.float())

        res = self.gp_layer(projected_x)
        return res



def predict(model, likelihood, X):
    model.eval()
    likelihood.eval()

    correct = 0
    with torch.no_grad():
        output = likelihood(model(X))  #

        pred_labels = output.mean.ge(0.5).float().cpu().numpy()

        probas = output.mean.cpu().numpy()
    return probas, pred_labels


def remap_state_dict_keys(state_dict):
    """
    Remap state dict keys to handle gpytorch version compatibility.
    In newer gpytorch versions, variational_distribution became _variational_distribution.
    """
    remapped_dict = {}
    for key, value in state_dict.items():
        # Replace 'variational_distribution' with '_variational_distribution' for compatibility
        new_key = key.replace('variational_distribution.', '_variational_distribution.')
        remapped_dict[new_key] = value
    return remapped_dict


def get_model():
    dirname = os.path.dirname(__file__)

    inducing_filename = os.path.join(dirname, 'WeightsKCalgo/inducing_points_A2.npy')
    model_file = os.path.join(dirname, 'WeightsKCalgo/finaldkl_final_model_epoch50.dat')
    data_dim = 128
    num_features = 16
    drop_out_rate = 0.8

    feature_extractor = LargeFeatureExtractor(input_dim=data_dim,
                                              output_dim=num_features,
                                              drop_out=drop_out_rate)

    X_induced = torch.from_numpy(np.load(inducing_filename))


    model = DKLModel(inducing_points=X_induced, feature_extractor=feature_extractor,
                     num_features=num_features)

    # Bernouilli likelihood because only 2 classes
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    # Load state dict with compatibility handling
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    model_state_dict = checkpoint['model']
    likelihood_state_dict = checkpoint['likelihood']
    
    # Try loading with remapped keys first (for newer gpytorch)
    try:
        remapped_model_dict = remap_state_dict_keys(model_state_dict)
        model.load_state_dict(remapped_model_dict, strict=True)
    except RuntimeError:
        # If that fails, try with original keys (might work for older gpytorch or if keys match)
        try:
            model.load_state_dict(model_state_dict, strict=True)
        except RuntimeError:
            # Last resort: load with strict=False (may miss some parameters)
            print("Warning: Loading model state dict with strict=False due to key mismatches")
            model.load_state_dict(remapped_model_dict, strict=False)
    
    # Load likelihood state dict
    try:
        likelihood.load_state_dict(likelihood_state_dict, strict=True)
    except RuntimeError:
        print("Warning: Loading likelihood state dict with strict=False due to key mismatches")
        likelihood.load_state_dict(likelihood_state_dict, strict=False)

    return model, likelihood


def score_KCs(C3, Fs, Stages,sleep_stages = [2,3]):
    dirname = os.path.dirname(__file__)
    scaler_filename = os.path.join(dirname,'WeightsKCalgo/scaler_final_A2.save')
    # Use compatibility loader for old sklearn versions
    try:
        scaler = joblib.load(scaler_filename)
    except (ModuleNotFoundError, AttributeError) as e:
        if 'sklearn.preprocessing.data' in str(e):
            # Try with compatibility loader
            scaler = load_joblib_with_sklearn_compat(scaler_filename)
        else:
            raise


    model, likelihood = get_model()


    amp_thres = 20 * 10 ** -6  # 20 micro volt
    dist = 2  # [s]
    post_peak = 3  # [s]
    pre_peak = 3  # [s]
    length_of_stages = Stages['dur'].values[0]
    
    # Filter stages for the requested sleep stages
    filtered_stages = Stages.loc[Stages['label'].isin(sleep_stages), :]
    
    if len(filtered_stages) == 0:
        raise ValueError(
            f"No sleep stages found matching {sleep_stages}. "
            f"Available stages in hypnogram: {Stages['label'].unique().tolist()}. "
            f"Please check that the hypnogram contains stages {sleep_stages}."
        )
    
    if len(filtered_stages) < 4:
        print(f"Warning: Only {len(filtered_stages)} stages found (minimum 4 needed for findN2peaks). "
              f"This may cause issues. Consider using a longer recording or different sleep stages.")
    
    peaks,stage_peaks = findN2peaks(Stages=filtered_stages,
                        data=C3, Fs=Fs, min=amp_thres, distance=dist,
                        criteria=length_of_stages)

    assert len(peaks)==len(stage_peaks)
    
    # Filter peaks to only include those with enough data around them (±pre_peak and ±post_peak seconds)
    pre_samples = int(pre_peak * Fs)
    post_samples = int(post_peak * Fs)
    valid_mask = (peaks >= pre_samples) & (peaks < len(C3) - post_samples)
    
    if np.sum(valid_mask) == 0:
        raise ValueError(
            f"No valid peaks found after filtering for boundary conditions. "
            f"Need at least {pre_samples} samples before and {post_samples} samples after each peak. "
            f"Signal length: {len(C3)} samples. "
            f"Peaks found: {len(peaks)}, but all are too close to signal boundaries."
        )
    
    peaks_filtered = peaks[valid_mask]
    stage_peaks_filtered = stage_peaks[valid_mask]
    
    if len(peaks_filtered) < len(peaks):
        print(f"Warning: Filtered out {len(peaks) - len(peaks_filtered)} peaks that were too close to signal boundaries")
        print(f"  Using {len(peaks_filtered)} valid peaks (need ±{pre_peak}s around each peak)")
    
    d = EpochData(peaks_filtered, C3, post_peak, pre_peak, Fs)
    
    # Check if we got any valid epochs
    if d.shape[0] == 0:
        raise ValueError(
            f"EpochData returned empty array. This can happen if peaks are too close to signal boundaries. "
            f"Signal length: {len(C3)} samples ({len(C3)/Fs:.2f}s). "
            f"Need ±{pre_peak}s before and ±{post_peak}s after each peak."
        )
    
    # Update peaks and stage_peaks to match the filtered set
    peaks = peaks_filtered
    stage_peaks = stage_peaks_filtered

    

    s = [0, 1, 2, 3, 4]
    wavelet = 'sym3'
    wa = wave(Fs, wavelet, d)

   # wa.showdwtfreqs()

    coefsnoden = wa.dwtdec().get_coef(wanted_scale=s)

    X = coefsnoden

    data_scaled = scaler.transform(X)

    probas, _ = predict(model, likelihood, torch.from_numpy(data_scaled))

    return peaks,stage_peaks,d,probas
