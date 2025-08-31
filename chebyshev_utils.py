import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from tqdm import tqdm_notebook as tqdm
import math
from pinnutils import *
import sys
sys.path.insert(0, '/mnt/home/smaddu/anaconda3/lib/python3.9/site-packages')
import ot as pot
sys.path.insert(0, '/mnt/home/smaddu/.local/lib/python3.9/site-packages/')
import chaospy
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.optimal_transport import OTPlanSampler
import time as timeit
import ot
from scipy.interpolate import CubicSpline
from torchcubicspline import torchcubicspline

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def chebyshev_basis_matrix(s, degree):
    s = s.to(dtype=torch.float32)
    device = s.device
    V = [torch.ones_like(s, dtype=torch.float32, device=device), s]
    for n in range(2, degree + 1):
        Tn = 2 * s * V[-1] - V[-2]
        V.append(Tn)
    return torch.stack(V, dim=-1)  # (..., degree+1)

def chebyshev_U_basis_matrix(s, degree):
    s = s.to(dtype=torch.float32)
    device = s.device
    U = [torch.ones_like(s, dtype=torch.float32, device=device)]
    if degree >= 1:
        U.append(2 * s)
    for k in range(2, degree):
        U_next = 2 * s * U[-1] - U[-2]
        U.append(U_next)
    return torch.stack(U, dim=-1)  # (..., degree)

def batched_chebyshev_interpolate(t_points, x_points, degree=None, lambda_reg=0.0, penalty='none'):
    """
    Batched Chebyshev interpolation with explicit control over polynomial degree.

    Parameters:
    - t_points: [B, n]
    - x_points: [B, n, d]
    - degree: degree of Chebyshev polynomial (must be < n)
    - lambda_reg: regularization strength
    - penalty: 'none', 'l2', 'velocity', or 'curvature'

    Returns:
    - coeffs: Chebyshev coefficients [B, degree+1, d]
    - interpolant: callable function P(t_eval)
    - derivative: callable function P'(t_eval)
    - (a, b): rescaling constants per batch
    """
    t_points = t_points.to(dtype=torch.float32)
    x_points = x_points.to(dtype=torch.float32)
    device = t_points.device
    lambda_reg = float(lambda_reg)

    B, n, d = x_points.shape

    if degree is None:
        degree = n - 1
    elif degree >= n:
        raise ValueError(f"Requested degree={degree} must be < number of time points n={n}")

    a = t_points.min(dim=1, keepdim=True).values  # [B, 1]
    b = t_points.max(dim=1, keepdim=True).values  # [B, 1]
    s_points = (2 * t_points - (a + b)) / (b - a)  # [B, n]

    V = chebyshev_basis_matrix(s_points, degree)  # [B, n, degree+1]
    VT = V.transpose(1, 2)                         # [B, degree+1, n]

    powers = torch.arange(degree + 1, dtype=torch.float32, device=device)
    if penalty == 'none' or lambda_reg == 0.0:
        R = torch.zeros_like(powers)
    elif penalty == 'l2':
        R = torch.ones_like(powers); R[0] = 0
    elif penalty == 'velocity':
        R = powers**2; R[0] = 0
    elif penalty == 'curvature':
        R = powers**4; R[0] = 0
    else:
        raise ValueError("Invalid penalty type")

    R_mat = torch.diag(R).unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, degree+1, degree+1]
    lhs = VT @ V + lambda_reg * R_mat                     # [B, degree+1, degree+1]
    rhs = VT @ x_points                                   # [B, degree+1, d]
    coeffs = torch.linalg.solve(lhs, rhs).float()         # [B, degree+1, d]

    def interpolant(t_eval):  # t_eval: [B, n_eval]
        t_eval = t_eval.to(dtype=torch.float32, device=device)
        s_eval = (2 * t_eval - (a + b)) / (b - a)  # [B, n_eval]
        basis = chebyshev_basis_matrix(s_eval, degree)     # [B, n_eval, degree+1]
        return torch.einsum('bnd,bdc->bnc', basis, coeffs)  # [B, n_eval, d]

    def derivative(t_eval):  # t_eval: [B, n_eval]
        t_eval = t_eval.to(dtype=torch.float32, device=device)
        s_eval = (2 * t_eval - (a + b)) / (b - a)              # [B, n_eval]
        dsdt = (2 / (b - a)).float().unsqueeze(-1)             # [B, 1]
        U_basis = chebyshev_U_basis_matrix(s_eval, degree)     # [B, n_eval, degree]

        deriv = coeffs[:, 1:2, :].expand(-1, t_eval.shape[1], -1).clone()  # [B, n_eval, d]
        for k in range(2, degree + 1):
            uk = U_basis[:, :, k - 1:k]
            coeff_k = coeffs[:, k:k+1, :]
            deriv += k * uk * coeff_k

        return dsdt * deriv                     # [B, n_eval, d]

    return coeffs, interpolant, derivative, (a, b)

def compute_pairwise_ot_plans(samples, method="exact", epsilon=1e-2):
    """
    Compute pairwise optimal transport plans π_k^* between K+1 time snapshots.

    Parameters
    ----------
    samples : np.ndarray of shape (K+1, B, d)
        Time-series snapshots. Each slice [k] is a B x d array of samples.
    method : str
        OT method: 'sinkhorn' or 'exact'
    epsilon : float
        Entropic regularization strength (only used for Sinkhorn)

    Returns
    -------
    pi_list : list of np.ndarray of shape (B, B)
        Pairwise optimal transport plans π_k^* for k = 0 to K-1
    """
    K_plus_1, B, d = samples.shape
    pi_list = []

    uniform_weights = torch.ones(B) / B

    for k in range(K_plus_1 - 1):
        x = samples[k]      # shape: (B, d)
        y = samples[k + 1]  # shape: (B, d)

        C = ot.dist(x, y, metric='sqeuclidean')  # cost matrix: (B, B)

        if method == "sinkhorn":
            pi = ot.sinkhorn(uniform_weights, uniform_weights, C, reg=epsilon)
        elif method == "exact":
            pi = ot.emd(uniform_weights, uniform_weights, C)
        else:
            raise ValueError(f"Unknown OT method: {method}")

        pi_list.append(pi)

    return pi_list


def sample_trajectory(pi_list, num_samples=1):
    """
    Sample index trajectories (i_0, ..., i_K) from the MMFM joint.

    Parameters
    ----------
    pi_list : list of np.ndarray of shape (B, B)
        List of pairwise OT plans π_k^* for k = 0 to K-1
    num_samples : int
        Number of full trajectories to sample

    Returns
    -------
    trajectories : np.ndarray of shape (num_samples, K+1)
        Each row is a sampled sequence of indices (i_0, ..., i_K)
    """
    K = len(pi_list) + 1
    B = pi_list[0].shape[0]

    trajectories = np.zeros((num_samples, K), dtype=int)

    for n in range(num_samples):
        i_k = np.random.choice(B)
        trajectories[n, 0] = i_k

        for k in range(K - 1):
            probs = pi_list[k][i_k]
            probs = probs / probs.sum()
            i_kp1 = np.random.choice(B, p=probs)
            trajectories[n, k + 1] = i_kp1
            i_k = i_kp1

    return trajectories

def get_sample_paths(samples, trajectories, num_paths):
    """
    Convert index trajectories to actual data-space trajectories.

    Parameters
    ----------
    samples : np.ndarray of shape (K+1, B, d)
    trajectories : np.ndarray of shape (N, K+1)

    Returns
    -------
    sample_paths : np.ndarray of shape (N, K+1, d)
    """
    N, K_plus_1 = trajectories.shape
    d = samples.shape[-1]

    sample_paths = torch.zeros((K_plus_1, num_paths ,  d),dtype=torch.float32)

    for t in range(K_plus_1):
        sample_paths[t,: , :] = samples[t][trajectories[:, t]]

    return sample_paths

def mmot_trajectories(samples,num_paths,device='cuda'):
    pi_list = compute_pairwise_ot_plans(samples, method="exact");
    trajectories = sample_trajectory(pi_list, num_samples=num_paths);
    paths = get_sample_paths(samples, trajectories, num_paths).to(device);
    return paths #torch.tensor(paths,dtype=torch.float32,device=device)

def stitch_ot_trajectories(samples, method="exact"):
    
    ot_sampler = OTPlanSampler(method="exact")
    """
    Computes optimal transport stitching of samples across time using the specified OT method.

    Parameters
    ----------
    samples : torch.Tensor of shape (K, B, D)
        Input tensor containing K time-snapshots, each with B samples in D dimensions.
    method : str
        OT method to use: "exact", "sinkhorn", etc.

    Returns
    -------
    ot_samples : torch.Tensor of shape (K, B, D)
        Reordered tensor where sample trajectories are stitched across time via OT.
    """

    K, B, D = samples.shape
    ot_sampler = OTPlanSampler(method=method)

    # Prepare inputs
    tsamples = samples[:, :, :D].clone()
    labels = torch.arange(B).unsqueeze(1).float()

    # Stitch trajectories
    start_time = time.time()
    ind_list = []

    for k in range(K - 1):
        xk = tsamples[k]
        xkp1 = tsamples[k + 1]

        pi = ot_sampler.get_map(xk, xkp1)
        i, j = np.nonzero(pi)

        if k == 0:
            label_old = labels[i].int()
            label_new = labels[j].int()
            ind_list.append(label_old.squeeze())
            ind_list.append(label_new.squeeze())
        else:
            label_old = labels[i].int()
            sort_ind = (label_old[:, 0].unsqueeze(0) == label_new[:, 0].unsqueeze(1)).nonzero(as_tuple=True)[1]
            label_new = labels[j].int()[sort_ind, :] + 0
            ind_list.append(label_new.squeeze())

    #print("Stitching time: {:.3f} sec".format(time.time() - start_time))

    # Reorder samples according to stitched OT labels
    ot_samples = torch.zeros_like(tsamples)
    for k in range(K):
        ot_samples[k] = tsamples[k][ind_list[k], :]

    return ot_samples


def stitch_ot_trajectories_with_atac(samples, samples_atac, method="exact"):
    
    ot_sampler = OTPlanSampler(method="exact")
    """
    Computes optimal transport stitching of samples across time using the specified OT method.

    Parameters
    ----------
    samples : torch.Tensor of shape (K, B, D)
        Input tensor containing K time-snapshots, each with B samples in D dimensions.
    method : str
        OT method to use: "exact", "sinkhorn", etc.

    Returns
    -------
    ot_samples : torch.Tensor of shape (K, B, D)
        Reordered tensor where sample trajectories are stitched across time via OT.
    """

    K, B, D = samples.shape
    ot_sampler = OTPlanSampler(method=method)

    # Prepare inputs
    tsamples = samples[:, :, :D].clone()
    labels = torch.arange(B).unsqueeze(1).float()

    # Stitch trajectories
    start_time = time.time()
    ind_list = []

    for k in range(K - 1):
        xk = tsamples[k]
        xkp1 = tsamples[k + 1]

        pi = ot_sampler.get_map(xk, xkp1)
        i, j = np.nonzero(pi)

        if k == 0:
            label_old = labels[i].int()
            label_new = labels[j].int()
            ind_list.append(label_old.squeeze())
            ind_list.append(label_new.squeeze())
        else:
            label_old = labels[i].int()
            sort_ind = (label_old[:, 0].unsqueeze(0) == label_new[:, 0].unsqueeze(1)).nonzero(as_tuple=True)[1]
            label_new = labels[j].int()[sort_ind, :] + 0
            ind_list.append(label_new.squeeze())

    #print("Stitching time: {:.3f} sec".format(time.time() - start_time))

    # Reorder samples according to stitched OT labels
    ot_samples = torch.zeros_like(tsamples)
    ot_samples_atac = torch.zeros_like(tsamples)
    for k in range(K):
        ot_samples[k] = tsamples[k][ind_list[k], :]
        ot_samples_atac[k] = samples_atac[k][ind_list[k], :]

    return ot_samples, ot_samples_atac

def compute_conditional_distributions_chebyshev(Dist, batch_size, nodes_fit, nodes_eval, reg_, sigma, err_flag=False, atacflag='False',device='cpu'):
    """
    Compute conditional distributions using Chebyshev polynomial interpolation.

    Parameters:
    - Dist: Tensor of shape [T, nsamples, Ndim], time × samples × dimensions
    - batch_size: Number of trajectories to sample
    - nodes_fit: Tensor of shape [B, T1], time points for fitting (can vary per batch)
    - nodes_eval: Tensor of shape [B, T2], time points for evaluation (can vary per batch)
    - reg_: Regularization strength for Chebyshev interpolation
    - sigma: Standard deviation of Gaussian noise to add to inputs
    - ot: (Optional) flag for using optimal transport logic (unused here)

    Returns:
    - Xtrain: [batch_size * T2, Ndim + 2], inputs for training with noise and time appended
    - ytrain: [batch_size * T2, Ndim], interpolated derivatives
    - err: Relative interpolation error
    """

    nsamples = Dist.shape[1]
    Np = nodes_eval.shape[1]
    Ndim = Dist.shape[2]
    err = 0;

    # Initialize output tensors
    Xtrain = torch.zeros((batch_size * Np, Ndim + 2), dtype=torch.float32, device=device)
    ytrain = torch.zeros((batch_size * Np, Ndim), dtype=torch.float32, device=device)

    # Select batch of trajectories at random
    xind = torch.randint(0, nsamples, (batch_size,))
    Dist = torch.tensor(Dist, dtype=torch.float32, device=device)

    # Permute and extract [B, T, Ndim]
    x = torch.permute(Dist[:, xind, 0:Ndim], (1, 0, 2))

    # Fit Chebyshev polynomial interpolants
    coeffs, P, P_prime, (a, b) = batched_chebyshev_interpolate(
        nodes_fit, x, degree = nodes_fit.shape[1] - 1, lambda_reg=reg_, penalty='curvature'
    ) # always fitting polynomial of degree n-1.

    # Evaluate polynomial and derivative
    x_interp = P(nodes_eval)        # [B, T2, Ndim]
    dx_interp = P_prime(nodes_eval) # [B, T2, Ndim]

    # Compute interpolation error (relative MSE)
    if(err_flag):
        err = torch.mean((x_interp - x) ** 2) / torch.mean(x ** 2)
        print(" lam ", reg_, " err ", err)

    # Construct training set
    Xtrain[:, Ndim] = 0.01
    Xtrain[:, Ndim + 1] = nodes_eval.view(batch_size * Np)

    mut = x_interp.view(batch_size * Np, Ndim)
    Xtrain[:, 0:Ndim] = mut + sigma * torch.randn_like(mut)
    ytrain[:, 0:Ndim] = dx_interp.view(batch_size * Np, Ndim)

    return Xtrain, ytrain, err

def compute_conditional_distributions_chebyshev_withgrowth(Dist, batch_size, mass_, nodes_fit, nodes_eval, reg_, sigma, err_flag=False, atacflag='False',device='cpu'):
    """
    Compute conditional distributions using Chebyshev polynomial interpolation.

    Parameters:
    - Dist: Tensor of shape [T, nsamples, Ndim], time × samples × dimensions
    - batch_size: Number of trajectories to sample
    - nodes_fit: Tensor of shape [B, T1], time points for fitting (can vary per batch)
    - nodes_eval: Tensor of shape [B, T2], time points for evaluation (can vary per batch)
    - reg_: Regularization strength for Chebyshev interpolation
    - sigma: Standard deviation of Gaussian noise to add to inputs
    - ot: (Optional) flag for using optimal transport logic (unused here)

    Returns:
    - Xtrain: [batch_size * T2, Ndim + 2], inputs for training with noise and time appended
    - ytrain: [batch_size * T2, Ndim], interpolated derivatives
    - err: Relative interpolation error
    """

    nsamples = Dist.shape[1]
    Np = nodes_eval.shape[1]
    Ndim = Dist.shape[2]
    err = 0;

    # Initialize output tensors
    Xtrain = torch.zeros((batch_size * Np, Ndim + 2), dtype=torch.float32, device=device)
    ytrain = torch.zeros((batch_size * Np, Ndim), dtype=torch.float32, device=device)

    # Select batch of trajectories at random
    xind = torch.randint(0, nsamples, (batch_size,))
    Dist = torch.tensor(Dist, dtype=torch.float32, device=device)

    mass_weight = torch.permute(mass_[:,xind],(1,0))

    # Permute and extract [B, T, Ndim]
    x = torch.permute(Dist[:, xind, 0:Ndim], (1, 0, 2))

    # Fit Chebyshev polynomial interpolants
    coeffs, P, P_prime, (a, b) = batched_chebyshev_interpolate(
        nodes_fit, x, degree = nodes_fit.shape[1] - 1, lambda_reg=reg_, penalty='curvature'
    ) # always fitting polynomial of degree n-1.

    # Evaluate polynomial and derivative
    x_interp = P(nodes_eval)        # [B, T2, Ndim]
    dx_interp = P_prime(nodes_eval) # [B, T2, Ndim]

    # Compute interpolation error (relative MSE)
    if(err_flag):
        err = torch.mean((x_interp - x) ** 2) / torch.mean(x ** 2)
        print(" lam ", reg_, " err ", err)

    # Construct training set
    Xtrain[:, Ndim] = 0.01
    Xtrain[:, Ndim + 1] = nodes_eval.view(batch_size * Np)

    mut = x_interp.view(batch_size * Np, Ndim)
    Xtrain[:, 0:Ndim] = mut + sigma * torch.randn_like(mut)
    ytrain[:, 0:Ndim] = dx_interp.view(batch_size * Np, Ndim)

    return Xtrain, ytrain, mass_weight

def compute_conditional_distributions_chebyshev_atac(Dist, Dist_atac, batch_size, nodes_fit, nodes_eval, reg_, sigma, err_flag=False, atacflag='False',device='cpu'):
    """
    Compute conditional distributions using Chebyshev polynomial interpolation.

    Parameters:
    - Dist: Tensor of shape [T, nsamples, Ndim], time × samples × dimensions
    - batch_size: Number of trajectories to sample
    - nodes_fit: Tensor of shape [B, T1], time points for fitting (can vary per batch)
    - nodes_eval: Tensor of shape [B, T2], time points for evaluation (can vary per batch)
    - reg_: Regularization strength for Chebyshev interpolation
    - sigma: Standard deviation of Gaussian noise to add to inputs
    - ot: (Optional) flag for using optimal transport logic (unused here)

    Returns:
    - Xtrain: [batch_size * T2, Ndim + 2], inputs for training with noise and time appended
    - ytrain: [batch_size * T2, Ndim], interpolated derivatives
    - err: Relative interpolation error
    """

    nsamples = Dist.shape[1]
    Np = nodes_eval.shape[1]
    Ndim = Dist.shape[2]
    err = 0;

    # Initialize output tensors
    Xtrain = torch.zeros((batch_size * Np, Ndim + 2), dtype=torch.float32, device=device)
    ATACtrain = torch.zeros((batch_size * Np, Ndim ), dtype=torch.float32, device=device)
    ytrain = torch.zeros((batch_size * Np, Ndim), dtype=torch.float32, device=device)

    # Select batch of trajectories at random
    xind = torch.randint(0, nsamples, (batch_size,))
    Dist = torch.tensor(Dist, dtype=torch.float32, device=device)

    # Permute and extract [B, T, Ndim]
    x = torch.permute(Dist[:, xind, 0:Ndim], (1, 0, 2))
    xatac = torch.permute(Dist_atac[:, xind, 0:Ndim], (1,0,2))

    # Fit Chebyshev polynomial interpolants
    coeffs, P, P_prime, (a, b) = batched_chebyshev_interpolate(
        nodes_fit, x, degree = nodes_fit.shape[1] - 1, lambda_reg=reg_, penalty='curvature'
    ) # always fitting polynomial of degree n-1.

    # Evaluate polynomial and derivative
    x_interp = P(nodes_eval)        # [B, T2, Ndim]
    dx_interp = P_prime(nodes_eval) # [B, T2, Ndim]

    coeffs_atac, Patac, _, (a, b) = batched_chebyshev_interpolate(
        nodes_fit, xatac, degree = nodes_fit.shape[1] - 1, lambda_reg=reg_, penalty='curvature'
    ) # always fitting polynomial of degree n-1.

    xatac_interp = Patac(nodes_eval) # xatac + 0

    # Compute interpolation error (relative MSE)
    if(err_flag):
        err = torch.mean((x_interp - x) ** 2) / torch.mean(x ** 2)
        print(" lam ", reg_, " err ", err)

    # Construct training set
    Xtrain[:, Ndim] = 0.01
    Xtrain[:, Ndim + 1] = nodes_eval.view(batch_size * Np)

    mut = x_interp.view(batch_size * Np, Ndim)
    Xtrain[:, 0:Ndim] = mut + sigma * torch.randn_like(mut)
    ATACtrain[:,0:Ndim] = xatac_interp.reshape(batch_size*Np, Ndim)
    ytrain[:, 0:Ndim] = dx_interp.view(batch_size * Np, Ndim)

    return Xtrain, ytrain, ATACtrain, err

def compute_conditional_distributions_torch_cubic(Dist, batch_size, nodes_fit, nodes_eval, sigma, err_flag=False, device='cpu'):
    """
    Compute conditional distributions using cubic spline interpolation in PyTorch.

    Parameters:
    - Dist: Tensor of shape [T, B, Ndim] (time × samples × dimensions)
    - batch_size: Number of trajectories to sample
    - nodes_fit: Tensor of shape [T], time points for fitting the spline
    - nodes_eval: Tensor of shape [T], time points for evaluating the spline
    - sigma: Standard deviation of Gaussian noise added to inputs

    Returns:
    - Xtrain: Tensor [batch_size * T, Ndim + 2], input points + time and noise dims
    - ytrain: Tensor [batch_size * T, Ndim], spline derivatives
    - err: Relative interpolation error
    """
    
    nsamples = Dist.shape[1]
    Np = nodes_eval.shape[0]
    Ndim = Dist.shape[2]
    err = 0;

    Xtrain = torch.zeros((batch_size * Np, Ndim + 2), dtype=torch.float32, device=device)
    ytrain = torch.zeros((batch_size * Np, Ndim), dtype=torch.float32, device=device)

    # Randomly select batch of trajectories
    xind = torch.randint(0, nsamples, (batch_size,))
    Dist = torch.permute(Dist[:, xind, :], (1, 0, 2))  # [B, T, Ndim]

    # Prepare cubic spline interpolant
    tempdist = Dist[None, :, :, 0:Ndim]  # [1, B, T, Ndim]
    coeffs = torchcubicspline.interpolate.natural_cubic_spline_coeffs(nodes_fit, tempdist)
    spline = torchcubicspline.interpolate.NaturalCubicSpline(coeffs)

    # Evaluate spline and its derivative
    eval_ = spline.evaluate(nodes_eval)        # [1, B, T, Ndim]
    derv_ = spline.derivative(nodes_eval)      # [1, B, T, Ndim]

    eval_ = torch.permute(eval_, [2, 1, 0, 3]).squeeze()  # [T, B, Ndim]
    derv_ = torch.permute(derv_, [2, 1, 0, 3]).squeeze()  # [T, B, Ndim]

    eval_ = torch.permute(eval_, (1, 0, 2))  # [B, T, Ndim]
    derv_ = torch.permute(derv_, (1, 0, 2))  # [B, T, Ndim]

    # Compute interpolation error
    if(err_flag):
        err = torch.mean((eval_ - Dist) ** 2) / torch.mean(Dist ** 2)

    # Create training set with noise-perturbed inputs and spline velocity targets
    Xtrain[:, Ndim] = 0.01
    Xtrain[:, Ndim + 1] = nodes_eval.expand(batch_size, -1).reshape(batch_size * Np)
    
    mut = eval_.reshape(batch_size * Np, Ndim)
    Xtrain[:, 0:Ndim] = mut + sigma * torch.randn_like(mut)
    ytrain[:, 0:Ndim] = derv_.reshape(batch_size * Np, Ndim)

    return Xtrain, ytrain, err

def P1_vectorized(t_eval, t_fit, y_fit):
    """
    Vectorized linear interpolation and derivative calculation.

    Parameters:
    - t_eval: [B, T2], evaluation times (sorted)
    - t_fit: [B, T1], fit times (sorted)
    - y_fit: [B, T1, D], fit values

    Returns:
    - interp: [B, T2, D], interpolated values
    - deriv: [B, T2, D], derivatives
    """
    B, T1, D = y_fit.shape
    T2 = t_eval.shape[1]

    # Find interval indices [B, T2]
    idx = torch.searchsorted(t_fit, t_eval, right=True) - 1
    idx = idx.clamp(0, T1 - 2)  # Ensure valid range

    # Gather t0, t1, y0, y1 for all batches and eval points
    t0 = torch.gather(t_fit, 1, idx)               # [B, T2]
    t1 = torch.gather(t_fit, 1, idx + 1)           # [B, T2]

    y0 = torch.gather(y_fit, 1, idx.unsqueeze(-1).expand(-1, -1, D))     # [B, T2, D]
    y1 = torch.gather(y_fit, 1, (idx + 1).unsqueeze(-1).expand(-1, -1, D))  # [B, T2, D]

    # Compute alpha and delta_t
    delta_t = (t1 - t0).unsqueeze(-1)  # [B, T2, 1]
    alpha = ((t_eval - t0) / (t1 - t0)).unsqueeze(-1)  # [B, T2, 1]
    alpha = torch.where(delta_t != 0, alpha, torch.zeros_like(alpha))

    # Interpolation and derivative
    interp = (1 - alpha) * y0 + alpha * y1
    deriv = torch.where(delta_t != 0, (y1 - y0) / delta_t, torch.zeros_like(y0))

    return interp, deriv

def compute_conditional_distributions_linear(Dist, batch_size, nodes_fit, nodes_eval, reg_, sigma, err_flag=False,device='cuda'):
    """
    Compute conditional distributions using Chebyshev polynomial interpolation.

    Parameters:
    - Dist: Tensor of shape [T, nsamples, Ndim], time × samples × dimensions
    - batch_size: Number of trajectories to sample
    - nodes_fit: Tensor of shape [B, T1], time points for fitting (can vary per batch)
    - nodes_eval: Tensor of shape [B, T2], time points for evaluation (can vary per batch)
    - reg_: Regularization strength for Chebyshev interpolation
    - sigma: Standard deviation of Gaussian noise to add to inputs
    - ot: (Optional) flag for using optimal transport logic (unused here)

    Returns:
    - Xtrain: [batch_size * T2, Ndim + 2], inputs for training with noise and time appended
    - ytrain: [batch_size * T2, Ndim], interpolated derivatives
    - err: Relative interpolation error
    """

    nsamples = Dist.shape[1]
    Np = nodes_eval.shape[1]
    Ndim = Dist.shape[2]
    err = 0;

    # Initialize output tensors
    Xtrain = torch.zeros((batch_size * Np, Ndim + 2), dtype=torch.float32, device=device)
    ytrain = torch.zeros((batch_size * Np, Ndim), dtype=torch.float32, device=device)

    # Select batch of trajectories at random
    xind = torch.randint(0, nsamples, (batch_size,))
    Dist = torch.tensor(Dist, dtype=torch.float32, device=device)

    # Permute and extract [B, T, Ndim]
    x = torch.permute(Dist[:, xind, 0:Ndim], (1, 0, 2))
    x_interp, dx_interp = P1_vectorized(nodes_eval, nodes_fit, x)

    # Compute interpolation error (relative MSE)
    if(err_flag):
        err = torch.mean((x_interp - x) ** 2) / torch.mean(x ** 2)

    # Construct training set
    Xtrain[:, Ndim] = 0.01
    Xtrain[:, Ndim + 1] = nodes_eval.view(batch_size * Np)

    mut = x_interp.view(batch_size * Np, Ndim)
    Xtrain[:, 0:Ndim] = mut + sigma * torch.randn_like(mut)
    ytrain[:, 0:Ndim] = dx_interp.view(batch_size * Np, Ndim)

    return Xtrain, ytrain, err

def select_best_lambda(batch_ot_samples, batch_size, data_batch, eval_batch, Ndim, device, sigma=0.001, 
                       lam_vals=None, rel_tol=0.8, verbose=True):
    """
    Sweep over lambda values to compute test error and velocity magnitude,
    and return the best lambda based on relative error drop.
    
    Parameters:
    - batch_ot_samples, batch_size, data_batch, uniform_batch: data inputs
    - drift_net, scoreNet: trained networks
    - D: diffusion coefficient
    - Ndim: number of features
    - device: torch device
    - sigma: regularization kernel width
    - lam_vals: optional list of lambdas
    - rel_tol: relative error drop tolerance for selection
    - verbose: if True, prints summary
    
    Returns:
    - best_lambda: lambda with ≥ `rel_tol` error drop
    - lam_arr, test_err, vel_mag: all values for plotting/debugging
    """
    if lam_vals is None:
        lam_arr = np.array([0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0])
    else:
        lam_arr = np.array(lam_vals)

    test_err = np.zeros_like(lam_arr)
    vel_mag = np.zeros_like(lam_arr)

    for k in range(len(lam_arr)):
        _, ytrain, _ = compute_conditional_distributions_chebyshev(
            batch_ot_samples, batch_size, data_batch, eval_batch,
            reg_=lam_arr[k], sigma=sigma, err_flag=False, device=device)
        vel_mag[k] = torch.mean(ytrain**2).cpu().item()

    err0 = vel_mag[0]
    rel_err_drop = (err0 - vel_mag) / err0
    mask = rel_err_drop >= rel_tol

    if np.any(mask):
        idx = np.argmax(mask)
        best_lambda = lam_arr[idx]
    else:
        best_lambda = lam_arr[np.argmin(test_err)]  # fallback

    if verbose:
        print(f"[λ-selection] Initial error: {err0:.4f}")
        print(f"[λ-selection] Best λ (≥{rel_tol*100:.0f}% drop): {best_lambda:.4f}")
        print(f"[λ-selection] Test errors: {test_err}")
        print(f"[λ-selection] Vel magnitudes: {vel_mag}")

    return best_lambda, lam_arr, test_err, vel_mag

def generate_firstkind_nodes(a, b, n):
    k = torch.arange(n)
    x_k = torch.cos(np.pi * (2 * k + 1) / (2 * n))
    nodes = 0.5 * (b + a) + 0.5 * (b - a) * x_k
    return torch.sort(nodes)[0]  # scale from [-1,1] to [a,b]

def generate_secondkind_nodes(a,b,n):
    nodes, weights = chaospy.quadrature.clenshaw_curtis(n-1, (a,b))
    return torch.tensor(nodes.squeeze(),dtype=torch.float32), torch.tensor(weights,dtype=torch.float32)

def interpolate_old2new(Dist, old_nodes, new_nodes):
    """
    Interpolates the time series data in `Dist` from old_nodes to new_nodes using 1D linear interpolation.

    Parameters:
    - Dist: Tensor of shape [T, B, Ndim] — time × batch × dimension
    - old_nodes: Tensor of shape [B, T] — time coordinates associated with `Dist`
    - new_nodes: Tensor of shape [B, T_new] — new time points to interpolate to

    Returns:
    - x_interp: Tensor of shape [T_new, B, Ndim] — interpolated data
    """

    T, B, D = Dist.shape
    x = torch.permute(Dist, (1, 0, 2))  # [B, T, Ndim]

    # Perform per-batch per-dimension 1D linear interpolation
    x_interp = torch.stack([
        torch.stack([
            torch.tensor(
                np.interp(
                    new_nodes[b].cpu().numpy(),
                    old_nodes[b].cpu().numpy(),
                    x[b, :, d].cpu().numpy()
                ),
                dtype=torch.float32
            )
            for d in range(D)
        ], dim=1)  # [T_new, Ndim]
        for b in range(B)
    ], dim=0)  # [B, T_new, Ndim]

    return torch.permute(x_interp, (1, 0, 2))  # [T_new, B, Ndim]

def interpolate_old2new_chebyshev(Dist, old_nodes, new_nodes, reg_):
    """
    Interpolates the time series data in `Dist` from old_nodes to new_nodes using 1D linear interpolation.

    Parameters:
    - Dist: Tensor of shape [T, B, Ndim] — time × batch × dimension
    - old_nodes: Tensor of shape [B, T] — time coordinates associated with `Dist`
    - new_nodes: Tensor of shape [B, T_new] — new time points to interpolate to

    Returns:
    - x_interp: Tensor of shape [T_new, B, Ndim] — interpolated data
    """
    coeffs, P, P_prime, (a, b) = batched_chebyshev_interpolate(old_nodes, torch.permute(Dist,(1,0,2)), degree = old_nodes.shape[1]-1,lambda_reg=reg_, penalty='curvature')
    return torch.permute(P(new_nodes),(1,0,2))

def div_neural_hutchinson_scale(drift, inp_, Ndim, inf_lx, num_samples=1, v_distribution="rademacher"):
    """
    Computes the divergence of a neural network's output using the Hutchinson's trace estimator.

    Args:
        drift (torch.Tensor): The output of the neural network (shape: b x d),
            where b is the batch size and d is the dimension.
        inp_ (torch.Tensor): The input to the neural network (shape: b x d).
        Ndim (int): The dimension of the input and output (d).
        inf_lx (torch.Tensor): A tensor to add to the divergence.
        num_samples (int, optional): Number of random vectors to use for the
            Hutchinson estimate.  A small value (e.g., 1 to 10) is often sufficient.
            Defaults to 1.
        v_distribution (str, optional): The distribution to use for the random vector v.
            Can be "normal" (standard normal) or "rademacher" (+/- 1 with equal probability).
            Defaults to "normal".

    Returns:
        torch.Tensor: The estimated divergence (shape: b x d).
    """
    #print(" I am here ")
    batch_size = inp_.shape[0]
    div = drift.clone() + 0  # Initialize divergence with the drift term

    # Check dimensionality of input tensors
    if drift.ndim != 2 or inp_.ndim != 2:
        raise ValueError("drift and inp_ must be 2-dimensional tensors (b x d)")
    if drift.shape != inp_.shape:
        raise ValueError("drift and inp_ must have the same shape")
    if drift.shape[1] != Ndim:
        raise ValueError(f"drift and inp_ must have dimension Ndim={Ndim}")

    for _ in range(num_samples):
        # Generate a random vector v
        if v_distribution == "normal":
            v = torch.randn(batch_size, Ndim, device=inp_.device)  # shape: (b x d)
        elif v_distribution == "rademacher":
            v = torch.randint(0, 2, (batch_size, Ndim), device=inp_.device) * 2 - 1
        else:
            raise ValueError(f"v_distribution must be 'normal' or 'rademacher', got {v_distribution}")

        # Compute the Jacobian-vector product (Jv) using autograd
        drift_v = torch.sum(drift * v, dim=1)  # shape: (b)
        Jv = grad(outputs=drift_v, inputs=inp_,
                  grad_outputs=torch.ones_like(drift_v),
                  create_graph=True)[0]  # shape: (b x d)

        # Accumulate the trace estimate
        div += (v * Jv).sum(dim=1, keepdim=True)  # shape (b x 1)
    div = inf_lx + div / num_samples  # shape (b x d)
    return div

class DNN(nn.Module):
    def __init__(self, sizes, mean=0, std=1, seed=0, activation=nn.Tanh()):
        super(DNN, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.bn = BatchNorm(mean, std)
        layer = []
        for i in range(len(sizes)-2):
            linear = LayerNoWN(sizes[i], sizes[i+1], seed, activation)
            layer += [linear, activation]
        layer += [LayerNoWN(sizes[-2], sizes[-1], seed, activation)]
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.net(self.bn(x))

def geometric_sequence(L):
    r = np.exp((-2/L)*np.log(10))
    sigma = np.zeros((L,))
    for i in range(0,L):
        sigma[i] = 1*r**i
    return sigma

def generate_noisy_training_data(Dist, Ndim, tp, L, nsamples, nsnaps, device):
    sigma = geometric_sequence(L);
    transform_data = np.zeros((nsnaps,L*nsamples,Ndim+2))
    for tind in range(0,nsnaps):
        for i in range(0,L):
            for j in range(0,nsamples):
                mean = Dist[tind,j,0:Ndim]; cov = (sigma[i]**2)*np.eye(Ndim)
                transform_data[tind,j+i*nsamples,0:Ndim] = np.random.multivariate_normal(mean, cov)
                transform_data[tind,j+i*nsamples,Ndim] = sigma[i]
                transform_data[tind,j+i*nsamples,Ndim+1] = tp[tind]

        print(" done with time ", tind)
            
    X_train = torch.tensor(transform_data, dtype=torch.float32,requires_grad=True,device=device)
    print(" X_train shape ", X_train.shape)
    X_data = torch.tensor(Dist, dtype=torch.float32,device=device)
    print(" X_data shape ", X_data.shape)

    X_mean = torch.zeros((Ndim+2,)); X_std = torch.ones((Ndim+2,))
    for i in range(0,Ndim+2):
        X_mean[i] = torch.tensor(np.mean(transform_data[:,:,i].flatten(),axis=0, keepdims=True), dtype=torch.float32)
        X_std[i]  = torch.tensor(np.std(transform_data[:,:,i].flatten(),axis=0, keepdims=True), dtype=torch.float32)

    X_mean = X_mean[np.newaxis,:]; X_std = X_std[np.newaxis,:]; 
    print(" X-mean shape ", X_mean.shape, " X-std shape ", X_std.shape)
    return X_train, X_data, X_mean, X_std, sigma


def DSM_training(Ndim, net, X_train, X_data, sigma, nsnaps, nsamples, L, adp_flag=0, n_epochs=10001, device='cpu'):
    """
    Train the encoder with adaptive per-timepoint loss weighting.
    """
    optimizer = torch.optim.Adam(list(net.parameters()), lr=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[2500, 5000, 7500], gamma=0.1)

    c_ = torch.ones((nsnaps,), dtype=torch.float32, device=device)
    alpha_ann = 0.5
    adapt_int = 10

    sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=X_train.device)
    sigma_reshaped = sigma_tensor.view(L, 1, 1)

    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        uhat = net(X_train)
        lcomp = torch.zeros((nsnaps,), device=device)
        std_ = torch.zeros((nsnaps,), device=device)

        for tind in range(nsnaps):
            loss_sum = 0

            u_pred = uhat[tind].view(L, nsamples, Ndim)
            x_t    = X_train[tind, :, :Ndim].view(L, nsamples, Ndim)
            x_data = X_data[tind, :, :Ndim].unsqueeze(0).expand(L, nsamples, Ndim)
            
            u_true = (x_t - x_data) / (sigma_reshaped ** 2)
            residual = u_pred + u_true
            residual_squared = residual.pow(2).mean(dim=(1, 2))  # shape (L,)
            loss_sum = 0.5 * torch.sum((sigma_tensor ** 2) * residual_squared)

            if adp_flag == 1 and epoch % adapt_int == 0:
                with torch.no_grad():
                    std_[tind] = loss_grad_std(loss_sum, net, device)

            lcomp[tind] = loss_sum

        if adp_flag == 1 and epoch % adapt_int == 0:
            with torch.no_grad():
                lamb_hat = torch.max(std_) / std_
                c_ = (1 - alpha_ann) * c_ + alpha_ann * lamb_hat
                c_ = c_ / torch.sum(c_)

        loss = sum(c_[tind] * lcomp[tind] for tind in range(nsnaps))
        if epoch % 250 == 0:
            print("epoch:", epoch, "c_:", c_)

        #print(" i  am here ")
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"epoch {epoch+1}/{n_epochs}, loss={loss.item():.10f}, lr={optimizer.param_groups[0]['lr']:.5f}", end="\r")

    return net

def generate_noisy_training_data_batch(Dist, Ndim, tp, L, nsamples, nsnaps, device):
    sigma = geometric_sequence(L);
    transform_data = np.zeros((nsnaps,L,nsamples,Ndim+2))
    for tind in range(0,nsnaps):
        for i in range(0,L):
            for j in range(0,nsamples):
                mean = Dist[tind,j,0:Ndim]; cov = (sigma[i]**2)*np.eye(Ndim)
                transform_data[tind,i,j,0:Ndim] = np.random.multivariate_normal(mean, cov)
                transform_data[tind,i,j,Ndim] = sigma[i]
                transform_data[tind,i,j,Ndim+1] = tp[tind]

        print(" done with time ", tind)
            
    X_train = torch.tensor(transform_data, dtype=torch.float32,requires_grad=True,device=device)
    print(" X_train shape ", X_train.shape)
    X_data = torch.tensor(Dist, dtype=torch.float32,device=device)
    print(" X_data shape ", X_data.shape)

    X_mean = torch.zeros((Ndim+2,)); X_std = torch.ones((Ndim+2,))
    for i in range(0,Ndim+2):
        X_mean[i] = torch.tensor(np.mean(transform_data[:,:,:,i].flatten(),axis=0, keepdims=True), dtype=torch.float32)
        X_std[i]  = torch.tensor(np.std(transform_data[:,:,:,i].flatten(),axis=0, keepdims=True), dtype=torch.float32)

    X_mean = X_mean[np.newaxis,:]; X_std = X_std[np.newaxis,:]; 
    print(" X-mean shape ", X_mean.shape, " X-std shape ", X_std.shape)
    return X_train, X_data, X_mean, X_std, sigma

def DSM_training_batched(Ndim, net, X_train, X_data, sigma, nsnaps, nsamples, L, adp_flag=0, n_epochs=10001, bs=500, device='cpu'):

    print(device)
    
    Xtrain = torch.tensor(X_train, dtype=torch.float32).to(device)
    Xdata = torch.tensor(X_data, dtype=torch.float32).to(device)
    loader = FastTensorDataLoader(torch.permute(Xtrain, (2,0,1,3)), torch.permute(Xdata,(1,0,2)), batch_size=bs, shuffle=True)
    print(" number of batches ", len(loader))

    nb = len(loader) + 0;

    print(" Xtrain device ", Xtrain.device, " Xdata device ", Xdata.device)
    
    """
    Train the encoder with adaptive per-timepoint loss weighting.
    """
    optimizer = torch.optim.Adam(list(net.parameters()), lr=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(2500), int(5000), int(7500)], gamma=0.1)

    c_ = torch.ones((nsnaps,), dtype=torch.float32, device=device)
    alpha_ann = 0.5
    adapt_int = 10;
    weight_decay = 1e-4;

    sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=X_train.device)
    sigma_reshaped = sigma_tensor.view(L, 1, 1)

    for epoch in tqdm(range(int(n_epochs))):
        for j, (Xbatch, ybatch) in enumerate(loader):
            optimizer.zero_grad()

            Xbatch = Xbatch.clone().detach().requires_grad_(True)
            ybatch = ybatch.clone().detach().requires_grad_(True)
            
            Xbatch = torch.permute(Xbatch, (1,2,0,3))
            ybatch = torch.permute(ybatch, (1,0,2))

            uhat = net(Xbatch)
            lcomp = torch.zeros((nsnaps,), device=device)
            std_ = torch.zeros((nsnaps,), device=device)
            for tind in range(nsnaps):
                loss_sum = 0
                u_pred = uhat[tind]#.view(L, nsamples, Ndim)
                x_t    = Xbatch[tind, :, :, :Ndim]#.view(L, nsamples, Ndim)
                x_data = ybatch[tind, :, :Ndim][np.newaxis,:,:]
            
                u_true = (x_t - x_data) / (sigma_reshaped ** 2)
                residual = u_pred + u_true
                residual_squared = residual.pow(2).mean(dim=(1, 2))  # shape (L,)
                loss_sum = 0.5 * torch.sum((sigma_tensor ** 2) * residual_squared)

                if adp_flag == 1 and epoch % adapt_int == 0:
                    with torch.no_grad():
                        std_[tind] = loss_grad_std(loss_sum, net, device)
                lcomp[tind] = loss_sum

            if adp_flag == 1 and epoch % adapt_int == 0:
                with torch.no_grad():
                    lamb_hat = torch.max(std_) / std_
                    c_ = (1 - alpha_ann) * c_ + alpha_ann * lamb_hat
                    c_ = c_ / torch.sum(c_)
            
            loss = sum(c_[tind] * lcomp[tind] for tind in range(nsnaps))

            weight_norm = sum((p**2).sum() for p in net.parameters())
            loss = loss + weight_decay * weight_norm

            if epoch % 250 == 0:
                print("epoch:", epoch, "c_:", c_)
        
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        print(f"epoch {epoch+1}/{int(n_epochs)}, loss={loss.item():.10f}, lr={optimizer.param_groups[0]['lr']:.5f}", end="\r")

    return net