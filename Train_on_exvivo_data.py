from simulation_utils import *
from torch.autograd import grad
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from scipy.interpolate import griddata
from itertools import product, combinations
from scipy.io import savemat, loadmat
import scipy.io as sio
import matplotlib.pyplot as plt
from pinnutils import *
from chebyshev_utils import *
from GEOMLOSS import samples_loss
import os
from genesets import genesets
from genesets_exvivo import genesets_exvivo
import time as timeit
import argparse

import scvelo as scv
import scanpy as sc
import anndata as ad

from TrajectoryNet.parse import parser

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32768"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from scipy.linalg import sqrtm

def compute_fid_pytorch(X, Y):
    """
    Compute FID between two sets of samples X and Y.
    Inputs:
    - X: [N, D] generated samples (torch.Tensor)
    - Y: [N, D] real samples (torch.Tensor)
    Returns:
    - fid (float)
    """
    mu_x = X.mean(dim=0)
    mu_y = Y.mean(dim=0)

    cov_x = torch.cov(X.T)
    cov_y = torch.cov(Y.T)

    # Move to CPU + NumPy for sqrtm
    cov_x_np = cov_x.cpu().numpy()
    cov_y_np = cov_y.cpu().numpy()
    cov_prod_sqrt = sqrtm(cov_x_np @ cov_y_np)

    if np.iscomplexobj(cov_prod_sqrt):
        cov_prod_sqrt = cov_prod_sqrt.real  # eliminate imaginary parts due to numerical errors

    diff = mu_x - mu_y
    fid = diff.dot(diff) + torch.trace(cov_x + cov_y - 2 * torch.tensor(cov_prod_sqrt, dtype=torch.float32))

    return fid.item()

def load_data(path,nsamples,genes,time_str,cell_type_str,seed=0):
    Ndim = len(genes);
    args = parser.parse_args(args=[])
    args.dataset = path
    adata = sc.read_h5ad(args.dataset)

    #print(" observables ", adata.obs)

    unique_times = adata.obs[str(time_str)].unique()
    nsnaps = len(unique_times); 
    samples = np.zeros((len(unique_times),nsamples,Ndim))

    cell_type_categories = list(adata.obs[str(cell_type_str)].cat.categories)
    cell_type_to_int = {ct: i for i, ct in enumerate(cell_type_categories)}
    ind_array = np.zeros((nsnaps, nsamples), dtype=int)

    print(cell_type_categories, cell_type_to_int)
    for k in range(0,len(unique_times)):
        time_point = unique_times[k] # Replace with actual time (e.g., "10h" or 10)
        cells_at_time = adata[adata.obs[str(time_str)] == time_point]
        cell_names = cells_at_time.obs_names.tolist()
        selected = np.arange(0,nsamples)

        expression_t0 = cells_at_time[:, genes].X
        expression_t0_numpy = expression_t0.toarray() if hasattr(expression_t0, "toarray") else np.array(expression_t0)
        print(" shape ", k, " : ", expression_t0_numpy.shape)

        np.random.seed(seed)
        randind = np.arange(0, len(cell_names))
        half = int(nsamples/1)
        for ll in range(0,1):
            np.random.shuffle(randind); 
            selected =  randind[0:half] # np.arange(0,nsamples)
            cell_types_k = cells_at_time.obs[str(cell_type_str)].iloc[selected].values
            ind_array[k, ll*half:(ll+1)*half] = [cell_type_to_int[ct] for ct in cell_types_k]
            samples[k,ll*half:(ll+1)*half,0:Ndim] = expression_t0_numpy[selected,:]
            # samples[k,ll*half:(ll+1)*half,Ndim]   = 2*k

    return samples, unique_times, ind_array, adata.obs[str(cell_type_str)]

def mini_batchOT(Dist,nb,device):
    Ndim = Dist.shape[2];
    mDist = torch.zeros_like(Dist).to(device);
    bs = int(Dist.shape[1]/nb);
    ind = np.arange(0,Dist.shape[1])
    for k in range(0, nb):
        np.random.shuffle(ind)
        before_ot = torch.tensor(Dist[:,ind[0:bs],0:Ndim], dtype=torch.float32)
        batch_ot_samples = stitch_ot_trajectories(before_ot).to(device)
        mDist[:,k*bs:(k+1)*bs,:] = batch_ot_samples
        print(" done stitching ", k, "-th batch ", batch_ot_samples.shape)
    return mDist
        
# Train flow-matching model
def train_simulation_free(samples, Ndim, Nf, nsamples, tp, model_params_, fac, device, nmb, interp='cheb',dm='Langevin'):

    lx = model_params_[0]
    print(" Inferring with noise model ", dm)

    start_time = timeit.time()
    Dist = samples[:,:,0:Ndim]
    print(" # number of mini-batches ", nmb)

    batch_ot_samples = mini_batchOT(torch.tensor(samples[:,:,0:Ndim].copy(),dtype=torch.float32,device=device),nmb,device)
    print(" time spent stitching trajectories ", timeit.time() - start_time)

    uniform_kind = torch.tensor(np.linspace(tp[0],tp[-1],fac * tp.shape[0]))
    second_kind, weights = generate_secondkind_nodes(tp[0], tp[-1], fac * tp.shape[0])
    print(" data nodes ", tp)
    print(" nodes of second kind ", second_kind)

    data_nodes = torch.tensor(tp, dtype=torch.float32, device=device).expand(Dist.shape[1], -1)
    second_kind_nodes = torch.tensor(second_kind, dtype=torch.float32, device=device).expand(Dist.shape[1], -1)
    
    #batch_size = 500;
    print(" with batch_size ", batch_size )
    second_kind_batch = second_kind.expand(batch_size, -1).to(device)
    uniform_kind_batch = uniform_kind.expand(batch_size,-1).to(device)
    data_batch = torch.tensor(tp,dtype=torch.float32).expand(batch_size,-1).to(device)

    X_mean = torch.zeros((Ndim,)); X_std = torch.ones((Ndim,))
    for i in range(Ndim):
        X_mean[i] = torch.tensor(np.mean(Dist[:,:,i]), dtype=torch.float32)
        X_std[i]  = torch.tensor(np.std(Dist[:,:,i]), dtype=torch.float32)
    X_mean = X_mean[np.newaxis,:]; X_std = X_std[np.newaxis,:]
    print(" X-mean shape ", X_mean.shape, " X-std shape ", X_std.shape)

    drift_net_FM = SpectralNormDNN(sizes=[Ndim, Nf, Nf, Nf, Nf, Ndim], mean=X_mean.to(device), std=X_std.to(device), seed=seed, activation=nn.ELU()).to(device)
    print("#parameters:", sum(p.numel() for p in drift_net_FM.parameters() if p.requires_grad))

    optimizer_FM = torch.optim.Adam([{'params': drift_net_FM.parameters(), 'lr': 1e-3}])
    scheduler_FM = MultiStepLR(optimizer_FM, milestones=[1000, 1500, 8000, 15000], gamma=0.1)

    if(interp=='cheb'):
        data_full = torch.tensor(tp,dtype=torch.float32).expand(batch_ot_samples.shape[1],-1).to(device)
        #data_full = uniform_kind.expand(batch_ot_samples.shape[1],-1).to(device)
        data_highres = uniform_kind.expand(batch_ot_samples.shape[1],-1).to(device)
        best_lam,_,_,_ = select_best_lambda(batch_ot_samples, batch_ot_samples.shape[1], data_full, data_full, Ndim, device);
        print(" best_lam ", best_lam)
    
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    n_epochs = 2000; 
    for epoch in range(n_epochs):
        optimizer_FM.zero_grad()

        Xtrain, ytrain, _ = compute_conditional_distributions_chebyshev(batch_ot_samples, batch_size, data_batch, uniform_kind_batch,reg_=best_lam,sigma=0.001,err_flag=False,device=device)
        inp_ = Xtrain[:, :Ndim].clone()
        inp_.requires_grad_()

        drift = drift_net_FM(inp_)

        if(dm=='Langevin'):
            Dx = drift + lx*Xtrain[:,0:Ndim]
            div_d = div_neural_pos(Dx,inp_,Ndim);
            fold  = (drift - lx*Xtrain[:,0:Ndim]) - (0.5)*div_d - 0.5*(Dx)*scoreNet(Xtrain[:,:])
        elif(dm=='Additive'):
            fold  = (drift - lx*Xtrain[:,0:Ndim]) - 0.5*scoreNet(Xtrain[:,:]) 
        elif(dm=='Mult'):
            ones  = torch.ones_like(Xtrain[:,0:Ndim])
            fold  = (drift - lx*Xtrain[:,0:Ndim]) - 0.5*ones - 0.5*Xtrain[:,0:Ndim]*scoreNet(Xtrain[:,:])
        else:
            fold  = drift - lx*Xtrain[:,0:Ndim]
        
        total_loss = torch.mean(((fold - ytrain)**2).flatten())

        total_loss.backward()
        optimizer_FM.step()
        scheduler_FM.step()

        if epoch % 200 == 0:
            print("epoch {}/{}, loss={:.10f}, lr={:,.5f}".format(
                epoch+1, n_epochs, total_loss.item(), optimizer_FM.param_groups[0]['lr']))

    training_time = timeit.time() - start_time

    mem_current = torch.cuda.memory_allocated(device) / (1024**2)
    mem_max = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(" total-time taken ", training_time)
    print( " current memory ", mem_current, " max memory ", mem_max)
    
    return drift_net_FM, training_time, mem_current, mem_max

def div_neural_pos(drift,inp_,Ndim):
    div = drift.clone() + 0;
    for i in range(0, Ndim):
        out_ = drift[:,i]
        gradient = grad(outputs=out_, inputs=inp_,
                      grad_outputs=torch.ones_like(out_), 
                      create_graph=True)[0]
        div[:,i] = gradient[:,i].clone() + 0;
    
    return div

def generate_data_DSM(eps,maxiter,infNet,Nsamples,init_,time_,L):
    eps = 1e-4;
    sol = torch.tensor(init_,dtype=torch.float32).to(device)
    sol[:,Ndim+1] = time_
    sigma = geometric_sequence(L);
    for k in range(0,L):
        alpha = eps*((sigma[k]**2)/(sigma[L-1]**2))
        sol[:,Ndim] = sigma[k]
        for t in range(0,500):
            z = torch.normal(0, 1, size=(Nsamples, Ndim)).to(device)
            guru = infNet(sol) 
            sol[:,0:Ndim] = sol[:,0:Ndim] + 0.5*alpha*guru + np.sqrt(alpha)*z
    print(" final mean ",torch.mean(sol,axis=0)[0:Ndim])       
    sol = sol.cpu().data.numpy().reshape(Nsamples,Ndim+2)
    return sol

def validate_score(scoreNet,Ndim,nsnaps,Dist,L):
    nsamples = Dist.shape[1];
    loss = samples_loss.SamplesLoss("energy")
    eps = 5e-3; maxiter = 100; 
    for tinf in range(0,nsnaps):
        with torch.no_grad():
            infNet = scoreNet.eval()
            init_  = np.random.uniform(1.0,4.0,(nsamples,Ndim+2))  #
            init_[:,0:Ndim] = Dist[tinf,:,0:Ndim]
            print(" true mean ",np.mean(Dist[tinf,:,0:Ndim],axis=0)[0:Ndim])
            sol = generate_data_DSM(eps,maxiter,infNet,nsamples,init_,tp[tinf],L);
            divergence = loss(torch.tensor(sol[:,0:Ndim],dtype=torch.float32,device=device),torch.tensor(Dist[tinf,:,0:Ndim],dtype=torch.float32,device=device))
            print(" divergence for tinf ", tinf, " is ", divergence)
            FID = compute_fid_pytorch(torch.tensor(sol[:,0:Ndim],dtype=torch.float32,device='cpu'),torch.tensor(Dist[tinf,:,0:Ndim],dtype=torch.float32,device='cpu'))
            print(" divergence for tinf ", tinf, " is ", FID)
            print(" \n ")
    return scoreNet

def setup_model(Ndim, mean, std, Np, seed_, spectral_flag=True):
    """
    Setup the encoder model and optimizer.
    """
    if(spectral_flag):
        print(" Applying Spectral-Norms ")
        encoder = SpectralNormDNN(
            sizes=[Ndim + 2, Np, Np, Np, Np, Np, Ndim],
            mean=mean.to(device), std=std.to(device),
            seed=seed_, activation=nn.ELU()
        ).to(device)
    else:
        print(" Vanilla DNN ")
        encoder = DNN(
            sizes=[Ndim + 2, Np, Np, Np, Np, Np, Ndim],
            mean=mean.to(device), std=std.to(device),
            seed=seed_, activation=nn.ELU()
        ).to(device)       
    
    print("#parameters:", sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    return encoder, list(encoder.parameters())

parser_param = argparse.ArgumentParser(description='Runtime arguments for simulation.')
parser_param.add_argument('--seed', type=int, default=0, help='Random seed')
parser_param.add_argument('--loadp', type=int, default=2, help='Wasserstein norm')
parser_param.add_argument('--epsilon', type=float, default=0.05, help='blur parameter of the Wasserstein')
parser_param.add_argument('--fac', type=int, required=True, help='Scaling factor that sets resolution of the evaluation points along time-axis')
parser_param.add_argument('--nmb', type=int, required=True, help='Number of mini-batches used for OT stitching')
parser_param.add_argument('--nsamples', type=int, required=True, help='Number of samples')
parser_param.add_argument('--Np', type=int, default=50, help='Number of parameters of the score model')
parser_param.add_argument('--Nf', type=int, default=50, help='Number of parameters of the force model')
parser_param.add_argument('--batch_size', type=int, default=500, help='batch-size used in CFM')
parser_param.add_argument('--simflag', type=str2bool, required=True,
                    help='True: if you need simulation-based inference (use --simflag True or False)')
parser_param.add_argument('--nepochs', type=int,default=5000,help='Epochs for training score models')
parser_param.add_argument('--batch_size_score', type=int,default=6000,help='batch-size used in CFM')
parser_param.add_argument('--dm', type=str, required=True, help='Choose diffusion model: Langevin, Mult, Additive, ODE')
parser_param.add_argument('--spectral_flag', type=str2bool, required=True,
                    help='True: if you need spectral normalization')
parser_param.add_argument('--geneset_num', type=int, default=3,help='select genes used for inference from the model genesets')
parser_param.add_argument('--lx', type=float, default=0.7, help='degradation rate of mRNA')

args = parser_param.parse_args()

# Access variables
seed = args.seed
loadp = args.loadp
epsilon = np.double(args.epsilon)
fac = args.fac
nmb = args.nmb
nsamples = args.nsamples
Np = args.Np
Nf = args.Nf
simflag = args.simflag
batch_size = args.batch_size
nepochs = args.nepochs
batch_size_score = args.batch_size_score
dm = args.dm
n_epochs = args.nepochs
spectral_flag = args.spectral_flag
geneset_num = args.geneset_num
lx = args.lx

print(" done loading parameters ")
for key, value in vars(args).items():
    print(f"  {key}: {value}")

genes = genesets_exvivo[geneset_num] #['fli1', 'klf1','gata1','gata2','gfi1b','runx1','tal1','jun','spi1','zfpm1','lmo2','etv6','erg','mef2c']
Ndim = len(genes);
print(" genes being used ", genes)
model_params = [lx];

path = '/mnt/home/vchardes/ceph/datasets/HSC_data/10XChromiumV3_10.1038_s41467-021-27159-x_10.5281_zenodo.5291737_exvivo.h5ad'
Dist, tp, ind_array, cell_types = load_data(path,nsamples,genes,"day","cell_type",seed)
print(cell_types)

print(" time ", tp, " Dist shape ", Dist.shape)

################################ score-matching ################################
nsamples_ = Dist[:,0:nsamples,:].shape[1]; nsnaps = Dist.shape[0]; L = 5;
#model_path = f"/mnt/ceph/users/smaddu/stochastic_inference/scaletest_score_bifur_N{10000}_Ndim{Ndim}.pth"

if(spectral_flag):
    model_path = f"/mnt/ceph/users/smaddu/stochastic_inference/score_newset_spectral_exvivo_HSPC_general_N{6000}_Ndim{Ndim}_Np{Np}_seed{seed}_set{geneset_num}.pth"
else:
    model_path = f"/mnt/ceph/users/smaddu/stochastic_inference/score_newset_exvivo_HSPC_general_N{6000}_Ndim{Ndim}_Np{Np}_seed{seed}_set{geneset_num}.pth"
    
if os.path.exists(model_path):
    print(f"Model already exists at {model_path}, skipping training.")
    scoreNet = torch.load(model_path)
    scoreNet = scoreNet.eval()
else:
    print("Model not found. Starting training...")    
    X_train, X_data, X_mean, X_std, sigma = generate_noisy_training_data_batch(Dist[:,0:nsamples,:], Ndim, tp, L, nsamples, nsnaps, device);
    scoreNet, params = setup_model(Ndim, X_mean, X_std, Np, 0, spectral_flag);
    start_time = time.time()

    scoreNet = DSM_training_batched(Ndim, scoreNet, X_train, X_data, sigma, nsnaps, nsamples, L, adp_flag=1, n_epochs=n_epochs, bs=batch_size_score, device=device);
    print(" time-taken ", time.time() - start_time)
    torch.save(scoreNet, model_path)
    scoreNet = scoreNet.eval()

################################ validate-score ##################################
_ = validate_score(scoreNet,Ndim,nsnaps,Dist[:,:,0:Ndim],L);
# # ################################ validate-score ##################################

samples = np.zeros((tp.shape[0],nsamples,Ndim+2))
samples[:,:,0:Ndim] = Dist[:,:,0:Ndim]
samples[:,:,Ndim]   = 0.01;
samples[:,:,Ndim+1] = tp[:,None]

###########################################  loading data ######################################## 

if(simflag):
    drift_net, time_SB, mem_curr_SB, mem_max_SB, acc_SB = train_simulation_based(samples, Ndim, Nf, nsamples, tp, model_params, epsilon, loadp, seed, device);
    mdic = {
        'mem_current_SB': mem_curr_SB,
        'mem_max_SB': mem_max_SB,
        'time_SB': time_SB,
        'acc_SB': acc_SB,
    }
    savemat("/mnt/ceph/users/smaddu/FMpaper/exvivoHSPC_N"+str(nsamples)+"_Ndim"+str(Ndim)+"_Np"+str(Np)+"_Nf"+str(Nf)+"_fac"+str(fac)+".mat", mdic)

# # #######################################################################################################################
# # #######################################################################################################################

drift_net_FM_cheb, time_FM_cheb, mem_curr_FM_cheb, mem_max_FM_cheb = train_simulation_free(samples, Ndim, Nf, nsamples, tp, model_params, fac, device, nmb, interp='cheb', dm=dm);
print(" Done with FM with Chebyshev Interpolant ")
print(" \n ")

if(spectral_flag):
    force_model_path = f"/mnt/ceph/users/smaddu/stochastic_inference/fm_newset_fullspectral_exvivoHSPC_N{nsamples}_Ndim{Ndim}_Np{Np}_Nf{Nf}_fac{fac}_seed{seed}_dm{dm}_lx{lx}_set{geneset_num}.path"
    torch.save(drift_net_FM_cheb, force_model_path)   
else:
    force_model_path = f"/mnt/ceph/users/smaddu/stochastic_inference/fm_newset_exvivoHSPC_N{nsamples}_Ndim{Ndim}_Np{Np}_Nf{Nf}_fac{fac}_seed{seed}_dm{dm}_lx{lx}_set{geneset_num}.path"
    torch.save(drift_net_FM_cheb, force_model_path)      



