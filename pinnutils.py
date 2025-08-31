import os
import time
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn

from torch.autograd import grad
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

import sys
import torch.nn.utils as nn_utils

class SpectralNormDNN(nn.Module):
    
    def __init__(self, sizes, mean=0, std=1, seed=0, activation=nn.Tanh()):
        super(SpectralNormDNN, self).__init__()
        
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.bn = BatchNorm(mean, std)

        layers = []
        for i in range(len(sizes) - 2):
            linear = nn.Linear(sizes[i], sizes[i + 1])
            linear = nn_utils.spectral_norm(linear)  
            layers.append(linear)
            layers.append(activation)
        
        final_linear = nn.Linear(sizes[-2], sizes[-1])
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.bn(x))

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

class BatchNorm(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
        
    def __call__(self, x):
        return (x-self.mean)/self.std
    
class BatchNormCNN(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
        
    def __call__(self,x):
        norm  = transforms.Normalize(self.mean,self.std)
        imgs_norm = torch.stack([norm(img_) for img_ in x], dim=3)
        imgs_norm = imgs_norm.permute(3,0,1,2)
        return imgs_norm

class LayerNoWN(nn.Module):
    def __init__(self, in_features, out_features, seed, activation):
        super(LayerNoWN, self).__init__()
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
            
        gain = 5/3 if isinstance(activation, nn.Tanh) else 1
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.zeros_(self.linear.bias)
        
        self.linear = self.linear
        
    def forward(self, x):
        return self.linear(x)

class PINNNoWN(nn.Module):
    
    def __init__(self, sizes, mean=0, std=1, seed=0, activation=nn.Tanh()):
        super(PINNNoWN, self).__init__()
        
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

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class Sin(nn.Module):
    
    def forward(self, x):
        return torch.sin(x)

######################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(SpectralConv2d, self).__init__()
        
        torch.manual_seed(0) # setting seed
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1./(in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, self.modes, dtype=torch.complex64))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, self.modes, dtype=torch.complex64))

    def compl_mul1d(self, input, weights):   
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.tensor:
        
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.complex64)
        out_ft[:,:,:self.modes,:self.modes] = self.compl_mul1d(x_ft[:,:,:self.modes,:self.modes], self.weights1)
        out_ft[:,:,-self.modes:,:self.modes] = self.compl_mul1d(x_ft[:,:,-self.modes:,:self.modes], self.weights2)
        x = torch.fft.irfft2(out_ft,s=(x.size(-2), x.size(-1)))
        return x
    
class FNO2d(nn.Module):
    def __init__(self, Ninp, Nout, modes, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: a driving function observed at T timesteps + 1 locations (u(1, x), ..., u(T, x),  x).
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.modes = modes
        self.width = width

        self.conv0 = SpectralConv2d(Ninp,self.width,self.modes)
        self.conv1 = SpectralConv2d(self.width,self.width,self.modes)
        # self.conv2 = SpectralConv2d(self.width,self.width,self.modes)
        # self.conv3 = SpectralConv2d(self.width,self.width,self.modes)
        self.conv4 = SpectralConv2d(self.width,Nout,self.modes)
        
        # self.w0 = nn.Conv2d(Ninp, self.width, kernel_size=1)
        # self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1)
        # self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1)
        # # self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1)
        # self.w4 = nn.Conv2d(self.width, Nout, 1)

    def forward(self, x):
        x1 = self.conv0(x); #x2 = self.w0(x); 
        x = x1 #+ x2; 
        x = Sin()(x)
        
        x1 = self.conv1(x); #x2 = self.w1(x); 
        x = x1 #+ x2; 
        x = Sin()(x)
        
#         x1 = self.conv2(x); #x2 = self.w2(x); 
#         x = x1 #+ x2; 
#         x = Sin()(x)
        
        # x1 = self.conv3(x); x2 = self.w3(x);
        # x = x1 + x2; x = Sin()(x)
        
        x1 = self.conv4(x); #x2 = self.w4(x); 
        x = x1 #+ x2;
        return x    
    
def generate_snaps(list_, n_snaps, seed):
    
    print(" length of list_ ", len(list_))
    
    np.random.seed(seed);
    temp  = np.arange(0, len(list_));
    np.random.shuffle(temp);
    ind = temp[0:n_snaps];
    snaps = list_[ind];
    print(" removing... ", snaps)
    list_ = np.asarray([i for i in list_ if i not in list_[ind]])
    
    return snaps, list_



def loss_grad_std_full(loss, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32,device=device)
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        if(m == 0):
            w = grad(loss, m.weight, retain_graph=True)[0]
            b = grad(loss, m.bias, retain_graph=True)[0]        
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = grad(loss, m.weight, retain_graph=True)[0]
            b = grad(loss, m.bias, retain_graph=True)[0]        
            grad_ = torch.cat((grad_,w.view(-1), b))
            
    return torch.std(grad_)

def loss_grad_max_full(loss, net, lambg=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32,device=device)
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        if(m == 0):
            w = torch.abs(lambg*grad(loss, m.weight, retain_graph=True)[0])
            b = torch.abs(lambg*grad(loss, m.bias, retain_graph=True)[0])        
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = torch.abs(lambg*grad(loss, m.weight, retain_graph=True)[0])
            b = torch.abs(lambg*grad(loss, m.bias, retain_graph=True)[0])        
            grad_ = torch.cat((grad_,w.view(-1), b))
    
    return torch.max(grad_), torch.mean(grad_)

def network_gradient(loss,net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32,device=device)
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        if(m == 0):
            w = grad(loss, m.weight, retain_graph=True)[0]
            b = grad(loss, m.bias, retain_graph=True)[0]        
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = grad(loss, m.weight, retain_graph=True)[0]
            b = grad(loss, m.bias, retain_graph=True)[0]        
            grad_ = torch.cat((grad_,w.view(-1), b))
        
    return grad_

def reweight(d_, num_tasks, eps):
    nz_ind = np.where(d_>eps)
    z_ind  = np.delete(np.arange(num_tasks),nz_ind)
    if(len(z_ind)==0):
        return d_
    d_[nz_ind] = d_[nz_ind] - eps/len(d_[nz_ind])
    d_[z_ind]  = eps/len(z_ind)
    return d_

def solver_mine(Q, num_tasks, tol, maxiter=500):
    alphas = (1./num_tasks)*np.ones((num_tasks,))
    direct = np.zeros((num_tasks,2))

    for it in range(0, maxiter):
        ind_vec = np.zeros((num_tasks,));
        grad = Q @ alphas
        idx_oracle   = np.argmin(grad);
        ind_vec[idx_oracle] = 1.0;

        direct[:,0] = ind_vec; direct[:,1] = alphas;
        MM = (direct.T @ Q) @ direct

        if(MM[0,1] >= MM[0,0]):
            step_size = 1.0;
        elif(MM[0,1] >= MM[1,1]):
            step_size = 0;
        else:
            step_size = (MM[1,1] - MM[0,1])/(MM[0,0] + MM[1,1] - MM[0,1] - MM[1,0])

        alphas = (1. - step_size) * alphas
        alphas[idx_oracle] = alphas[idx_oracle] + step_size * ind_vec[idx_oracle]

    return reweight(alphas,num_tasks,tol)

# better/faster weight computation
def loss_grad_std_wn(loss, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    for elem in grad(loss, net.parameters(), retain_graph=True):
        grad_ = torch.cat((grad_, elem.view(-1)))
        
    return torch.std(grad_)

def loss_grad_max_wn(loss, net, lambg=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    for elem in grad(loss, net.parameters(), retain_graph=True):
        grad_ = torch.cat((grad_, elem.view(-1)))
        
    grad_ = torch.abs(lambg*grad_)
        
    return torch.max(grad_), torch.mean(grad_)

def network_gradient_wn(loss, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    for elem in grad(loss, net.parameters(), retain_graph=True):
        grad_ = torch.cat((grad_, elem.view(-1)))
        
    return grad_

def loss_grad_std(loss, net, device):
    var = []
    siz = []
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        
        w = grad(loss, m.weight, retain_graph=True)[0]
        b = grad(loss, m.bias, retain_graph=True)[0]
        
        wb = torch.cat((w.view(-1), b))
        
        nit  = torch.numel(wb)
        var.append((nit - 1) * torch.var(wb))
        siz.append(nit)

    vart = torch.tensor(var, dtype=torch.float32,device=device)
    sizt = torch.tensor(siz, dtype=torch.float32,device=device)
    
    return torch.sqrt(torch.sum(vart)/(torch.sum(sizt) - len(sizt)))
