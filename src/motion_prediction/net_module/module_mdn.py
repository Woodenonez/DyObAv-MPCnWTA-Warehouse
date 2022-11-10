import os, sys

import torch
from torch import nn

class ClassicMixtureDensityModule(nn.Module):
    def __init__(self, dim_input, dim_output, num_components):
        super(ClassicMixtureDensityModule, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.M = num_components

        self.layer_mapping = nn.Linear(dim_input, (2*dim_output+1)*num_components)
        self.layer_alpha = nn.Softmax(dim=1) # If 1, go along each row

    def forward(self, x):
        p = self.layer_mapping(x)
        alpha = self.layer_alpha(p[:,:self.M])
        mu    = p[:,self.M:(self.dim_output+1)*self.M]
        sigma = torch.exp(p[:, (self.dim_output+1)*self.M:])
        mu    = mu.view(-1, self.M, self.dim_output)
        sigma = sigma.view(-1, self.M, self.dim_output)
        return alpha, mu, sigma

class SamplingMixtureDensityModule(nn.Module):
    def __init__(self, dim_input, num_hypos, num_gaus):
        super(SamplingMixtureDensityModule, self).__init__()
        self.dim_input  = dim_input
        self.K = num_hypos
        self.M = num_gaus

        self.myMLP = nn.Linear(dim_input*num_hypos, num_hypos*num_gaus)
        self.sfx = nn.Softmax(dim=2) # for each hypo

    def forward(self, x, device='cuda'): # x as a feature vector, in nbatch * dimension of x
        '''
            x: Bx(KxC)
            gamma = r1,1 r1,2 ... r1,M
                    r2,1 r2,2 ... r2,M
                    ...
                    rN,1 rN,2 ... rN,M
            alpha = a1, a2, ... aM
            mu = u1,1 ...
                 u2,1 ...
                 ...
                 uM,1 ...
        '''
        z = self.myMLP(x).reshape((-1, self.K, self.M))
        xK = x.reshape((-1, self.K, self.dim_input))
        gamma = self.sfx(z) # BxKxM
        alpha = torch.sum(gamma, axis=1)/self.K # BxM
        mu    = torch.zeros(x.shape[0], self.M, self.dim_input).to(device)
        sigma = torch.zeros(x.shape[0], self.M, self.dim_input).to(device)
        for i in range(self.M):
            mu[:,i,:]    = torch.sum(gamma[:,:,i].unsqueeze(2) * xK, axis=1) / torch.sum(gamma[:,:,i], dim=1).unsqueeze(1)
            sigma[:,i,:] = torch.sum(gamma[:,:,i].unsqueeze(2) * (xK-mu[:,i,:].unsqueeze(1))**2, axis=1) / torch.sum(gamma[:,:,i], dim=1).unsqueeze(1)
        return alpha, mu, sigma

def take_mainCompo(alp, mu, sigma, main=3):
    '''
    Description:
        Take several main components from a GMM.
    Arguments:
        main <int> - The number of components to take.
    Return:
        main_alp   <tensor> - Weights of selected components.
        main_mu    <tensor> - Means of selected components.
        main_sigma <tensor> - Variances of selected components.
    '''
    if len(alp[0,:])<=main:
        return alp, mu, sigma
    alp   = alp[0,:]
    mu    = mu[0,:,:]
    sigma = sigma[0,:,:]
    main_alp   = alp[:main]       # placeholder
    main_mu    = mu[:main,:]       # placeholder
    main_sigma = sigma[:main,:] # placeholder
    _, indices = torch.sort(alp) # ascending order
    for i in range(1,main+1):
        idx = indices[-i].item() # largest to smallest
        main_alp[i-1]     = alp[idx]
        main_mu[i-1,:]    = mu[idx,:]
        main_sigma[i-1,:] = sigma[idx,:]
    return main_alp.unsqueeze(0), main_mu.unsqueeze(0), main_sigma.unsqueeze(0) # insert the "batch" dimension

def take_goodCompo(alp, mu, sigma, thre=0.1):
    '''
    Description:
        Take several non-degraded components from a GMM.
    Arguments:
        thre <float> - The threshold of being a good components comparing to the one with the largest weight.
    Return:
        good_alp   <tensor> - Weights of selected components.
        good_mu    <tensor> - Means of selected components.
        good_sigma <tensor> - Variances of selected components.
    '''
    # if len(alp[0,:])<=1:
    if len(alp)<=1:
        return alp, mu, sigma
    # alp   = alp[0,:]
    # mu    = mu[0,:,:]
    # sigma = sigma[0,:,:]
    idx = (alp>thre*max(alp))
    good_alp   = alp[idx]
    good_mu    = mu[idx,:]
    good_sigma = sigma[idx,:]
    return good_alp, good_mu, good_sigma # .unsqueeze(0): insert the "batch" dimension