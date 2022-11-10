import sys
import math

import torch
from motion_prediction.net_module import module_wta

def meta_loss(hypos, M, labels, loss, k_top=1, relax=0): # for batch per step
    # relax=0, k_top=1 -> WTA
    # relax>0, k_top=1 -> Relaxed WTA
    # relax=0, k_top=n -> Evolving WTA
    assert((relax>=0)&(k_top>=0)), ('All parameters must be non-negative.')
    assert((relax<1)), ('Parameters exceed limits.')
    k_top = min(k_top, M)

    hy = hypos  # tensor - Mx[BxC] - [ts[1,..,C*M],...,ts[1,..,C*M]]
    gt = labels # tensor - BxC

    hyM = module_wta.disassemble(hy, M) # BxMxC
    # gts = torch.stack([gt for _ in range(M)], axis=1)
    gts = gt.repeat(1, M, 1)

    D = loss(hyM, gts) # BxM

    if   (relax==0) & (k_top==1): # meta-loss
        sum_loss = torch.mean(torch.min(D,dim=1).values)   
    elif (relax >0) & (k_top==1): # relaxed meta-loss
        sum_loss = (1-2*relax) * torch.mean(torch.min(D,dim=1).values)
        for i in range(M):
            sum_loss += relax/(M-1) * torch.mean(D[:,i])
    elif (relax==0) & (k_top >1): # envolving meta-loss
        sum_loss = 0
        topk = torch.topk(D, k_top, dim=1, largest=False, sorted=False).values # BxM
        for i in range(k_top):
            sum_loss += torch.mean(topk[:,i])
        sum_loss /= k_top
    else:
        raise ModuleNotFoundError('The mode is unkonwn. Check the parameters.')
    return sum_loss

def ameta_loss(hypos, M, labels, loss, k_top): # for batch per step
    '''
    Do a "clustering" for computing the loss
    '''
    hy = hypos  # tensor - Mx[BxC] - [ts[1,..,C*M],...,ts[1,..,C*M]]
    gt = labels # tensor - BxC
    hyM = module_wta.disassemble(hy, M) # BxMxC
    # gts = torch.stack([gt for _ in range(M)], axis=1)
    gts = gt.repeat(1, M, 1)

    D = loss(hyM, gts) # BxM

    if k_top <= 1: # NOTE adaptive
        Dmin = torch.min(D, dim=1).values
        Dmax = torch.max(D, dim=1).values
        Dthre = Dmin + 0.1 * (Dmax - Dmin)
        A = D<=Dthre.reshape(-1,1)
        DA = D*A

    if k_top == 0:
        Dmin = torch.min(D, dim=1).values
        Dmax = torch.max(D, dim=1).values
        Dthre = Dmin + 0.1 * (Dmax - Dmin)
        A = D<=Dthre.reshape(-1,1)
        D = torch.tile(Dmin.reshape(-1,1), (1,M))
        DA = D*A

    sum_loss = 0
    if k_top > 1:
        topk = torch.topk(D, k_top, dim=1, largest=False, sorted=False).values # BxM
        for i in range(k_top):
            sum_loss += torch.mean(topk[:,i])
        sum_loss /= k_top
    else: # k=0 or 1
        for i in range(M):
            sum_loss += torch.mean(DA[:,i])
        sum_loss /= M

    return sum_loss

def output2mdn(outputs, M, labels, loss, k_top=None):
    alp, mu, sigma = outputs[0], outputs[1], outputs[2]
    return loss(alp, mu, sigma, labels)

# XXX Overhaul version [deprecated]
def bmeta_loss(hypos, M, labels, loss, k_top, overhaul=False):
    '''
    Newly including "overhaul" and "pre-check"
    Here the "M" is actually "K"
    1. k>1, (pre-check) + stablization + overhaul + decay k
    2. k=1, adaptive update,like bmeta
    '''
    
    hy = hypos  # tensor - Mx[BxC] - [ts[1,..,C*M],...,ts[1,..,C*M]]
    gt = labels # tensor - BxC

    hyM = module_wta.disassemble(hy, M) # BxMxC
    gts = torch.stack([gt for _ in range(M)], axis=1)

    D = loss(hyM, gts) # BxM

    if k_top == 1:
        Dmin = torch.min(D, dim=1).values
        Dmax = torch.max(D, dim=1).values
        Dthre = Dmin + 0.1 * (Dmax - Dmin)
        A = D<=Dthre.reshape(-1,1)
        DA = D*A

    sum_loss = 0
    if k_top > 1:
        topk = torch.topk(D, k_top, dim=1, largest=False, sorted=False).values # BxM
        if overhaul: # overhaul
            topk_thre = (torch.max(topk, dim=1).values + torch.min(topk, dim=1).values) / 2
            for i in range(len(topk_thre)):
                topk_batch = topk[i,:]
                equilibrium = topk_batch[topk_batch>=topk_thre[i]]
                half_equilibrium = torch.topk(equilibrium, (len(equilibrium)+1)//2, largest=False, sorted=False).values
                sum_loss += (torch.sum(topk_batch[topk_batch<topk_thre[i]]) + torch.sum(half_equilibrium)) / k_top
            sum_loss /= len(topk_thre)
        else: # stabilization
            sum_loss = torch.sum(torch.mean(topk, dim=0))
            sum_loss /= k_top
    else: # adaptive
        # relax = 0.1
        sum_loss = torch.sum(torch.mean(DA, dim=0))
        # sum_loss = (1-2*relax) * torch.mean(torch.min(DA,dim=1).values)
        # for i in range(M):
        #     sum_loss += relax/(M-1) * torch.mean(D[:,i])
        sum_loss /= M

    return sum_loss

# XXX Confidence version [deprecated]
def lmeta_loss(hypos, M, labels, loss, k_top):
    hy_withc = hypos  # tensor - Mx[BxC] - [ts[1,..,C*M],...,ts[1,..,C*M]]
    gt = labels # tensor - BxC
    sfx = torch.nn.Softmax(dim=1)
    kld = torch.nn.KLDivLoss(reduction='batchmean')

    hyM_withc = module_wta.disassemble(hy_withc, M) # BxMxC
    gts = torch.stack([gt for _ in range(M)], axis=1)

    hyM = hyM_withc[:,:,:-1]
    hyC = hyM_withc[:,:,-1]

    D  = loss(hyM, gts) # BxM
    C = sfx(D)

    sum_loss = 0
    if k_top > 1:
        topk = torch.topk(D, k_top, dim=1, largest=False, sorted=False).values # BxM
        for i in range(k_top):
            sum_loss += torch.mean(topk[:,i])
        sum_loss /= k_top
    else: # k=0 or 1
        sum_loss = torch.mean(torch.min(D,dim=1).values)

    sum_loss += kld(hyC.log(), C) # (C * (C / hyC).log()).sum(dim=1).mean()

    return sum_loss

def cal_GauProb(mu, sigma, x):
    """
    Arguments:
        mu    (BxMxC) - The means of the Gaussians. 
        sigma (BxMxC) - The standard deviation of the Gaussians.
        x     (BxC)   - A batch of data points (coordinates of position).

    Return:
        probabilities (BxM): probability of each point in the probability
             distribution with the corresponding mu/sigma index.
            (Assume the dimensions of the output are independent to each other.)
    """
    x = x.unsqueeze(1).expand_as(mu) # BxC -> Bx1xC -> BxMxC
    prob = torch.rsqrt(torch.tensor(2*math.pi)) * torch.exp(-((x-mu)/sigma)**2 / 2) / sigma
    return torch.prod(prob, dim=2) # overall probability for all output's dimensions in each component, BxM

def cal_multiGauProb(alp, mu, sigma, x):
    '''
    Description:
        Return the probability of "data" given MoG parameters "mu" and "sigma".
    Arguments:
        (same as 'loss_NLL')
    Return:
        prob (Bx1) - The probability of each point in the distribution in the corresponding mu/sigma index.
    '''
    prob = alp * cal_GauProb(mu, sigma, x) # BxG
    prob = torch.sum(prob, dim=1) # Bx1, overall prob for each batch (sum is for all compos)
    return prob


def loss_NLL(alp, mu, sigma, data):
    '''
    Description:
        Calculates the negative log-likelihood loss.
    Arguments:
        alp   (BxM)   - Component's weight.
        mu    (BxMxC) - The means of the Gaussians. 
        sigma (BxMxC) - The standard deviation of the Gaussians.
        data  (BxC)   - A batch of data points.
    Return:
        NLL <value> - The negative log-likelihood loss.
    '''
    alp = alp/torch.sum(alp, dim=1) #normalization
    nll = -torch.log(cal_multiGauProb(alp, mu, sigma, data)) 
    return torch.mean(nll)

def loss_MaDist(alp, mu, sigma, data):
    '''
    Description:
        Calculates the weighted Mahalanobis distance.
    Arguments:
        alp   (BxG)   - Component's weight.
        mu    (BxGxC) - The means of the Gaussians. 
        sigma (BxGxC) - The standard deviation of the Gaussians.
        data  (BxC)   - Batches of data points.
    Return:
        MD  <list>  - The MD of each component.
        WMD <value> - The weighted MD.
    '''
    alp = alp/torch.sum(alp, dim=1) #normalization
    mu0 = (data.expand_as(mu)-mu) # (x-mu)
    S_inv_1 = 1/sigma[:,:,0] # S^-1 inversed covariance matrix
    S_inv_2 = 1/sigma[:,:,1]
    md = torch.sqrt( S_inv_1*mu0[:,:,0]**2 + S_inv_2*mu0[:,:,1]**2 ) # BxG
    return md, torch.sum(md*alp, dim=1)

def loss_CentralOracle(mu, data):
    '''
    Arguments:
        mu    (BxGxC) - The means of the Gaussians. 
        data  (BxC)   - Batches of data points.
    '''
    mse = torch.sum((mu - data.expand_as(mu))**2, dim=2) # BxG
    return torch.min(mse, dim=1).values


def loss_mse(data, labels): # for batch
    # data, labels - BxMxC
    squared_diff = torch.square(data-labels)
    squared_sum  = torch.sum(squared_diff, dim=2) # BxM
    loss = squared_sum/data.shape[0] # BxM
    return loss

def loss_msle(data, labels): # for batch
    # data, labels - BxMxC
    squared_diff = torch.square(torch.log(data)-torch.log(labels))
    squared_sum  = torch.sum(squared_diff, dim=2) # BxM
    loss = squared_sum/data.shape[0] # BxM
    return loss

def loss_mae(data, labels): # for batch
    # data, labels - BxMxC
    abs_diff = torch.abs(data-labels)
    abs_sum  = torch.sum(abs_diff, dim=2) # BxM
    loss = abs_sum/data.shape[0] # BxM
    return loss

def loss_nll(data, labels):
    # data, labels - BxMxC
    # data: For each batch [[x,y,sx,sy],[x,y,sx,sy],...]
    mu = data[:,:,:2]
    sigma = data[:,:,2:]
    nll = -torch.log(cal_GauProb(mu, sigma, labels) + 1e-6) # BxM
    return nll



