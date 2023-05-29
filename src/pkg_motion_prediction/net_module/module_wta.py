"""
A module for a WTA loss (meta-loss) based layer
"""
import sys

import torch
import torch.nn as nn
from torch import tensor as ts

'''
Process:
    Input:x -> Some model  (body) -> Characteristic vector: z (feature)
            -> Swarm layer (head) -> Hypotheses     vector: h (output)
Further:
    h  ---(optional)-->  probabilistic distribution
'''

class MultiHypothesisModule(nn.Module):
    """ 
    A multiple hypothesis Module
    
    Symbols:
        B - Batch size
        M - Number of hypotheses
        F - Feature's dimension
        C - Output's dimension for one hypothesis
    
    Arguments:
        dim_fea (int): the feature's dimension
        dim_out (int): the output's dimension for one hypothesis
        num_hypos (int): the number of hypotheses
    
    Input:
        minibatch (BxF)
    Output:
        hypotheses (Bx(M*C))
    """
    def __init__(self, dim_fea, dim_out, num_hypos):
        super(MultiHypothesisModule, self).__init__()
        self.dim_fea = dim_fea
        self.dim_out = dim_out # for one hypothesis
        self.M = num_hypos

        self.layer_hypos = nn.Linear(dim_fea, dim_out*num_hypos) # for hypotheses

    def forward(self, x):
        hypos = self.layer_hypos(x) # Bx(CxM)
        return hypos

# XXX deprecated
class AdaptiveSwarmModule(nn.Module):
    """ 
    Adaptive Swarm Module
    Also output the weight of each hypo
    """
    def __init__(self, dim_fea, dim_out, num_swarms):
        super(AdaptiveSwarmModule, self).__init__()
        self.dim_fea = dim_fea
        self.dim_out = dim_out # for one hypothesis
        self.M = num_swarms

        self.sfx = nn.Softmax(dim=1)
        self.layer_hypos = nn.Linear(dim_fea, (dim_out+1)*num_swarms) # for hypotheses

    def forward(self, x):
        hypos = self.layer_hypos(x) # Bx((C+1)xM)
        hyM = disassemble(hypos, self.M)
        hyM[:,:,-1] = self.sfx(hyM[:,:,-1].clone())
        hypos = assemble(hyM)

        return hypos

def assemble(hypos_M):
    nbatchs = hypos_M.shape[0]
    return hypos_M.reshape(nbatchs,-1)

def disassemble(hypos, M):
    nbatchs = hypos.shape[0]
    return hypos.reshape(nbatchs,M,-1)


if __name__ == '__main__':
    M = 2
    mod = MultiHypothesisModule(dim_fea=4, dim_out=2, num_hypos=M)
    test_batch = ts([[1,2,1,2],[3,2,2,1],[4,4,2,2]]).float()
    gt = ts([[1.5,3],[2,4],[3,6]]).float()
    output = mod.forward(test_batch)
    outputM = disassemble(output, M)
    output_comp = assemble(outputM)
    print(output)
    print(outputM)
    print(gt)