import torch
import numpy as np
from os.path import expanduser
import matplotlib.pyplot as plt
import torch.nn as nn
import utils
import wandb

home = expanduser("~")
epsilon = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TFeps = torch.tensor(1e-5, dtype=torch.float32).to(device)


# function c2r contatenate complex input as new axis two two real inputs
# different from tensorflow, the 2nd channel stands for the real and imaginary part.
c2r = lambda x: torch.stack([x.real, x.imag], axis=1)
#r2c takes the last dimension of real input and converts to complex
r2c = lambda x: torch.complex(x[:,0,...], x[:,1,...])


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, last_layer=False):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv.weight)  # Xavier initialization
        #self.bn = nn.BatchNorm2d(out_channels) # inplace=True to save memory
        self.relu = nn.ReLU(inplace=True)
        self.last_layer = last_layer

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        if not self.last_layer:  # apply ReLU activation, except for the last layer
            x = self.relu(x)
        return x


class dw(nn.Module):
    '''
    This function create a layer of CNN consisting of convolution, batch-norm,
    and ReLU. Last layer does not have ReLU to avoid truncating the negative
    part of the learned noise and alias patterns.
    '''
    def __init__(self):
        super(dw, self).__init__()

        self.layer1 = CNNLayer(2, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = CNNLayer(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer3 = CNNLayer(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer4 = CNNLayer(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer5 = CNNLayer(64, 2, kernel_size=3, stride=1, padding=1, last_layer=True)
        self.shortcut = nn.Identity()
        

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # ensure the network learns the residual
        return shortcut + x
    
    
class Aclass:
    '''
    Implementation of the A operator in the MoDL paper
    Data consistency operator using CG-SENSE
    '''
    def __init__(self, csm, mask, lam):
        # get the size of mask
        s = mask.shape # the final 2 dim is nrow and ncol
        self.nrow, self.ncol = s[-2], s[-1]
        self.csm = csm
        self.mask = mask
        self.lam = lam
        self.SF = torch.complex(torch.sqrt(torch.tensor(self.nrow*self.ncol, dtype=torch.float32)), torch.tensor(0.0, dtype=torch.float32)).to(device)
    def myAtA(self,img):
        # img is nbatch x nrow x ncol, I want to broadcast it to nbatch x ncoil x nrow x ncol
        img1 = img.unsqueeze(1).repeat(1, self.csm.shape[1], 1, 1)
        coilImage = self.csm * img1
        kspace = torch.fft.fft2(coilImage)/self.SF
        temp = kspace*self.mask.unsqueeze(1).repeat(1, self.csm.shape[1], 1, 1)
        coilImgs = torch.fft.ifft2(temp)*self.SF
        coilComb = torch.sum(coilImgs*self.csm.conj(), axis=1) + self.lam*img
        #coilComb = coilComb + self.lam*img
        return coilComb

def myCG(A, rhs):
    '''
    Complex conjugate gradient on complex data
    '''
    rhs = r2c(rhs)
    def body(i, rTr, x, r, p):
        Ap = A.myAtA(p)
        alpha = rTr / (torch.sum(p.conj()*Ap)).real.to(torch.float32)
        alpha = torch.complex(alpha, torch.tensor(0.).to(device))
        x = x + alpha *p 
        r = r - alpha * Ap
        # take the real part
        rTrNew = (torch.sum(r.conj()*r)).real.to(torch.float32)
        beta = rTrNew / rTr
        beta = torch.complex(beta, torch.tensor(0.).to(device))
        p = r + beta * p
        return i+1, rTrNew, x, r, p
    
    # the initial values of the loop variables
    x = torch.zeros_like(rhs)
    i,r,p = 0, rhs, rhs
    # This should yield cast the complex to real, but no worries, 
    rTr = torch.sum(r.conj()*r).real.to(torch.float32)
    loopVar = i, rTr, x, r, p

    while i< 11 and rTr>1e-10:
        i,rTr,x,r,p = body(i, rTr, x, r, p)
    
    out = x
    return out

class dc(nn.Module):
    def __init__(self):
        super(dc, self).__init__()
        
    def forward(self, rhs, csm, mask, lam1):
        lam2 = torch.complex(lam1, torch.tensor(0.).to(device))
        Aobj = Aclass(csm, mask, lam2)
        y = myCG(Aobj, rhs)
        return y
    

class MoDL(nn.Module):
    def __init__(self):
        super(MoDL, self).__init__()
        self.dw = dw()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)
        self.dc = dc()
        
    def forward(self, atb, csm, mask, logger):
        # plot the atb
        out = self.dw(atb)
        rhs = atb + self.lam*out
        # plot the rhs
        logger.log({'lam': self.lam.item()})
        # rhs: nbatch x 2 x nrow x ncol, csm: nbatch*ncoil x nrow x ncol, mask: nbatch x nrow x ncol
        out = self.dc(rhs, csm, mask, self.lam)
        #print('lam1 = ', self.lam)
        # for calculating the loss, we need to return to the 2 channel real image
        # plot the final out
        utils.log_image_wandb(images = [r2c(atb)[0],r2c(rhs)[0],out[0]], name = 'atb rhs out', abs = True)
        out = c2r(out)
        return out
