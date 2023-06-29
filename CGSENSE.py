import torch
import numpy as np
from os.path import expanduser
from utils import getData
import matplotlib.pyplot as plt

home = expanduser("~")
epsilon = 1e-5
TFeps = torch.tensor(1e-5, dtype=torch.float32)

# function c2r contatenate complex input as new axis two two real inputs
c2r = lambda x: torch.stack([x.real, x.imag], axis=-1)
#r2c takes the last dimension of real input and converts to complex
r2c = lambda x: torch.complex(x[..., 0], x[..., 1])

class Aclass:
    '''
    Implementation of the A operator in the MoDL paper
    Data consistency operator using CG-SENSE
    '''
    #def __init__(self, csm, mask, lam):
    def __init__(self, csm, mask):
        # get the size of mask
        s = mask.shape
        self.nrow, self.ncol = s[0], s[1]
        self.csm = csm
        self.mask = mask
        #self.lam = lam
        self.SF = torch.complex(torch.sqrt(torch.tensor(self.nrow*self.ncol, dtype=torch.float32)), torch.tensor(0.0, dtype=torch.float32))
    def myAtA(self,img):
        # here the img should be multi-channel complex image with aliasing
        coilImage = self.csm * img
        kspace = torch.fft.fft2(coilImage)/self.SF
        temp = kspace*self.mask
        coilImgs = torch.fft.ifft2(temp)*self.SF
        coilComb = torch.sum(coilImgs*self.csm.conj(), axis=0)
        #coilComb = coilComb + self.lam*img
        return coilComb
    
def myCG(A, rhs):
    '''
    Complex conjugate gradient on complex data
    '''
    rhs = r2c(rhs)
    def body(i, rTr, x, r, p):
        Ap = A.myAtA(p)
        alpha = rTr / (torch.sum(p.conj()*Ap)).to(torch.float32)
        alpha = torch.complex(alpha, torch.tensor(0.))
        x = x + alpha *p 
        r = r - alpha * Ap
        rTrNew = (torch.sum(r.conj()*r)).to(torch.float32)
        beta = rTrNew / rTr
        beta = torch.complex(beta, torch.tensor(0.))
        p = r + beta * p
        return i+1, rTrNew, x, r, p
    
    # the initial values of the loop variables
    x = torch.zeros_like(rhs)
    i,r,p = 0, rhs, rhs
    # This should yield cast the complex to real, but no worries, 
    rTr = torch.sum(r.conj()*r).to(torch.float32)
    loopVar = i, rTr, x, r, p
    
    #def cond(loopVar):
    # Define your termination condition
    #    cond = torch.logical_and(torch.less(i, 10), rTr>1e-10) 
    #    return cond

    while i< 10 and rTr>1e-10:
        i,rTr,x,r,p = body(i, rTr, x, r, p)
    
    out = x
    #return c2r(x)
    return out

def dc(rhs,csm,mask):
    '''
    This function is called to create testing model. It apply CG on each image
    in the batch.
    '''
    #l2 = torch.complex(l,0.)
    def fn(tmp):
        c,m,r = tmp
        Aobj = Aclass(c,m)
        y = myCG(Aobj,r)
        return y
    inp = (csm,mask,rhs)
    # original tensorflow is parallel for loop
    rec = fn(inp)
    return rec

def makeModel(atb, csm, mask, training):
    '''
    This is the main function that creates the model
    '''
    rhs = atb
    out = dc(rhs, csm, mask)
    return out

# here to run the main
if __name__ == '__main__':
    org,atb,csm,mask = getData(trnTst='training',num=100,sigma=.01)
    # if you get the training data, you should convert the real to complex first
    # change all the data to torch tensor
    #org = torch.tensor(org[0], dtype=torch.float32)
    atb = torch.tensor(atb[1], dtype=torch.float32)

    csm = torch.tensor(csm[1], dtype=torch.complex64)
    mask = torch.tensor(mask[1], dtype=torch.complex64)
    out = makeModel(atb, csm, mask, True)
    out2 = out.numpy()
    plt.imshow(np.abs(out2), cmap='gray')
    #Aobj = Aclass(csm,mask)
    #atb = r2c(atb)
    #Ap = Aobj.myAtA(atb) # This is the coilcombined image!
