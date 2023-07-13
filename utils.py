"""
This file contains some supporting functions used during training and testing.
"""
import time
import numpy as np
import h5py as h5
import torch
import wandb

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filename, training, testslice = 100, sigma=.01):
        'initialization'
        self.filename = filename
        self.training = training
        self.testslice = testslice
        self.sigma = sigma
        with h5.File(filename) as f:
            if self.training =='training':
                self.org,self.csm,self.mask=f['trnOrg'][10:15],f['trnCsm'][10:15],f['trnMask'][10:15]
            elif self.training =='validation':
                self.org,self.csm,self.mask=f['tstOrg'][10:15],f['tstCsm'][10:15],f['tstMask'][10:15]
            else:
                org,csm,mask=f['tstOrg'][self.testslice],f['tstCsm'][self.testslice],f['tstMask'][self.testslice]
                na=np.newaxis
                self.org, self.csm, self.mask=org[na],csm[na],mask[na]
        print('Successfully read the data from file!')
        print('Now doing undersampling....')
        tic()
        self.atb = generateUndersampled(self.org, self.csm, self.mask, self.sigma)
        toc()
        print('Successfully undersampled data!')
        if self.training == 'training':
            self.atb = c2r(self.atb)
            self.org = c2r(self.org)
        elif self.training == 'validation':
            self.atb = c2r(self.atb)
            self.org = c2r(self.org)
        else:
            self.atb = c2r(self.atb)

        
    def __getitem__(self, index):
        return(self.org[index], self.atb[index], self.csm[index], self.mask[index])
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.org)


#%%
def div0( a, b ):
    """ This function handles division by zero """
    c=np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return c

#%% This provide functionality similar to matlab's tic() and toc()
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
#%%

def normalize01(img):
    """
    Normalize the image between o and 1
    """
    if len(img.shape)==3:
        nimg=len(img)
    else:
        nimg=1
        r,c=img.shape
        img=np.reshape(img,(nimg,r,c))
    img2=np.empty(img.shape,dtype=img.dtype)
    for i in range(nimg):
        img2[i]=div0(img[i]-img[i].min(),img[i].ptp())
        #img2[i]=(img[i]-img[i].min())/(img[i].max()-img[i].min())
    return np.squeeze(img2).astype(img.dtype)

#%%
def np_crop(data, shape=(320,320)):

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]

#%%

def myPSNR(org,recon):
    """ This function calculates PSNR between the original and
    the reconstructed     images"""
    mse=np.sum(np.square( np.abs(org-recon)))/org.size
    psnr=20*np.log10(org.max()/(np.sqrt(mse)+1e-10 ))
    return psnr

def piA(x,csm,mask,nrow,ncol,ncoil):
    """ This is a the A operator as defined in the paper"""
    ccImg=np.reshape(x,(nrow,ncol) )
    coilImages=np.tile(ccImg,[ncoil,1,1])*csm
    kspace=np.fft.fft2(coilImages)/np.sqrt(nrow * ncol)
    if len(mask.shape)==2:
        mask=np.tile(mask,(ncoil,1,1))
    res=kspace[mask!=0]
    return res

def piAt(kspaceUnder,csm,mask,nrow,ncol,ncoil):
    """ This is a the A^T operator as defined in the paper"""
    temp=np.zeros((ncoil,nrow,ncol),dtype=np.complex64)
    if len(mask.shape)==2:
        mask=np.tile(mask,(ncoil,1,1))

    temp[mask!=0]=kspaceUnder
    img=np.fft.ifft2(temp)*np.sqrt(nrow*ncol)
    # here just return the multi-coil image. 
    coilComb=np.sum(img*np.conj(csm),axis=0).astype(np.complex64)
    return coilComb


def generateUndersampled(org,csm,mask,sigma=0.):
    nSlice,ncoil,nrow,ncol=csm.shape
    atb=np.empty(org.shape,dtype=np.complex64)
    for i in range(nSlice):
        A  = lambda z: piA(z,csm[i],mask[i],nrow,ncol,ncoil)
        At = lambda z: piAt(z,csm[i],mask[i],nrow,ncol,ncoil)

        sidx=np.where(mask[i].ravel()!=0)[0]
        nSIDX=len(sidx)
        noise=np.random.randn(nSIDX*ncoil,)+1j*np.random.randn(nSIDX*ncoil,)
        noise=noise*(sigma/np.sqrt(2.))
        y=A(org[i]) + noise
        atb[i]=At(y)
    return atb


def r2c(inp):
    """  input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    if inp.dtype=='float32':
        dtype=np.complex64
    else:
        dtype=np.complex128
    out=np.zeros( inp.shape[0:2],dtype=dtype)
    out=inp[...,0]+1j*inp[...,1]
    # change the last dim to 2nd dim
    out = np.moveaxis(out, -1, 1)
    return out

def c2r(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    out=np.zeros( inp.shape+(2,),dtype=dtype)
    out[...,0]=inp.real
    out[...,1]=inp.imag
    # change the last dim to 2nd dim
    out = np.moveaxis(out, -1, 1)
    return out

def getData(trnTst='testing',num=100,sigma=.01):
    #num: set this value between 0 to 163. There are total testing 164 slices in testing data
    print('Reading the data. Please wait...')
    filename='/media/ssd/fanwen/dataset.hdf5' #set the correct path here
    #filename='/Users/haggarwal/datasets/piData/dataset.hdf5'

    tic()
    with h5.File(filename) as f:
        if trnTst=='training':
            org,csm,mask=f['trnOrg'][:5],f['trnCsm'][:5],f['trnMask'][:5]
        else:
            org,csm,mask=f['tstOrg'][num],f['tstCsm'][num],f['tstMask'][num]
            na=np.newaxis
            org,csm,mask=org[na],csm[na],mask[na]
    toc()
    print('Successfully read the data from file!')
    print('Now doing undersampling....')
    tic()
    atb=generateUndersampled(org,csm,mask,sigma)
    toc()
    print('Successfully undersampled data!')
    if trnTst=='testing':
        atb=c2r(atb)
    return org,atb,csm,mask

def log_image_wandb(
    images: list[torch.Tensor],
    name: str,
    abs: bool,
):
    '''
    compat the 2D tensor image to a list and create a compacted image with the list.
    '''
    height, width = images[0].shape
    new_image = torch.zeros((height, width * len(images)))
    for i, image in enumerate(images):
        if abs:
            new_image[: , i * width : (i + 1) * width] = torch.abs(image)
        else:
            new_image[: , i * width : (i + 1) * width] = image

    wandb.log({name: [wandb.Image(new_image, caption=name)]}, commit=True)