# this is to test the model
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from utils import MyDataset, myPSNR
from model import MoDL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
r2c = lambda x: torch.complex(x[:,0,...], x[:,1,...])


sigma = 0.01
batch_size = 1

# change the path to the dataset.
testing_set =MyDataset(filename = '/media/ssd/fanwen/dataset.hdf5', training = 'testing', testslice = 100,sigma = sigma)
testing_generator = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True, num_workers=4)


model = MoDL()
model.to(device)
# change the path to your model 
model.load_state_dict(torch.load('/media/ssd/fanwen/MoDL/model099.pth'))
model.eval()

with torch.no_grad():
    for batch, data in enumerate(testing_generator):
        org, atb, csm, mask = data
        org, atb, csm, mask = org.to(device), atb.to(device), csm.to(device), mask.to(device)
        # output atb and org are 2-chaneel
        out = model.forward(atb, csm, mask)
        out = r2c(out)
        atb = r2c(atb) 

# detach the tensor 
out = out.detach().cpu().numpy()
mask = mask.detach().cpu().numpy()
atb = atb.detach().cpu().numpy()
org = org.detach().cpu().numpy()
#%% Display the output images

plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, .8))
plt.clf()
plt.subplot(141)
plot(np.fft.fftshift(mask[0]))
plt.axis('off')
plt.title('Mask')
plt.subplot(142)
plot(np.abs(org[0]))
plt.axis('off')
plt.title('Original')
plt.subplot(143)
plot(np.abs(atb[0]))
plt.title('Input, PSNR='+str(myPSNR(np.abs(atb[0]),np.abs(org[0])).round(2))+' dB' )
plt.axis('off')
plt.subplot(144)
plot(np.abs(out[0]))
plt.title('Output, PSNR='+ str(myPSNR(np.abs(out[0]), np.abs(org[0])).round(2)) +' dB')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
plt.show()