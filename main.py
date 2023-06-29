import torch
from utils import MyDataset
from model import MoDL
from tqdm import tqdm
import numpy as np
# c2r=lambda x:tf.stack([tf.real(x),tf.imag(x)],axis=-1)

# self implementation of MoDL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 100
batch_size = 30
val_batch = 10
sigma = 0.01
model = MoDL()
model.to(device)

# define loss
loss_fn = torch.nn.MSELoss(reduction='sum')
# important: the lambda should also be a parameter to train.
params = list(model.dw.parameters()) + [model.lam] + list(model.dc.parameters())

# define optimizer
optimizer = torch.optim.Adam(params, lr=0.001)
#optimizer = torch.optim.AdamW(model.dw.parameters(),lr=0.001)

# change the path to your dataset.
training_set = MyDataset(filename = '/media/ssd/fanwen/dataset.hdf5', training = 'training', sigma = sigma)
validation_set = MyDataset(filename = '/media/ssd/fanwen/dataset.hdf5', training = 'validation', sigma = sigma)
training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=val_batch, shuffle=True, num_workers=4)
# change the savedir for your model.
save_dir = '/media/ssd/fanwen/MoDL/'
TRNLOSS = []
VALLOSS = []
# here to check the data_loader

for epoch in range(epochs):
    model.train()
    trnloss = []
    for batch, data in enumerate(training_generator):
        # org,atb,csm,mask = data
        # batchsize, 256, 232; batchsize, 256, 232, 2; batchsize, 12, 256, 232; batchsize, 256, 232
        org, atb, csm, mask = data
        # change the varible to cuda
        org, atb, csm, mask = org.to(device), atb.to(device), csm.to(device), mask.to(device)
        # permute batch* 256* 232* 2 to batch* 2* 256* 232
        out = model.forward(atb, csm, mask)

        loss = loss_fn(model.forward(atb, csm, mask), org)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # append the loss
        trnloss.append(loss.item())
    LOSS = sum(trnloss)/training_set.__len__()
    TRNLOSS.append(LOSS)
    print('epoch: ', epoch, 'trnloss: ', LOSS)
    if epoch % 1 == 0:
        valloss = []
        for batch, data in enumerate(validation_generator):
            with torch.no_grad():
                org, atb, csm, mask = data
                org, atb, csm, mask = org.to(device), atb.to(device), csm.to(device), mask.to(device)
                loss = loss_fn(model.forward(atb, csm, mask), org)
                valloss.append(loss.item())
        LOSS = sum(valloss)/validation_set.__len__()
        VALLOSS.append(LOSS)
        print('------------epoch: ', epoch, 'valloss: ', LOSS)
    # save the training and validation loss to npy
    np.save(save_dir + 'loss.npy', TRNLOSS, VALLOSS)
    # save every epoch
    modelname = 'model' + str(epoch).zfill(3) + '.pth'
    save_path = save_dir + modelname
    torch.save(model.state_dict(), save_path)

    