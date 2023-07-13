import torch
from utils import MyDataset, log_image_wandb
from model import MoDL
from tqdm import tqdm
import numpy as np
import wandb
from config import parser
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
# c2r=lambda x:tf.stack([tf.real(x),tf.imag(x)],axis=-1)
r2c = lambda x: torch.complex(x[:,0,...], x[:,1,...])

# self implementation of MoDL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()

model = MoDL()
model.to(device)

# Logger to wandb
logger = wandb.init(project="MoDL", config=args, resume="allow", entity="fanwen-wang")

# define loss
loss_fn = torch.nn.MSELoss(reduction='sum')
# important: the lambda should also be a parameter to train. dc module does not include trainable parameters.
params = list(model.dw.parameters()) + [model.lam]

# define optimizer
optimizer = torch.optim.Adam(params, lr=0.001)
#optimizer = torch.optim.AdamW(model.dw.parameters(),lr=0.001)

# change the path to your dataset.
training_set = MyDataset(filename = '/media/ssd/fanwen/dataset.hdf5', training = 'training', sigma = args.sigma)
validation_set = MyDataset(filename = '/media/ssd/fanwen/dataset.hdf5', training = 'validation', sigma = args.sigma)
training_generator = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=args.val_batch, shuffle=True, num_workers=4)
# change the savedir for your model.
save_dir = '/media/ssd/fanwen/MoDL/'
TRNLOSS = []
VALLOSS = []
# here to check the data_loader

model.train()

# print and log number of parameters
num_params = sum(p.numel() for p in params if p.requires_grad)
print("Number of parameters:", num_params)
logger.log({"num_params": num_params})

for epoch in range(args.epoch):
    trnloss = []
    trnpsnr = []
    trnssim = []
    for batch, data in enumerate(training_generator):
        # org,atb,csm,mask = data
        # batchsize, 256, 232; batchsize, 256, 232, 2; batchsize, 12, 256, 232; batchsize, 256, 232
        org, atb, csm, mask = data
        # change the varible to cuda
        org, atb, csm, mask = org.to(device), atb.to(device), csm.to(device), mask.to(device)
        # permute batch* 256* 232* 2 to batch* 2* 256* 232
        out = model.forward(atb, csm, mask,logger)

        loss = loss_fn(out, org)
        # get PSNR and SSIm for logger.
        psnr = peak_signal_noise_ratio(org, out)
        ssim = structural_similarity_index_measure(org, out)
        
        # here to draw the image to wandb
        log_image_wandb(images = [r2c(atb)[0],r2c(out)[0], r2c(org)[0]], name = 'trn atb rec org', abs = True)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # append the loss
        trnloss.append(loss.item())
        trnpsnr.append(psnr.item())
        trnssim.append(ssim.item())
    LOSS = sum(trnloss)/training_set.__len__()
    PSNR = sum(trnpsnr)/training_set.__len__()
    SSIM = sum(trnssim)/training_set.__len__()
    logger.log({"trnloss": LOSS, "trn_psnr": PSNR, "trnssim": SSIM})
    TRNLOSS.append(LOSS)
    print('epoch: ', epoch, 'trnloss: ', LOSS)
    if epoch % 1 == 0:
        valloss = []
        valpsnr = []
        valssim = []
        for batch, data in enumerate(validation_generator):
            with torch.no_grad():
                org, atb, csm, mask = data
                org, atb, csm, mask = org.to(device), atb.to(device), csm.to(device), mask.to(device)
                out = model.forward(atb, csm, mask, logger)
                loss = loss_fn(out, org)
                # here to draw the image to wandb
                log_image_wandb(images = [r2c(atb)[0],r2c(out)[0], r2c(org)[0]], name = 'val atb rec org', abs=True)
                # get PSNR and SSIm for logger.
                psnr = peak_signal_noise_ratio(org, out)
                ssim = structural_similarity_index_measure(org, out)
                valloss.append(loss.item())
                valpsnr.append(psnr.item())
                valssim.append(ssim.item())

        LOSS = sum(valloss)/validation_set.__len__()
        PSNR = sum(valpsnr)/validation_set.__len__()
        SSIM = sum(valssim)/validation_set.__len__()
        logger.log({"valloss": LOSS, "val_psnr": PSNR, "valssim": SSIM})
        VALLOSS.append(LOSS)
        print('------------epoch: ', epoch, 'valloss: ', LOSS)
    # save the training and validation loss to npy
    np.save(save_dir + 'loss.npy', TRNLOSS, VALLOSS)
    # save every epoch
    modelname = 'model' + str(epoch).zfill(3) + '.pth'
    save_path = save_dir + modelname
    torch.save(model.state_dict(), save_path)

    