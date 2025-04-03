import torch

from .utils import save_img, save_side_by_side

import numpy as np
import os
import matplotlib.pyplot as plt

def extract_AE_features(dataloader, net, savefile):
    '''Extract data features from given model'''
    
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        net.cuda()

    net.eval()
    for sample in dataloader:
        inputs = sample['image']

        if is_cuda:
            inputs = inputs.cuda()

        #outputs, _ = net.encode(inputs)
        outputs = net.encode(inputs)

        with open(savefile, 'ab') as f:
    	    np.savetxt(f, outputs.cpu().data, fmt='%f')

def visualize_AE_recon(dataloader, net, savedir):
    '''saves example reconstruction to save directory'''

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        net.cuda()

    net.eval()

    for idx, sample in enumerate(dataloader):
        # print("dict",sample)
        # print("image",sample['image'].shape)
        inputs = sample['image']
        # print("inputs",inputs.shape)

        if is_cuda:
            inputs = inputs.cuda()
        
        with torch.no_grad():
            mu, logvar = net.encode(inputs)
            z = net.reparameterize(mu, logvar)
            outputs = net.generate(z)


        print(f"Inputs - dtype: {inputs.dtype}, shape: {inputs.shape}, min: {inputs.min().item()}, max: {inputs.max().item()}, mean: {inputs.mean().item()}")
        print(f"Outputs - dtype: {outputs.dtype}, shape: {outputs.shape}, min: {outputs.min().item()}, max: {outputs.max().item()}, mean: {outputs.mean().item()}")

        save_side_by_side(inputs, outputs, savedir + "/recon_compare_%s.png" % idx)
        save_img(inputs, os.path.join(savedir, "input_%s.png" % idx))
        save_img(outputs, os.path.join(savedir, "recon_%s.png" % idx))

        if idx >= 16:
            break
