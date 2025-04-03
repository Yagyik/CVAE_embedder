import pickle 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


import skimage.io as io
import pandas as pd
import numpy as np

import os


class CellImageDataset(Dataset):

    def __init__(self,metafile,mode='train',**kwargs):
        self.file_metadata = pd.read_csv(metafile)
        self.mode = mode

        if mode == 'train':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(90),
            transforms.ToTensor()
            ])

        elif mode == 'val':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
            ])

        else:
            raise KeyError("dataset mode must be one of ['train', 'val'], not %s" % mode)

    def __len__(self):
        return len(self.file_metadata)
   
    def load_image(self, fname):
        img = io.imread(fname, plugin='tifffile')
        img = img.astype(np.float32)
        # print("img min",np.min(img),"img max",np.max(img),"img mean",np.mean(img),img.shape)
        # img /= np.max(img)
        # plt.imshow(img, cmap='viridis')
        # plt.show()
        # print("old img min",np.min(img),"img max",np.max(img),"img mean",np.mean(img),img.shape)
        # p99 = np.percentile(img, 99)
        # img = np.clip(img, 0, p99)
        # print("new img min",np.min(img),"img max",np.max(img),"img mean",np.mean(img),img.shape)

        img = torch.from_numpy(img).view(1, 64, 64).type(torch.float32)
        # print("Tensor min:", torch.min(img).item(), "Tensor max:", torch.max(img).item(), "Tensor mean:", torch.mean(img).item())
        return img



    # def __getitem__(self, idx):
    #     sample = self.file_metadata.iloc[idx]
    #     # fileprefix = str(int(sample['fileprefix']))
    #     fileprefix = f"{int(sample['fileprefix']):02d}"
    #     nuc_index = int(sample['img_nuc_index'])
    #     condition_index = int(sample['condition_index'])
    #     expt_index = int(sample['expt_index'])
    #     imgdatadir = os.path.join(self.datadir,
    #                                "cond_" + str(condition_index),
    #                                 "expt_" + str(expt_index))

    #     # print(fileprefix,nuc_index,condition_index,imgdatadir)    
    #     imgfile = os.path.join(imgdatadir,fileprefix + "_nuc_ind" + str(nuc_index) + ".tiff")
    #     if not os.path.exists(imgfile):
    #         raise FileNotFoundError(f"Image file {imgfile} not found.")
    #     img = self.load_image(imgfile)


    #     img = self.transform(img)

    #     # print("pre to dict",sample)
    #     # sample_dict  = sample.to_dict()
    #     # print("to dict",sample_dict)
    #     sample_dict = {}
    #     sample_dict['label'] = condition_index
    #     sample_dict['expt_label'] = expt_index
    #     sample_dict['nuc_label'] = nuc_index
    #     sample_dict.update({'image': img})
    #     sample_dict.update({'key': imgfile})
    #     # print("sample dict",sample_dict)

    #     return sample_dict


    def __getitem__(self, idx):
        sample = self.file_metadata.iloc[idx]
        # fileprefix = str(int(sample['fileprefix']))

        nuc_index = int(sample['cell_label'])
        mainpath_parts = sample['mainpath'].split('/')
        datadir = '/'.join(mainpath_parts[:-1])
        # datadir = datadir.replace("/goswam_y/", "/yagyik/")

        batch = sample['batch']
        condition = sample['condition']
        subbatch = sample['subbatch']
        # print(datadir,batch,condition,subbatch)
        imgdatadir = datadir
        for key in ['batch', 'condition', 'subbatch']:
            # print(key,sample[key],imgdatadir)
            if not pd.isna(sample[key]):
                
                imgdatadir = os.path.join(imgdatadir, sample[key])
        # print(nuc_index,condition,imgdatadir)    
        imgfile = os.path.join(imgdatadir,sample['file'])
        if not os.path.exists(imgfile):
            raise FileNotFoundError(f"Image file {imgfile} not found.")
        img = self.load_image(imgfile)


        img = self.transform(img)
        # print("Tensor min:", torch.min(img).item(), "Tensor max:", torch.max(img).item(), "Tensor mean:", torch.mean(img).item())

        # print("pre to dict",sample)
        # sample_dict  = sample.to_dict()
        # print("to dict",sample_dict)
        sample_dict = {}
        sample_dict['label'] = condition
        sample_dict['label_int'] = float(sample['condition_int'])
        sample_dict['batch_label'] = batch
        sample_dict['subbatch_label'] = subbatch
        sample_dict['nuc_label'] = nuc_index
        sample_dict.update({'image': img})
        sample_dict.update({'key': imgfile})
        # print("sample dict",sample_dict)

        return sample_dict