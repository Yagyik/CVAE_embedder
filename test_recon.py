import torch
from torch.utils.data import DataLoader

from features.AE_features import visualize_AE_recon #, visualize_PCA_recon


import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt

from models import model_dict, classifier_dict
from datasets import dataset_dict
from utils import setup_logger,read_and_combine_dataframes, split_and_save_dataframe
# parse arguments
def setup_args():

    options = argparse.ArgumentParser()
    
    options.add_argument('--dataframe-list', dest='dataframe_list',action="store", default="dataframe_list.txt")  
    options.add_argument('--metafile', action="store", default = "train_dataset_std_full.csv")

    options.add_argument('--pretrained_file', type=str, default='results/CVAE/models/best.pth', help='Path to the pretrained VAE model')
    # options.add_argument('--classifier_file', type=str, default='results/Classifier/models/best.pth', help='Path to the pretrained classifier model')
    # model parameters
    options.add_argument('--optimizer', action="store", dest="optimizer", default='adam')
    options.add_argument('--latent-dims', action="store", dest="latent_dims", default=128, type = int)
    options.add_argument('--model-type', action="store", dest="model_type", default='CVAE')
    options.add_argument('--classifier-type',dest="classifier_type",action="store",default='FC_Classifier')

    options.add_argument('--lamb', action="store", dest="lamb", default=0.00001, type = float)
    options.add_argument('--lamb2', action="store", dest="lamb2", default=0.1, type = float)
    options.add_argument('--conditional', action="store_true",default=False)

    # options.add_argument('--metafile', type=str, default='train_dataset_std_full.csv', help='Path to the metafile')
    options.add_argument('--dataset-type', action="store", default='default')
    options.add_argument('--train-metafile', dest='train_metafile',action="store", default="train_dataset_cvae.csv")
    options.add_argument('--val-metafile', dest = 'val_metafile', action="store", default="test_dataset_cvae.csv")
    
    options.add_argument('--save-dir', action="store", dest="save_dir", default="results/recon/long/")
    options.add_argument('--batch-size', action="store", dest="batch_size", default=1, type = int)

    options.add_argument('--ae-features', action="store_false")
    options.add_argument('--pca-features', action="store_true")

    return options.parse_args()

def main(args, logger):
    
    # Load the combined dataframe
    combined_df = read_and_combine_dataframes(args.dataframe_list)

    combined_df.to_csv('combined_recon_df.csv', index=False)

    # Load the best VAE 
    if args.model_type == 'CVAE':
        net = model_dict[args.model_type](latent_variable_size=args.latent_dims,
                                      lamb=args.lamb,
                                      lamb2=args.lamb2,
                                      batchnorm=False)
    else:
        net = model_dict[args.model_type](latent_variable_size=args.latent_dims,
                                      lamb=args.lamb,
                                      batchnorm=False)
    
    net.load_state_dict(torch.load(args.pretrained_file)['state_dict'])
    # net.eval()

    # Load the classifier model
    if args.conditional:
        classifier = classifier_dict[args.classifier_type](nz=args.latent_dims,n_out=3)
        classifier.load_state_dict(torch.load(args.pretrained_file)['classifier_state_dict'])
    # classifier.eval()

    # Create a dataset and dataloader
    dataset = dataset_dict[args.dataset_type](metafile='combined_recon_df.csv', mode='val')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    logger.info(net)

    # extract AE features
    print("using AE features, PCA features??",args.ae_features,args.pca_features)
    if args.ae_features:
        os.makedirs(os.path.join(args.save_dir, 'crops'), exist_ok=True)
        visualize_AE_recon(dataloader=dataloader, net=net, savedir=os.path.join(args.save_dir, 'crops'))

    # extract PCA features
    # if args.pca_features:
    #     os.makedirs(os.path.join(args.save_dir, 'PCA_recon'), exist_ok=True)
    #     visualize_PCA_recon(dataloader=dataloader, dataloader_bs1=dataloader_bs1, savedir=os.path.join(args.save_dir, 'PCA_recon'),
    #                         n_components=args.latent_dims)
    

if __name__ == "__main__":
    args = setup_args()
    metadata_df = read_and_combine_dataframes(args.dataframe_list)
    # metadata_df = format_fix_dataframe(metadata_df)
    print(metadata_df)
    split_and_save_dataframe(metadata_df,
                              train_file=args.train_metafile,
                                test_file=args.val_metafile)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(name='recon_log', save_dir=args.save_dir)
    logger.info(" ".join(sys.argv))
    logger.info(args)
    main(args, logger)
