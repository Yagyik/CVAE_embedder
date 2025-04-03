import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os

from models import model_dict, classifier_dict
from datasets import dataset_dict
from utils import read_and_combine_dataframes, split_and_save_dataframe
# from models.train_utils import mse_loss, kl_divergence

def mse_loss(outputs, targets):
    return torch.nn.MSELoss()(outputs, targets)

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def setup_args():
    parser = argparse.ArgumentParser(description='Generate feature vectors and predicted labels')
    parser.add_argument('--dataframe-list', type=str, default='dataframe_list.txt', help='Path to the dataframe list')
    parser.add_argument('--pretrained_file', type=str, default='results/CVAE/models/best.pth', help='Path to the pretrained VAE model')
    # parser.add_argument('--classifier_file', type=str, default='results/Classifier/models/best.pth', help='Path to the pretrained classifier model')
    # model parameters
    parser.add_argument('--optimizer', action="store", dest="optimizer", default='adam')
    parser.add_argument('--latent-dims', action="store", dest="latent_dims", default=128, type = int)
    parser.add_argument('--model-type', action="store", dest="model_type", default='CVAE')
    parser.add_argument('--classifier-type',dest="classifier_type",action="store",default='FC_Classifier')

    parser.add_argument('--lamb', action="store", dest="lamb", default=0.00001, type = float)
    parser.add_argument('--lamb2', action="store", dest="lamb2", default=0.1, type = float)
    parser.add_argument('--conditional', action="store_true",default=True)

    # parser.add_argument('--metafile', type=str, default='train_dataset_std_full.csv', help='Path to the metafile')
    parser.add_argument('--dataset-type', action="store", default='default')
    parser.add_argument('--train-metafile', dest='train_metafile',action="store", default="train_dataset_cvae.csv")
    parser.add_argument('--val-metafile', dest = 'val_metafile', action="store", default="test_dataset_cvae.csv")


    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--output_file', type=str, default='combined_with_features.csv', help='Output file for the new dataframe')
    return parser.parse_args()

def main():
    args = setup_args()

    # Load the combined dataframe
    combined_df = read_and_combine_dataframes(args.dataframe_list)

    combined_df.to_csv('combined_recon_df.csv', index=False)

    # Load the best VAE model
    net = model_dict[args.model_type](latent_variable_size=args.latent_dims,
                                      lamb=args.lamb,
                                      lamb2=args.lamb2,
                                      batchnorm=False)
    
    net.load_state_dict(torch.load(args.pretrained_file)['state_dict'])
    # net.eval()

    # Load the classifier model
    if args.conditional:
        classifier = classifier_dict[args.classifier_type](nz=args.latent_dims,
                                                           n_hidden = args.latent_dims,
                                                           n_out=3)
    classifier.load_state_dict(torch.load(args.pretrained_file)['classifier_state_dict'])
    # classifier.eval()

    # Create a dataset and dataloader
    dataset = dataset_dict[args.dataset_type](metafile='combined_recon_df.csv', mode='val')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Initialize lists to store feature vectors and predicted class labels
    feature_vectors = []
    predicted_labels = []

    # Initialize list to store errors
    errors = []

    # Iterate through each row of the dataframe
    for i, data in enumerate(dataloader):
        inputs = data['image']
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            net = net.cuda()
            classifier = classifier.cuda()

        with torch.no_grad():
            recon_inputs, latents, mu, logvar = net(inputs)
            feature_vector = latents.cpu().numpy().flatten()
            feature_vectors.append(feature_vector)
            outputs = classifier(latents)
            print(outputs)

            # Calculate predicted label
            # criterion = torch.nn.CrossEntropyLoss()
            predicted_label = torch.argmax(outputs, dim=1).cpu().item()
            print(predicted_label)
            predicted_labels.append(predicted_label)

            # Calculate MSE and KL divergence
            mse = mse_loss(recon_inputs, inputs).cpu().item()
            kl_div = kl_divergence(mu, logvar).cpu().item()
            errors.append([mse, kl_div])

    # Convert lists to numpy arrays
    feature_vectors = np.array(feature_vectors)
    predicted_labels = np.array(predicted_labels)
    errors = np.array(errors)

    # Add feature vectors, predicted labels, and errors to the dataframe
    for i in range(args.latent_dims):
        combined_df[f'feature_{i}'] = feature_vectors[:, i]
    combined_df['predicted_label'] = predicted_labels
    combined_df['mse'] = errors[:, 0]
    combined_df['kl_divergence'] = errors[:, 1]

    # Save the new dataframe
    combined_df.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    main()