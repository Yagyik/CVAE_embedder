import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import model_dict, classifier_dict
from models.train_utils import train_classifier_model, evaluate_classifier_model, save_checkpoint, setup_optimizer
from datasets import dataset_dict
from utils import setup_logger, read_and_combine_dataframes, format_fix_dataframe, split_and_save_dataframe

import numpy as np
import argparse
import os
import sys

# parse arguments
def setup_args():

    options = argparse.ArgumentParser()
    
    options.add_argument('--dataframe-list', dest='dataframe_list',action="store", default="dataframe_list.txt")

    # options.add_argument('--datadir', action="store", default="datapaths.txt")
    # options.add_argument('--train-metafile', action="store", default="splits/train_total.csv")
    # options.add_argument('--val-metafile', action="store", default="splits/val_total.csv")
    options.add_argument('--train-metafile', dest='train_metafile',action="store", default="train_dataset_std_full.csv")
    options.add_argument('--val-metafile', dest = 'val_metafile', action="store", default="test_dataset_std_full.csv")
    options.add_argument('--save-dir', dest = 'save_dir',action="store", default='results/VAE/')
    options.add_argument('--save-freq', action="store", default=50, type=int)
    options.add_argument('--seed', action="store", default=42, type=int)

    # model parameters
    options.add_argument('--optimizer', action="store", dest="optimizer", default='adam')
    options.add_argument('--latent-dims', action="store", dest="latent_dims", default=128, type = int)
    options.add_argument('--model-type', action="store", dest="model_type", default='CVAE')
    options.add_argument('--classifier-type',dest="classifier_type",action="store",default='FC_Classifier')

    # training parameters
    options.add_argument('--dataset-type', action="store", default='default')
    options.add_argument('--batch-size', action="store", dest="batch_size", default=256, type=int)
    options.add_argument('--num-workers', action="store", dest="num_workers", default=8, type=int)
    options.add_argument('--learning-rate', action="store", dest="learning_rate", default=1e-4, type=float)
    options.add_argument('--max-epochs', action="store", dest="max_epochs", default=500, type=int)
    options.add_argument('--weight-decay', action="store", dest="weight_decay", default=1e-5, type=float)
    options.add_argument('--lamb', action="store", dest="lamb", default=0.00001, type = float)
    options.add_argument('--lamb2', action="store", dest="lamb2", default=10, type = float)
    options.add_argument('--conditional', action="store_true",default=False)

    # debugging mode
    options.add_argument('--debug-mode', action="store_true", default=False)

    return options.parse_args()

def run_training(args, logger):
    
    os.makedirs(os.path.join(args.save_dir, "models"), exist_ok=True)

    # seed run
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Read and combine dataframes
    combined_df = read_and_combine_dataframes(args.dataframe_list)
    split_and_save_dataframe(combined_df, train_file=args.train_metafile, test_file=args.val_metafile)

    # load data
    trainset = dataset_dict[args.dataset_type](metafile=args.train_metafile, mode='train')
    testset = dataset_dict[args.dataset_type](metafile=args.val_metafile, mode='val')

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # load model
    print(model_dict,args.model_type,model_dict[args.model_type],args.conditional)

    if args.conditional:
        net = model_dict[args.model_type](latent_variable_size=args.latent_dims,
                                      lamb=args.lamb,
                                      lamb2=args.lamb2,
                                      batchnorm=False)
    else:
        net = model_dict[args.model_type](latent_variable_size=args.latent_dims,
                                      lamb=args.lamb,
                                      batchnorm=False)
    if args.conditional:
        netCondClf = classifier_dict[args.classifier_type](nz=args.latent_dims,
                                                           n_hidden=args.latent_dims,
                                                           n_out=3)
    else:
        netCondClf = None

    # if args.pretrained_file is not None:
    #     net.load_state_dict(torch.load(args.pretrained_file))
    #     print("Pre-trained model loaded")
    #     sys.stdout.flush()

    # CE_weights = torch.FloatTensor([4.5, 0.5]) 
    CE_weights = torch.FloatTensor([0.33, 0.33, 0.33])
    logger.info(net)

    if torch.cuda.is_available():
        print("using cuda")
        net = net.cuda()
        CE_weights = CE_weights.cuda()
        if args.conditional:
            print("using conditional")
            netCondClf = netCondClf.cuda()
    # exit(0)

    # setup optimizer and scheduler
    # optimizer = setup_optimizer(name=args.optimizer, param_list=[{'params': net.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay}])

    ### if no good justification for weight decay
    if args.conditional:
        optimizer = setup_optimizer(name=args.optimizer,
                                    param_list=[{'params':list(net.parameters())+list(netCondClf.parameters()),
                                    'lr':args.learning_rate,
                                    'weight_decay':args.weight_decay}])
    else:
        optimizer = setup_optimizer(name=args.optimizer,
                                     param_list=[{'params': net.parameters(),
                                                   'lr': args.learning_rate,
                                                   'weight_decay': args.weight_decay}])
        
    # if args.conditional:
    #     optimizer = optim.Adam(list(model.parameters())+list(netCondClf.parameters()), lr = args.lr)
    # else:
    #     optimizer = optim.Adam([{'params': model.parameters()}], lr = args.lr)

    # main training loop
    best_loss = np.inf

    for epoch in range(args.max_epochs):

        
        logger.info("Epoch %s:" % epoch)

        # train_summary = train_model(trainloader=trainloader, model=net, optimizer=optimizer, single_batch=args.debug_mode)
        # logger.info("Training summary: %s" % train_summary)

        # test_summary = evaluate_model(testloader=testloader, model=net, single_batch=args.debug_mode)
        # logger.info("Evaluation summary: %s" % test_summary)
        if args.conditional:
            train_summary = train_classifier_model(trainloader=trainloader,
                                                    model=net,classifier=netCondClf,
                                                    CE_weights=CE_weights,conditional=True,
                                                    optimizer=optimizer, single_batch=args.debug_mode)
            logger.info("Training summary: %s" % train_summary)

            test_summary = evaluate_classifier_model(testloader=testloader,
                                                  model=net,classifier=netCondClf,
                                                  CE_weights=CE_weights,conditional=True,
                                                    single_batch=args.debug_mode)
            logger.info("Evaluation summary: %s" % test_summary)
            current_state = {'epoch': epoch,
                          'state_dict': net.cpu().state_dict(),
                          'classifier_state_dict': netCondClf.cpu().state_dict(),
                            'optimizer': optimizer.state_dict()}


        else:
            train_summary = train_classifier_model(trainloader=trainloader,
                                                    model=net,classifier=None,CE_weights=None,
                                                    optimizer=optimizer, single_batch=args.debug_mode)
            logger.info("Training summary: %s" % train_summary)
            test_summary = evaluate_classifier_model(testloader=testloader,
                                                  model=net,classifier=None,CE_weights=None,
                                                    single_batch=args.debug_mode)

            logger.info("Evaluation summary: %s" % test_summary)
            current_state = {'epoch': epoch,
                          'state_dict': net.cpu().state_dict(),
                          'classifier_state_dict': None,
                            'optimizer': optimizer.state_dict()}

        

        

        if test_summary['test_loss'] < best_loss:
            best_loss = test_summary['test_loss']
            current_state['best_loss'] = best_loss
            save_checkpoint(current_state=current_state, filename=os.path.join(args.save_dir,"models/best.pth"))

        logger.info("Best loss: %s" % best_loss)
        
        if epoch % args.save_freq == 0:
            # print(test_summary)
            loss_details = ", ".join([f"{name}: {test_summary[name]}" for name in test_summary.keys()])
            print(f"EPOCH: {epoch}, loss: {test_summary['test_loss']}, best loss: {best_loss}, {loss_details}")
            save_checkpoint(current_state=current_state, filename=os.path.join(args.save_dir, "models/epoch_%s.pth" % epoch))

        save_checkpoint(current_state=current_state, filename=os.path.join(args.save_dir, "models/last.pth"))

        if torch.cuda.is_available():
            net.cuda()

if __name__ == "__main__":
    args = setup_args()

    # df_files = read_dataframe_list(args.dataframe_list)
    print(args.conditional,"doing conditional")
    metadata_df = read_and_combine_dataframes(args.dataframe_list)
    # metadata_df = format_fix_dataframe(metadata_df)
    print(metadata_df)
    split_and_save_dataframe(metadata_df,
                              train_file=args.train_metafile,
                                test_file=args.val_metafile)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(name='training_log', save_dir=args.save_dir)
    logger.info(" ".join(sys.argv))
    logger.info(args)
    run_training(args, logger)
