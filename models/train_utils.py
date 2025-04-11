import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchnet.meter.meter import Meter

import numpy as np

class AverageMeter(Meter):
    def __init__(self):
        super(AverageMeter, self).__init__()
        self.reset()

    def add(self, value, n):
        self.sum += value * n
        if n <= 0:
            raise ValueError("Cannot use a non-positive weight for the running stat.")
        elif self.n == 0:
            self.mean = 0.0 + value  # This is to force a copy in torch/numpy
            self.mean_old = self.mean
        else:
            self.mean = self.mean_old + n * (value - self.mean_old) / float(self.n + n)
            self.mean_old = self.mean

        self.n += n

    def value(self):
        return self.mean

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.mean = np.nan
        self.mean_old = 0.0

def compute_loss(outputs, targets, mu, logvar,lamb=0.0000001, loss_trackers=None):
    # Reconstruction loss
    # recon_loss = F.mse_loss(outputs, targets, reduction='sum')
    # MSE = nn.MSELoss()
    # recon_loss = MSE(outputs, targets)
    # print("Outputs min:", torch.min(outputs).item(), "Outputs max:", torch.max(outputs).item())
    # print("Targets min:", torch.min(targets).item(), "Targets max:", torch.max(targets).item())
    # print("Outputs shape:", outputs.shape)
    # print("Targets shape:", targets.shape)

    # Before computing loss
    if torch.isnan(outputs).any():
        print("WARNING: NaN detected in outputs")
        outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        
    # recon_loss = nn.functional.binary_cross_entropy(outputs, targets, reduction='sum')
    # recon_loss = nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction='sum') ### use when decoder does not have a final sigmoid activation
    recon_loss = nn.functional.mse_loss(outputs, targets, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    # Total loss
    loss = recon_loss + lamb * kl_loss

    # Update loss trackers if provided
    if loss_trackers:
        loss_trackers['recon_loss'].add(recon_loss.item(), targets.size(0))
        loss_trackers['kl_loss'].add(kl_loss.item(), targets.size(0))
        # loss_trackers['loss'].add(loss.item(), targets.size(0))

    return loss

def classifier_loss(outputs, targets,CE_weights,conditional=False,loss_trackers=None):
    loss = 0
    if conditional:
        CE = nn.CrossEntropyLoss(weight=CE_weights)
        # print("batch unique targets",targets.unique())
        loss = CE(outputs,targets)
        # loss_np = loss.detach().cpu().numpy()
        # print("loss (numpy):", loss_np)

    if loss_trackers:
        # print(targets,targets.min(),targets.max())
        # print(targets.size(),outputs.size())
        # print("targets",targets.dtype,outputs.dtype)
        # print("loss (numpy):", loss_np)
        loss_trackers['clf_loss'].add(loss.item(), targets.size(0))

    return loss



def train_classifier_model(trainloader, model,classifier, optimizer,CE_weights,conditional=False, single_batch=False):
    """
    Train the model for one epoch.

    Args:
        trainloader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        single_batch (bool, optional): If True, only one batch is processed. Defaults to False.

    Returns:
        dict: Dictionary containing the average training loss for each loss name.
    """
    model.train()
    if conditional:
        classifier.train()

    # Initialize loss trackers for each loss name in the model
    loss_trackers = {k: AverageMeter() for k in model.loss_names}

    for batch_idx, data in enumerate(trainloader):
        # Extract the image tensor and labels from the dictionary
        inputs = data['image']  # torch.Tensor of shape (batch_size, channels, height, width)
        labels = Variable(data['label_int'])

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            model = model.cuda()
            if conditional:
                classifier = classifier.cuda()
            

        optimizer.zero_grad()  # Reset gradients
        recon_inputs, latents, mu, logvar = model(inputs)  # Forward pass through the model
        loss = compute_loss(recon_inputs,inputs,mu,logvar,lamb=model.lamb,
                             loss_trackers=loss_trackers)  # Compute loss

        if conditional:
            

            # inputs = inputs.cpu()
            # labels = labels.cpu()
            # outputs = classifier(latents.cpu())
            # spam_loss = classifier_loss(outputs, labels.view(-1).long(),
            #                         CE_weights=CE_weights, 
            #                         conditional=True,
            #                           loss_trackers=loss_trackers)
            # print("spam_loss",spam_loss)
            # labels = torch.FloatTensor(data['label_int'])
               
            # print(latents.dtype,labels.dtype)
            outputs = classifier(latents)
            # print(labels.long(),labels.min(),labels.max())
            # print(labels.size(),outputs.size())
            # print(labels.dtype,outputs.dtype)
            loss += model.lamb2 * classifier_loss(outputs,labels.view(-1).long(),
                                    CE_weights=CE_weights,
                                    conditional=True,
                                    loss_trackers=loss_trackers)  # total_clf_loss
        loss_trackers['loss'].add(loss.item(), labels.size(0))
            
             # total_clf_loss

        loss.backward()  # Backward pass to compute gradients
        optimizer.step()  # Update model parameters

        if single_batch:
            break

    # Return the average training loss for each loss name
    return {'train_' + k: loss_trackers[k].value() for k in model.loss_names}





def evaluate_classifier_model(testloader, model, classifier,CE_weights,conditional=False,single_batch=False):
    """
    Evaluate the performance of a given model on a test dataset.

    Args:
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset, 
            expected to yield batches of data in the form of dictionaries with 'image' and 'label' keys.
        model (torch.nn.Module): The model to be evaluated, expected to have a method `compute_loss` 
            and an attribute `loss_names` which is a list of loss component names.
        single_batch (bool, optional): If True, only evaluate on a single batch of data. Defaults to False.

    Returns:
        dict: A dictionary where keys are prefixed with 'test_' followed by the loss component names, 
            and values are the corresponding average loss values over the evaluated batches.
    """
    model.eval()
    if conditional:
        classifier.eval()

    loss_trackers = {k: AverageMeter() for k in model.loss_names}

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            # print(data.keys())
            # Extract the image tensor from the dictionary
            inputs = data['image']
            labels = Variable(data['label_int'])

            # print("Tensor min:", torch.min(inputs).item(), "Tensor max:", torch.max(inputs).item(), "Tensor mean:", torch.mean(inputs).item())



            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
                model = model.cuda()
                if conditional:
                    classifier = classifier.cuda()
                

            recon_inputs, latents, mu, logvar = model(inputs)
            
            # print("inputs at eval",np.min(inputs_np),np.max(inputs_np),np.mean(inputs_np))
            # print("Tensor min:", torch.min(inputs).item(), "Tensor max:", torch.max(inputs).item(), "Tensor mean:", torch.mean(inputs).item())
            # print("recon at eval",np.min(recon_inputs_np),np.max(recon_inputs_np),np.mean(recon_inputs_np))
            # print("lamb,lamb2",model.lamb,model.lamb2)
            loss = compute_loss(recon_inputs,inputs,mu,logvar,lamb=model.lamb, loss_trackers=loss_trackers)
            if conditional:
                
                outputs = classifier(latents)
                loss += model.lamb2 * classifier_loss(outputs,labels.view(-1).long(),
                                        CE_weights=CE_weights,
                                        conditional=True,
                                        loss_trackers=loss_trackers)

            loss_trackers['loss'].add(loss.item(), labels.size(0))
            # inputs_np = inputs.cpu().numpy()
            # recon_inputs_np = recon_inputs.cpu().numpy()


            if single_batch:
                break

    return {'test_'+k: loss_trackers[k].value() for k in model.loss_names}


def save_checkpoint(current_state, filename):
    torch.save(current_state, filename)

def setup_optimizer(name, param_list):
    if name == 'sgd':
        return optim.SGD(param_list, momentum=0.9)
    elif name == 'adam':
        return optim.Adam(param_list)
    else:
        raise KeyError("%s is not a valid optimizer (must be one of ['sgd', adam']" % name)
