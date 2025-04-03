# from .AE import AE
# from .augmentedAE import AugmentedAE
from .CVAE import VAE, CVAE
from .CVAE import FC_Classifier

# model_dict = {'AE': AE, 'AugmentedAE': AugmentedAE}

model_dict = {'VAE': VAE,'CVAE':CVAE}
classifier_dict = {'FC_Classifier': FC_Classifier}
