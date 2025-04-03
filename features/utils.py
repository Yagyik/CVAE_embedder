import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def save_img(array, filename):
    array = array.view(64, 64).cpu().data.numpy().astype(np.float32)
    # img = Image.fromarray(np.rint(array * 4000).astype(np.float32))
    # array = (array - array.min()) / (array.max() - array.min())
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    ax.imshow(array, cmap='viridis', vmin=0, vmax=1)
    # fig.colorbar(cax, ax=ax)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_side_by_side(array1, array2, filename):
    array1 = array1.view(64, 64).cpu().data.numpy().astype(np.float32)
    array2 = array2.view(64, 64).cpu().data.numpy().astype(np.float32)
    amax = array1.max()
    amin = array1.min()
    array1 = (array1 - amin) / (amax - amin)
    array2 = (array2 - amin) / (amax - amin)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(array1, cmap='viridis', vmin=0, vmax=1)
    axes[0].axis('off')
    axes[1].imshow(array2, cmap='viridis', vmin=0, vmax=1)
    axes[1].axis('off')
    
    plt.savefig(filename, format='png')
    plt.close(fig)