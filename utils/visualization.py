import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_loss_curve(loss_histories, title='Loss Curve', save_path=None):
    """
    Plot loss curves for one or multiple models.

    Parameters:
        loss_histories (dict or list): A dictionary with model names as keys and loss values as values, 
                                       or a list of loss values for a single model.
        title (str): Title of the plot.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
    """
    plt.figure(figsize=(10, 6))

    if isinstance(loss_histories, dict):
        for model_name, loss_history in loss_histories.items():
            plt.plot(loss_history, label=model_name)
    else:
        plt.plot(loss_histories, label='Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_field(field, x, y, title='Field Visualization', cmap='jet', levels=100, save_path=None, plot_type='contourf'):
    """
    Plot a field as a 2D contour or 3D surface.

    Parameters:
        field (numpy array): The field values to plot (e.g., temperature, velocity).
        x, y (numpy array): Coordinates corresponding to the field values.
        title (str): Title of the plot.
        cmap (str): Colormap for the plot.
        levels (int): Number of levels for contour plots.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
        plot_type (str): Type of plot ('contourf' or 'surface').
    """
    plt.figure(figsize=(10, 8))
    if plot_type == 'contourf':
        contour = plt.contourf(x, y, field, levels=levels, cmap=cmap)
        plt.colorbar(contour)
    elif plot_type == 'surface':
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, field, cmap=cmap, edgecolor='none')
        ax.set_zlabel('Field Value')

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")

    plt.show()
