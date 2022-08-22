"""
Exemplary predictive model.

You must provide at least 2 methods:
- __init__: Initialization of the class instance
- predict: Uses the model to perform predictions.

The following convenience methods are provided:
- load_microwave_volume: Load three-dimensional microwave image
- visualize_microwave_volume: Visualize the slices of a three-dimensional microwave image
"""

import os
import json
import glob
import numpy as np
import skimage.io
from matplotlib import pyplot as plt

from lib.model.resnet_fcn import ResNetFCN
from lib.dataset.heatmap_dataset import SubmissionHeatmapDataset


def resume_model(checkpoint_dir, model):
    """ Resume if checkpoint is available. """
    last_checkpoint = os.path.join(checkpoint_dir, 'checkpoint.pth')
    state_data = torch.load(last_checkpoint)
    model.load_state_dict(state_data['model'])


class Model:
    """
    Rohde & Schwarz Engineering Competition 2022 class template
    """

    def __init__(self):
        """
        Initialize the class instance

        Important: If you want to refer to relative paths, e.g., './subdir', use
        os.path.join(os.path.dirname(__file__), 'subdir')
        """

        # build model
        self.model = ResNetFCN(
            base_model_name='fcn_resnet50',
            output_size=128,
            n_out_channels=2,
            pretrained=True,
            use_pretrain_head=True,
            use_aux=False
        )

        resume_model(os.path.join(os.path.dirname(__file__), 'ckpt'), self.model)


    def predict(self, data_set_directory):
        """
        This function should provide predictions of labels on a data set.

        Make sure that the predictions are in the correct format for the scoring metric. The method should return an
        array of dictionaries, where the number of dictionaries must match the number of tiff files in
        data_set_dictionary.
        """

        # build dataset
        self.ds = SubmissionHeatmapDataset(
            data_set_directory,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )

        predictions = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.ds):
                # Forward pass
                filename = f'{data["_id"]}.tiff'
                ipt = data['input']
                outputs = self.model(ipt)
                heatmap_pred = outputs['out']
                # fill your code to calculate present/missing pills

                """
                1 1 0 1 1
                1 1 0 1 1
                1 1 0 0 0
                1 1 0 1 1

                a = 1 1 0 2 2   a == 1  ==> 1 1 0 0 0 
                    1 1 0 2 2               1 1 0 0 0 
                    1 1 0 0 0               1 1 0 0 0 
                    1 1 0 3 3               1 1 0 0 0
                """

                prediction = {
                    'file': filename,  # filename of the input image
                    'missing_pills': 0,  # number of missing pills
                    'present_pills': 0,  # number of present pills
                    'coordinates': {
                        'missing': [  # centroids of missing pills. The order of the centroids is arbitrary.
                            
                        ],
                        'present': [  # centroids of present pills. The order of the centroids is arbitrary.
                           
                        ]
                    }
                }

                predictions.append(prediction)
        return predictions

    @staticmethod
    def load_microwave_volume(input_file):
        """
        Load microwave volume from tiff file. Each provided volume contains three slices in propagation direction. The
        provided microwave volumes are given in linear scale.

        :param string input_file: Path to tiff file
        :return ndarray: Image as ndarray with shape MxNx3
        """

        return skimage.io.imread(input_file)

    @staticmethod
    def visualize_microwave_volume(input_file, dynamic_range=25, label=None):
        """
        Visualize the slices of a microwave image volume in logarithmic scale with the given dynamic_range
        :param string input_file: Path to input file
        :param float dynamic_range: Dynamic range in dB (default: 25)
        """

        img = Model.load_microwave_volume(input_file)

        if label is None:
            label_filename = input_file.replace('.tiff', '.json')
            if os.path.exists(label_filename):
                with open(label_filename, 'r') as file:
                    label = json.loads(file.read())

        fig, axs = plt.subplots(1, 3, figsize=(16, 7))
        for i in range(img.shape[2]):
            volume = 20 * np.log10(img[:, :, i])
            max_val = np.max(volume)
            axs[i].imshow(volume, vmax=max_val, vmin=max_val - dynamic_range)
            axs[i].set_title(f"Slice {i + 1:d}")

            if label is not None:
                x_coords = [coord[0] for coord in label['coordinates']['present']]
                y_coords = [257 - coord[1] for coord in label['coordinates']['present']]
                axs[i].scatter(x_coords, y_coords, color='white')

                x_coords = [coord[0] for coord in label['coordinates']['missing']]
                y_coords = [257 - coord[1] for coord in label['coordinates']['missing']]
                axs[i].scatter(x_coords, y_coords, color='red')

        if label is not None:
            fig.canvas.set_window_title(f"Present pills: {len(label['coordinates']['present'])}, "
                                        f"Missing pills: {len(label['coordinates']['missing'])}")

        plt.show()
