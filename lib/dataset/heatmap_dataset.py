import os
import json
import skimage.io as io

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class SubmissionHeatmapDataset(Dataset):
    def init(self, dataset_path, transform=None):
        assert heatmap_generator is not None, "[Error] Must provide a heatmap generator!"

        self.path = dataset_path
        self.transform = transform
        self.ids = [item.split('.')[0] for item in os.listdir(self.path) if item.endswith('tiff')]


    def _load_image_source(self, idx):
        """ load a single image by the given index. """
        img_fname = f'{self.ids[idx]}.tiff'
        img_fpath = Path(self.path, img_fname)
        img = io.imread(str(img_fpath), plugin='tifffile')
        return img

    def __len__(self):
        return len(self.ids)

    def getitem(self, idx):
        img = self._load_image_source(idx)

        if self.transform is not None:
            img = self.transform(img)


        data = {
            '_index': idx,
            '_id': self.ids[idx],
            'input': img,
        }

        return data

class DefaultHeatmapDataset(Dataset):
    def __init__(self, dataset_path,
                 heatmap_generator=None,
                 heatmap_size=None,
                 transform=None,
                 target_transform=None,
                 args=None):

        assert heatmap_generator is not None, "[Error] Must provide a heatmap generator!"

        self.path = dataset_path
        self.heatmap_generator = heatmap_generator
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.use_img_size_as_target = True if self.heatmap_size else False
        self.target_transform = target_transform
        self.ids = [item.split('.')[0] for item in os.listdir(self.path) if item.endswith('json')]
        self.image_size = args.image_size if args else 256
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        """ Pre-load all annotations from JSON files. """
        # TODO: this entire method can be refactored later? The json parsing can be duplicated.
        annotations = []
        for item in self.ids:
            coords = []
            json_fpath = Path(self.path, f'{item}.json')
            with open(json_fpath, 'r') as f:
                data = json.load(f)

            # TODO: maybe we should differentiate the missing/present coords?
            # for future extension, now every coord is stored in [x, y, s] format
            # s => missing: 0, present: 1
            for coord in data['coordinates']['missing']:
                # here - self.image_size for flipping the x coord
                coords.append([int(coord[0]), int(np.abs(coord[1] - self.image_size)), 0])  # missing: 0

            for coord in data['coordinates']['present']:
                coords.append([int(coord[0]), int(np.abs(coord[1] - self.image_size)), 1])  # present: 1

            annotations.append(np.asarray(coords))

        return annotations

    def _load_image_source(self, idx):
        """ load a single image by the given index. """
        img_fname = f'{self.ids[idx]}.tiff'
        img_fpath = Path(self.path, img_fname)
        img = io.imread(str(img_fpath), plugin='tifffile')
        return img

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self._load_image_source(idx)
        annot = self.annotations[idx]
        heatmap = self.heatmap_generator(annot)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            heatmap = self.target_transform(heatmap)

        data = {
            '_index': idx,
            '_id': self.ids[idx],
            'input': img,
            'heatmap': heatmap
        }

        return data


# if __name__ == '__main__':
#     from generators import StackedHeatmapGenerator
#     import matplotlib.pyplot as plt

#     image_size = 257
#     heatmap_size = 128

#     ds = DefaultHeatmapDataset(
#         '/home/kulbear/Desktop/cell/public_data',
#         heatmap_generator=StackedHeatmapGenerator(heatmap_size, rescale_factor=heatmap_size / image_size)
#     )

#     for i in range(4):
#         payload = ds[i]
#         # print('[INFO] Sample ID =>', payload['id'])

#         fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
#         axs = axs.flatten()
#         # plt.suptitle(payload['id'])
#         plt.sca(axs[0])
#         plt.imshow(payload['input'])
#         plt.sca(axs[1])
#         plt.title('Missing')
#         plt.imshow(payload['heatmap'][0])
#         plt.sca(axs[2])
#         plt.title('Present')
#         plt.imshow(payload['heatmap'][1])
#         plt.show()
