import config

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MapDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = os.path.join(self.root_dir, self.files[index])
        image = np.array(Image.open(file_path))

        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = config.transform(image=input_image, target_image=target_image)
        input_image, target_image = augmentations['image'], augmentations['target_image']

        input_image = config.transform_input(image=input_image)['image']
        target_image = config.transform_target(image=target_image)['image']

        return input_image, target_image


def test():
    dataset = MapDataset('data/maps/train')
    input_image, target_image = dataset[0]

    assert input_image.shape == (3, 256, 256)
    assert target_image.shape == (3, 256, 256)
    print('Test passed successfully')

if __name__ == '__main__':
    test()
