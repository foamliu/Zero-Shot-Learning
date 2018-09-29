import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from scipy.misc import imread, imresize
from torch.utils.data import Dataset
from tqdm import tqdm

from config import *
from utils import *


class ZslDataset(Dataset):
    def __init__(self, super_class, split):
        self.super_class = super_class
        self.split = split
        assert self.super_class in {'Animals', 'Fruits', 'Vehicles', 'Electronics', 'Hairstyles'}
        assert self.split in {'train', 'valid'}

        if super_class == 'Animals':
            annotations_labels = zsl_a_animals_train_annotations_labels
            annotations_attributes_per_clas = zsl_a_animals_train_annotations_attributes_per_clas
            self.image_folder = zsl_a_animals_train_image_folder

        labels = pd.read_csv(annotations_labels, header=None, usecols=[1, 6])
        labels.columns = ['label_id', 'img_path']
        labels['label_id'] = labels['label_id'].str.strip()
        labels['img_path'] = labels['img_path'].str.strip()
        attributes = pd.read_csv(annotations_attributes_per_clas, header=None)
        attributes.columns = ['label_id', 'attributes']
        attributes['attributes'] = attributes['attributes'].str.strip()

        samples = pd.merge(labels, attributes, on='label_id')
        train_count = int(len(samples) * train_split)

        if split == 'train':
            self.samples = samples[:train_count]
        else:
            self.samples = samples[train_count:]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([normalize])

    def __getitem__(self, i):
        img_path = self.samples['img_path'][i]
        attributes = parse_attributes(self.samples['attributes'][i])

        path = os.path.join(self.image_folder, img_path)
        # Read images
        img = imread(path)
        img = imresize(img, (224, 224))
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, 224, 224)
        assert np.max(img) <= 255
        img = torch.FloatTensor(img / 255.)
        if self.transform is not None:
            img = self.transform(img)
        attributes = torch.FloatTensor(attributes)

        return img, np.array(attributes)

    def __len__(self):
        return self.samples.shape[0]


if __name__ == '__main__':
    data_set = ZslDataset('Animals', 'train')
    print(data_set.__len__())
    print(data_set.__getitem__(0))

    print('Checking attributes...')
    for i in tqdm(range(len(data_set))):
        img_path, attributes = data_set[i]
        assert len(attributes) == 123
    print('DONE')
