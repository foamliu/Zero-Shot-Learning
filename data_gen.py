import pandas as pd
from torch.utils.data import Dataset

from config import *


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
        attributes = pd.read_csv(annotations_attributes_per_clas, header=None)
        attributes.columns = ['label_id', 'attributes']

        self.samples = pd.merge(labels, attributes, on='label_id')

    def __getitem__(self, i):
        img_path = self.samples['img_path'][i]
        attributes = self.samples['attributes'][i]
        return img_path, attributes

    def __len__(self):
        return self.samples.shape[0]


if __name__ == '__main__':
    data_set = ZslDataset('Animals', 'train')
    print(data_set.__len__())
    print(data_set.__getitem__(0))
    print(data_set.__getitem__(1))
    print(data_set.__getitem__(2))
    print(data_set.__getitem__(3))
    print(data_set.__getitem__(4))
