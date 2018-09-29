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

        with open(annotations_labels, 'r') as file:
            label_lines = file.readlines()

        for label_line in label_lines:
            tokens = [token.split() for token in label_line.split(',')]
            label_id = tokens[1]
            image_path = tokens[3]

        with open(annotations_attributes_per_clas, 'r') as file:
            attributes_lines = file.readlines()

        for attributes_line in attributes_lines:
            tokens = [token.split() for token in attributes_line.split(',')]
            label_id = tokens[0]



    def __getitem__(self, i):
        return None

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    dataset = ZslDataset('Animals', 'train')

