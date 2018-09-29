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

        if split == 'train':
            samples_path = 'data/samples_train.json'
            self.samples = json.load(open(samples_path, 'r'))
        else:
            samples_path = 'data/samples_valid.json'
            self.samples = json.load(open(samples_path, 'r'))

    def __getitem__(self, i):
        pair_batch = []

        for i_batch in range(batch_size):
            sample = self.samples[i + i_batch]
            pair_batch.append((sample['input'], sample['output']))

        return batch2TrainData(pair_batch)

    def __len__(self):
        return len(self.samples)
