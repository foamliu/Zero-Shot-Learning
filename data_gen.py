import numpy as np
import torchvision.transforms as transforms
from scipy.misc import imread, imresize
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import *


class ZslDataset(Dataset):
    def __init__(self, super_class, split):
        self.super_class = super_class
        self.split = split
        assert self.super_class in {'Animals', 'Fruits', 'Vehicles', 'Electronics', 'Hairstyles'}
        assert self.split in {'train', 'valid'}

        if super_class == 'Animals':
            annotations_labels = zsl_a_animals_train_annotations_labels
            annotations_attributes_per_class = zsl_a_animals_train_annotations_attributes_per_class
            self.image_folder = zsl_a_animals_train_image_folder

        annotations_labels = pd.read_csv(annotations_labels, header=None, usecols=[1, 6])
        annotations_labels.columns = ['label_id', 'img_path']
        annotations_labels['label_id'] = annotations_labels['label_id'].str.strip()
        annotations_labels['img_path'] = annotations_labels['img_path'].str.strip()
        attributes_per_class = pd.read_csv(annotations_attributes_per_class, header=None)
        attributes_per_class.columns = ['label_id', 'attributes']
        attributes_per_class['attributes'] = attributes_per_class['attributes'].str.strip()

        samples = pd.merge(annotations_labels, attributes_per_class, on='label_id')
        train_count = int(len(samples) * train_split)

        if split == 'train':
            self.samples = samples[:train_count]
            self.start_index = 0
        else:
            self.samples = samples[train_count:]
            self.start_index = train_count

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([normalize])

    def __getitem__(self, i):
        img_path = self.samples['img_path'][self.start_index + i]
        attributes = parse_attributes(self.samples['attributes'][self.start_index + i])

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
    train_data = ZslDataset('Animals', 'train')
    print(len(train_data))
    print(train_data[0])

    print('Checking train data attributes...')
    for i in tqdm(range(len(train_data))):
        img_path, attributes = train_data[i]
        assert len(attributes) == 123
    print('DONE')

    val_data = ZslDataset('Animals', 'valid')
    print(len(val_data))
    print(val_data[0])

    print('Checking valid data attributes...')
    for i in tqdm(range(len(val_data))):
        img_path, attributes = val_data[i]
        assert len(attributes) == 123
    print('DONE')
