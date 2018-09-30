import os

import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure training/optimization
learning_rate = 0.0001
print_every = 100
start_epoch = 0
epochs = 120
batch_size = 16
train_split = 0.8
feature_size = 2048  # ResNet-101 feature size
print_freq = 10

data_folder = 'data'
test_a_folder = 'data/ai_challenger_zsl2018_train_test_a_20180321'
test_b_folder = 'data/ai_challenger_zsl2018_test_b_20180423'

zsl_a_animals_train_folder = os.path.join(test_a_folder, 'zsl_a_animals_train_20180321')
zsl_a_animals_test_folder = os.path.join(test_a_folder, 'zsl_a_animals_test_20180321')
zsl_a_fruits_train_folder = os.path.join(test_a_folder, 'zsl_a_fruits_train_20180321')
zsl_a_fruits_test_folder = os.path.join(test_a_folder, 'zsl_a_fruits_test_20180321')
zsl_b_electronics_train_folder = os.path.join(data_folder, 'zsl_b_electronics_train_20180321')
zsl_b_electronics_test_folder = os.path.join(data_folder, 'zsl_b_electronics_test_20180321')
zsl_b_hairstyles_train_folder = os.path.join(data_folder, 'zsl_b_electronics_train_20180321')
zsl_b_hairstyles_test_folder = os.path.join(data_folder, 'zsl_b_electronics_train_20180321')
zsl_b_vehicles_train_folder = os.path.join(data_folder, 'zsl_b_electronics_train_20180321')
zsl_b_vehicles_test_folder = os.path.join(data_folder, 'zsl_b_electronics_train_20180321')

zsl_a_animals_train_image_folder = os.path.join(zsl_a_animals_train_folder, 'zsl_a_animals_train_images_20180321')

zsl_a_animals_train_annotations_labels = os.path.join(zsl_a_animals_train_folder,
                                                      'zsl_a_animals_train_annotations_labels_20180321.txt')
zsl_a_animals_train_annotations_attributes_per_class = os.path.join(zsl_a_animals_train_folder,
                                                                    'zsl_a_animals_train_annotations_attributes_per_class_20180321.txt')


def parse_attributes(attr_str):
    tokens = attr_str.split(' ')
    attr_list = []
    for i in range(1, len(tokens) - 1):
        attr_list.append(float(tokens[i]))

    return attr_list


# Cached data
annotations_attributes_per_class = zsl_a_animals_train_annotations_attributes_per_class
attributes = pd.read_csv(annotations_attributes_per_class, header=None)
attributes.columns = ['label_name', 'attributes']
attributes['attributes'] = attributes['attributes'].str.strip()

attributes_per_class = []
for i in range(len(attributes)):
    attributes_per_class.append(parse_attributes(attributes['attributes'][i]))
attributes_per_class = torch.tensor(attributes_per_class)
attributes_per_class.to(device)
# print(attributes_per_class.size())

annotations_labels = zsl_a_animals_train_annotations_labels
annotations_labels = pd.read_csv(annotations_labels, header=None, usecols=[1, 6])
annotations_labels.columns = ['label_name', 'img_path']
annotations_labels['label_name'] = annotations_labels['label_name'].str.strip()
annotations_labels['img_path'] = annotations_labels['img_path'].str.strip()

label_name2idx = dict()
for i in range(len(attributes_per_class)):
    label_name2idx[attributes_per_class['label_name'][i]] = i
