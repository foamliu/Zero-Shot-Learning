import os

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure training/optimization
learning_rate = 0.0001
print_every = 100
start_epoch = 0
epochs = 120
batch_size = 256
train_split = 0.8

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

zsl_a_animals_train_annotations_labels = os.path.join(zsl_a_animals_train_folder,
                                                      'zsl_a_animals_train_annotations_labels_20180321.txt')
zsl_a_animals_train_annotations_attributes_per_clas = os.path.join(zsl_a_animals_train_folder,
                                                                   'zsl_a_animals_train_annotations_attributes_per_class_20180321.txt')
