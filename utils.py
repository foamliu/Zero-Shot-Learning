from config import *


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ExpoAverageMeter(object):
    # Exponential Weighted Average Meter
    def __init__(self, beta=0.9):
        self.reset()

    def reset(self):
        self.beta = 0.9
        self.val = 0
        self.avg = 0
        self.t = 0

    def update(self, val):
        self.avg = self.beta * self.avg + (1 - self.beta) * val
        self.val = val / (1 - self.beta ** self.t)
        self.t += 1


def KNN(mat, k):
    mat = mat.float()
    mat_square = torch.mm(mat, mat.t())
    diag = torch.diagonal(mat_square)
    # print(diag)
    # print(diag.size())
    val, index = diag.topk(k, largest=False, sorted=True)
    return val, index


def batched_KNN(query, k, attributes):
    batch_size = query.size()[0]
    val_list = torch.zeros(batch_size, dtype=torch.float, device=device)
    index_list = torch.zeros(batch_size, dtype=torch.int, device=device)
    for i in range(batch_size):
        q = query[i].to(device)
        attributes = attributes.to(device)
        diff = q - attributes
        diff = torch.tensor(diff)
        diff = diff.to(device)
        val, index = KNN(diff, k)
        val_list[i] = val
        index_list[i] = index
    return val_list, index_list


def accuracy(scores, targets):
    batch_size = targets.size(0)
    correct = scores.eq(targets)
    # print('correct: ' + str(correct))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_checkpoint(epoch, model, W, optimizer, val_acc, is_best, superclass):
    ensure_folder(save_folder)
    state = {'model': model,
             'W': W,
             'optimizer': optimizer}

    if is_best:
        filename = '{0}/checkpoint_{1}_{2}_{3:.3f}.tar'.format(save_folder, superclass, epoch, val_acc)
        torch.save(state, filename)

        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        torch.save(state, '{}/BEST_{}_checkpoint.tar'.format(save_folder, superclass))


def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def get_label_list():
    import pandas as pd
    labels = pd.read_csv(
        'data/ai_challenger_zsl2018_train_test_a_20180321/zsl_a_animals_train_20180321/zsl_a_animals_train_annotations_label_list_20180321.txt',
        header=None)
    labels.columns = ['label_name', 'cat_name_en', 'cat_name_zh']
    labels['label_name'] = labels['label_name'].str.strip()
    labels['cat_name_zh'] = labels['cat_name_zh'].str.strip()
    label_list = []
    for i in range(len(labels)):
        label_list.append(labels['cat_name_zh'][i])

    print('len(label_list): ' + str(len(label_list)))
    return label_list


def get_annotations_by_superclass(superclass):
    if superclass == 'Animals':
        image_folder = zsl_a_animals_train_image_folder
        annotations_labels = zsl_a_animals_train_annotations_labels
        annotations_attributes_per_class = zsl_a_animals_train_annotations_attributes_per_class
        annotations_attribute_list = zsl_a_animals_train_annotations_attribute_list
    elif superclass == 'Fruits':
        image_folder = zsl_a_fruits_train_image_folder
        annotations_labels = zsl_a_fruits_train_annotations_labels
        annotations_attributes_per_class = zsl_a_fruits_train_annotations_attributes_per_class
        annotations_attribute_list = zsl_a_fruits_train_annotations_attribute_list
    elif superclass == 'Vehicles':
        image_folder = zsl_b_vehicles_train_image_folder
        annotations_labels = zsl_b_vehicles_train_annotations_labels
        annotations_attributes_per_class = zsl_b_vehicles_train_annotations_attributes_per_class
        annotations_attribute_list = zsl_b_vehicles_train_annotations_attribute_list
    elif superclass == 'Electronics':
        image_folder = zsl_b_electronics_train_image_folder
        annotations_labels = zsl_b_electronics_train_annotations_labels
        annotations_attributes_per_class = zsl_b_electronics_train_annotations_attributes_per_class
        annotations_attribute_list = zsl_b_electronics_train_annotations_attribute_list
    else:  # 'Hairstyles'
        image_folder = zsl_b_hairstyles_train_image_folder
        annotations_labels = zsl_b_hairstyles_train_annotations_labels
        annotations_attributes_per_class = zsl_b_hairstyles_train_annotations_attributes_per_class
        annotations_attribute_list = zsl_b_hairstyles_train_annotations_attribute_list

    return image_folder, annotations_labels, annotations_attributes_per_class, annotations_attribute_list


def get_label_name2idx_by_superclass(superclass):
    _, _, annotations_attributes_per_class, _ = get_annotations_by_superclass(superclass)
    attributes = pd.read_csv(annotations_attributes_per_class, header=None)
    attributes.columns = ['label_name', 'attributes']
    attributes['attributes'] = attributes['attributes'].str.strip()
    label_name2idx = dict()
    for i in range(len(attributes)):
        label_name2idx[attributes['label_name'][i]] = i
    # print(label_name2idx)
    return label_name2idx


def get_embedding_size_by_superclass(superclass):
    _, _, _, annotations_attribute_list = get_annotations_by_superclass(superclass)
    with open(annotations_attribute_list, 'r') as file:
        lines = file.readlines()

    lines = [line for line in lines if len(line.strip()) > 0]
    return len(lines)


def get_attributes_per_class_by_superclass(superclass):
    _, _, annotations_attributes_per_class, _ = get_annotations_by_superclass(superclass)
    attributes = pd.read_csv(annotations_attributes_per_class, header=None)
    attributes.columns = ['label_name', 'attributes']
    attributes['attributes'] = attributes['attributes'].str.strip()

    attributes_per_class = []
    for i in range(len(attributes)):
        attributes_per_class.append(parse_attributes(attributes['attributes'][i]))
    attributes_per_class = torch.tensor(attributes_per_class)
    attributes_per_class.to(device)
    return attributes_per_class


def get_test_folder_by_superclass(superclass):
    if superclass == 'Animals':
        test_folder = zsl_a_animals_test_folder
    elif superclass == 'Fruits':
        test_folder = zsl_a_fruits_test_folder
    elif superclass == 'Vehicles':
        test_folder = zsl_b_vehicles_test_folder
    elif superclass == 'Electronics':
        test_folder = zsl_b_electronics_test_folder
    else:  # 'Hairstyles'
        test_folder = zsl_b_hairstyles_test_folder

    return test_folder


def get_attribute_names_by_superclass(superclass):
    _, _, _, annotations_attribute_list = get_annotations_by_superclass(superclass)
    attribute_list = pd.read_csv(annotations_attribute_list, header=None, usecols=[2])
    attribute_list.columns = ['attribute_name']
    attribute_list['attribute_name'] = attribute_list['attribute_name'].str.strip()
    attribute_names = []
    for i in range(len(attribute_list)):
        attribute_name = attribute_list['attribute_name'][i]
        attribute_name = attribute_name.split(': ')[1]
        attribute_names.append(attribute_name)
    attribute_names = np.array(attribute_names)
    return attribute_names
