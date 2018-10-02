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
        self.count = 0

    def update(self, val):
        self.val = val
        self.avg = self.beta * self.avg + (1 - self.beta) * self.val


def KNN(mat, k):
    mat = mat.float()
    mat_square = torch.mm(mat, mat.t())
    diag = torch.diagonal(mat_square)
    # print(diag)
    # print(diag.size())
    val, index = diag.topk(k, largest=False, sorted=True)
    return val, index


def batched_KNN(query, k):
    attributes = attributes_per_class
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


def save_checkpoint(epoch, model, optimizer, val_acc, is_best):
    state = {'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_{0}_{1:.3f}.tar'.format(epoch, val_acc)
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


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
