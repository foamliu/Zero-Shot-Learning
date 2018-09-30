import json
import random

import numpy as np
import torchvision.transforms as transforms
from scipy.misc import imread, imresize, imsave

from config import *
from utils import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([normalize])


def main():
    checkpoint = 'BEST_checkpoint.tar'  # model checkpoint
    # Load model
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    # model = model.to(device)
    model = model.cuda()
    # model.eval()

    files = [os.path.join(zsl_a_animals_test_folder, file) for file in os.listdir(zsl_a_animals_test_folder) if
             file.lower().endswith('.jpg')]
    samples = random.sample(files, 10)

    imgs = torch.zeros([10, 3, 224, 224], dtype=torch.float, device=device)

    for i, path in enumerate(samples):
        # Read images
        img = imread(path)
        img = imresize(img, (224, 224))
        imsave('images/image_{}.jpg'.format(i), img)

        img = img.transpose(2, 0, 1)
        assert img.shape == (3, 224, 224)
        assert np.max(img) <= 255
        img = torch.FloatTensor(img / 255.)
        img = transform(img)
        imgs[i] = img

    imgs = torch.tensor(imgs)
    imgs.to(torch.device("cuda"))
    print('imgs.device: ' + str(imgs.device))

    result = []
    with torch.no_grad():
        preds = model(imgs)

    _, scores = batched_KNN(preds, 1)

    batch_size = preds.size()[0]
    label_list = get_label_list()

    for i in range(batch_size):
        embeded = preds[i]
        print('embeded: ' + str(embeded))
        score = scores[i]
        print('score: ' + str(score))
        result.append({'i': i, 'cat_name_zh': label_list[score]})

    with open('result.json', 'w') as file:
        json.dump(result, file, indent=4)


if __name__ == '__main__':
    main()
