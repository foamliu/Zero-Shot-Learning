import json
import random

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

    num_test_samples = 50
    samples = random.sample(files, num_test_samples)

    imgs = torch.zeros([num_test_samples, 3, 224, 224], dtype=torch.float, device=device)

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

    for i in range(num_test_samples):
        embeded = preds[i]
        embeded = embeded.cpu().numpy()
        attributes = attribute_names[embeded >= 0.9]
        attributes = ', '.join(attributes)
        # print('embeded: ' + str(embeded))
        labal_id = scores[i].item()
        label_name = 'Label_A_%02d' % (labal_id + 1,)
        print('labal_id: ' + str(labal_id))
        result.append(
            {'i': i, 'labal_id': labal_id, 'label_name': label_name, 'attributes': attributes})

    with open('result.json', 'w') as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
