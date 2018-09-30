import unittest

import pandas as pd

from config import *
from utils import *


class TestStringMethods(unittest.TestCase):

    def test_KNN(self):
        annotations_attributes_per_class = zsl_a_animals_train_annotations_attributes_per_class
        attributes = pd.read_csv(annotations_attributes_per_class, header=None)
        attributes.columns = ['label_id', 'attributes']
        attributes['attributes'] = attributes['attributes'].str.strip()

        attributes_per_class = []
        for i in range(len(attributes)):
            attributes_per_class.append(parse_attributes(attributes['attributes'][i]))
        attributes_per_class = torch.tensor(attributes_per_class)
        print(attributes_per_class.size())

        test_attr = parse_attributes(
            '[[ 0.20 0.25 0.00 0.00 0.55 0.90 0.05 0.05 0.00 0.00 0.40 0.00 1.00 0.00 1.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 1.00 0.35 0.00 0.00 0.90 0.10 0.00 0.30 0.70 0.00 1.00 0.00 0.00 0.00 0.00 0.00 1.00 1.00 0.00 0.00 1.00 1.00 1.00 0.00 0.15 0.85 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 1.00 0.00 1.00 1.00 0.00 1.00 1.00 0.00 1.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 1.00 0.00 0.00 1.00 0.00 1.00 0.80 1.00 0.00 1.00 0.00 1.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 1.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 1.00 0.00 0.00 ]')
        test_attr = torch.tensor(test_attr)
        print(test_attr.size())

        diff = test_attr - attributes_per_class
        print(diff.size())

        val, index = KNN(diff, 1)
        print(val, index)

        self.assertEqual(index.item(), 0)


if __name__ == '__main__':
    unittest.main()
