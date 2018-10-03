# -*- coding: utf-8 -*-
import json
import os

if __name__ == '__main__':
    for superclass in ['Animals', 'Fruits', 'Vehicles', 'Electronics', 'Hairstyles']:
        filename = 'result_{}.json'.format(superclass)
        if os.path.isfile(filename):
            with open(filename, 'r', encoding="utf-8") as file:
                result = json.load(file)

            with open('README.template', 'r', encoding="utf-8") as file:
                template = file.readlines()

            template = ''.join(template)

            for i in range(10):
                template = template.replace('$(attributes_{}_{})'.format(superclass, i), result[i]['attributes'])
                template = template.replace('$(cat_{}_{})'.format(superclass, i), result[i]['label_name'])

            with open('README.md', 'w', encoding="utf-8") as file:
                file.write(template)
