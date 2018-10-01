# -*- coding: utf-8 -*-
import json

if __name__ == '__main__':
    with open('result.json', 'r', encoding="utf-8") as file:
        result = json.load(file)

    with open('README.template', 'r', encoding="utf-8") as file:
        template = file.readlines()

    template = ''.join(template)

    for i in range(10):
        template = template.replace('$(attributes_{})'.format(i), result[i]['attributes'])
        template = template.replace('$(cat_{})'.format(i), result[i]['label_name'])

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(template)
