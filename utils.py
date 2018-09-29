import os


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def parse_attributes(attr_str):
    tokens = attr_str.split(' ')
    attr_list = []
    for i in range(1, len(tokens) - 1):
        attr_list.append(float(tokens[i]))
