import os


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
