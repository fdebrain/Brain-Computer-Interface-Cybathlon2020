import os


def clean_log_directory(path_to_clean):
    for files in path_to_clean.glob('*.txt'):
        os.remove(files)

    if not os.path.exists(path_to_clean):
        os.mkdir(path_to_clean)
