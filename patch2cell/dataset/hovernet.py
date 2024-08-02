import scipy.io
import numpy as np


class Hovernet:

    def __init__(self):
        self.input_image_dir_name = "Images_patches"
        self.input_label_dir_name = "Labels_patches"
        self.input_ihc_dir_name = None
        self.skip_labels = [5, 7]
        self.labeling_type = 'mask'
        self.first_valid_instance = 1

    @staticmethod
    def get_instance_name_from_file_name(file_name):
        instance_name = file_name.split('.')[-2].split('\\')[-1]
        return instance_name

    @staticmethod
    def read_instance_mask(file_path):
        # label = scipy.io.loadmat(file_path)
        label = np.load(file_path, allow_pickle=True)

        return label[()]["inst_map"].astype(np.int32)

    @staticmethod
    def read_type_mask(file_path):
        # label = scipy.io.loadmat(file_path)
        label = np.load(file_path, allow_pickle=True)
        return label[()]["type_map"].astype(np.int32)
