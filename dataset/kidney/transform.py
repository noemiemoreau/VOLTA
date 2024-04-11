import albumentations


class LabelTransform(object):

    def __init__(self, n_classes):
        if n_classes == 3:
            self.label_mapper = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}
        elif n_classes == 5:
            self.label_mapper = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4}
        elif n_classes == 7:
            self.label_mapper = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
        else:
            # self.label_mapper = lambda x: x
            raise ValueError("invalid number of classes, {}".format(n_classes))

    def __call__(self, x):
        return self.label_mapper[x]


def get_cell_normalization():
    return albumentations.Normalize(mean=[0.5452, 0.3847, 0.5605], std=[0.1911, 0.1716, 0.1435])  # scale 0
    return albumentations.Normalize(mean=[0.6121, 0.4434, 0.6072], std=[0.1957, 0.1861, 0.1509])  # scale 1


def get_patch_normalization(patch_size):
    if patch_size == 0:
        return albumentations.ToFloat()
    if patch_size == 100 or True:
        return albumentations.Normalize(mean=[0.6284, 0.4568, 0.6178], std=[0.1934, 0.1886, 0.1514])
    raise ValueError('no normalization is found for patch size of {}'.format(patch_size))
