import numpy as np
import scipy.io as sio
import os
from PIL import Image
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    split_json_path = "explainable_features/splits.json"

    # glom_images_dir_path = "/Users/nmoreau/Downloads/glom_extracted/glom_roi_images/"
    # glom_labels_dir_path = "/Users/nmoreau/Downloads/glom_extracted/glom_roi_labels/"
    # original_cell_seg_dir_path = "/Users/nmoreau/Downloads/glom_extracted/glom_roi_cell_seg_cellvit/"
    # save_dir = "/Users/nmoreau/Downloads/glom_extracted/"

    glom_images_dir_path = "/scratch/nmoreau/glom_classification_bis/glom_extracted/glom_images/"
    glom_labels_dir_path = "/scratch/nmoreau/glom_classification_bis/glom_extracted/glom_labels/"
    original_cell_seg_dir_path = "/scratch/nmoreau/glom_classification_bis/cellvit_no_filter_pretrained/results/"
    save_dir = "/scratch/nmoreau/volta/new_kidney_data_300_40x/"

    patch_size = (300, 300)

    with open(split_json_path, 'r') as f:
        data_split = json.load(f)

    train_list = data_split["split_1"] + data_split["split_2"]
    val_list = data_split["split_3"]
    test_list = data_split["test"]

    for glom_file_image in os.listdir(glom_images_dir_path):
        if not glom_file_image.startswith("."):
            print(glom_file_image)
            glom_file_image_for_split = glom_file_image.lower().split("_")
            split = []
            for item in train_list:
                item = item.lower().split()
                if 'pas' in item:
                    item.remove('pas')
                if set(glom_file_image_for_split) & set(item) == set(item):
                    split.append('train')
            for item in val_list:
                item = item.lower().split()
                if 'pas' in item:
                    item.remove('pas')
                if set(glom_file_image_for_split) & set(item) == set(item):
                    split.append('validation')
            for item in test_list:
                item = item.lower().split()
                if 'pas' in item:
                    item.remove('pas')
                if set(glom_file_image_for_split) & set(item) == set(item):
                    split.append('test')
            print(split)
            glom_pil = Image.open(glom_images_dir_path + glom_file_image)
            glom_array = np.array(glom_pil)[:1023, :1023, :]

            glom_mask = np.load(glom_labels_dir_path + glom_file_image[:-4]+".npy", allow_pickle=True)[:1023, :1023]
            cell_mask = np.load(original_cell_seg_dir_path + glom_file_image[:-4]+".npy", allow_pickle=True)

            # cell_mask = sio.loadmat(original_cell_seg_dir_path + glom_file_image[:-4] + ".mat")

            cell_mask_inst_map = cell_mask[()]["inst_map"].astype(np.int32)[:1023, :1023]
            cell_mask_type_map = np.zeros((cell_mask_inst_map.shape[0], cell_mask_inst_map.shape[1]))

            inst_list = list(np.unique(cell_mask_inst_map))  # get list of instances
            inst_list.remove(0)  # remove background

            for inst_idx, inst_id in enumerate(inst_list):
                inst_map_mask = np.array(cell_mask_inst_map == inst_id, np.uint8)  # get single object
                nb_pixel_cell = np.count_nonzero(inst_map_mask)
                inst_map_mask[glom_mask == 0] = 0
                nb_pixel_cell_inside_glom = np.count_nonzero(inst_map_mask)
                if nb_pixel_cell_inside_glom > nb_pixel_cell//2:
                    cell_mask_type_map[cell_mask_inst_map == inst_id] = 1
                else :
                    cell_mask_type_map[cell_mask_inst_map == inst_id] = 5

            path_number = 0
            if len(split) == 0 or len(split) > 1:
                print("pass")
            elif split[0] == 'train' or 'validation' or 'test':
                # print("pass")
                for x in range(0, glom_array.shape[0], patch_size[0]):
                    for y in range(0, glom_array.shape[1], patch_size[1]):
                        path_number += 1
                        WSI_patch = glom_array[x:x + patch_size[0], y:y + patch_size[1]]
                        GT_inst_map_patch = cell_mask_inst_map[x:x + patch_size[0], y:y + patch_size[1]]
                        GT_type_map_patch = cell_mask_type_map[x:x + patch_size[0], y:y + patch_size[1]]
                        # rand = random.randrange(5)
                        rand = 0
                        WSI_patch_non_zero = np.count_nonzero(WSI_patch)
                        if GT_inst_map_patch.shape == patch_size and rand == 0 and WSI_patch_non_zero >= (
                                patch_size[0] * patch_size[1]):
                            # print(WSI_patch.shape)
                            glom_pil = Image.fromarray(WSI_patch)
                            glom_pil.save(save_dir + split[0] + '/images_patches/' + glom_file_image[:-4] + "_" + str(
                                path_number) + ".png")
                            outdict = {"inst_map": GT_inst_map_patch, "type_map": GT_type_map_patch}
                            np.save(save_dir + split[0] + '/labels_patches/' + glom_file_image[:-4] + "_" + str(
                                path_number) + ".npy",
                                    outdict)
            else :
                print("ISSUES")


