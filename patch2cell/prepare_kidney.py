import os
import random

import matplotlib.pyplot as plt
from skimage.draw import polygon
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import json

TYPE_NUCLEI_DICT = {
    1: "Opal_480", #podocytes
    2: "Opal_520", #PECs
    3: "Opal_570", #endothelial
    4: "Opal_620", #mesangials
    5: "Opal_690", #immune
    6: "Outside",
    7: "Unclassified"
}

if __name__ == "__main__":
    WSIs_path = "/Users/nmoreau/Documents/Data/Kidney/new_organization/processed_data/volta_bis/data_cells_300/test/WSIs/"
    GTs_geojson_path = "/Users/nmoreau/Documents/Data/Kidney/new_organization/processed_data/volta_bis/data_cells_300/test/Cell_labels_geojson/"
    ROIs_geojson_path = "/Users/nmoreau/Documents/Data/Kidney/new_organization/processed_data/volta_bis/data_cells_300/test/ROIs_geojson/"
    images_path = "/Users/nmoreau/Documents/Data/Kidney/new_organization/processed_data/volta_bis/data_cells_300/test/images_patches/"
    labels_path = "/Users/nmoreau/Documents/Data/Kidney/new_organization/processed_data/volta_bis/data_cells_300/test/labels_patches/"
    TYPE_NUCLEI_DICT_inv = {TYPE_NUCLEI_DICT[k]: k for k in TYPE_NUCLEI_DICT.keys()}
    patch_size = (300, 300)
    for image_name in os.listdir(WSIs_path):
        if not image_name.startswith("."):
            image_name = image_name[:-8]

            with open(GTs_geojson_path + image_name + ".geojson", 'r') as f:
                gson_cells_gt = json.load(f)
            with open(ROIs_geojson_path + image_name + ".geojson", 'r') as f:
                gson_rois_gt = json.load(f)
            rois_list = gson_rois_gt["features"]
            cells_gt_list = gson_cells_gt["features"]

            WSI_pil = Image.open(WSIs_path + image_name + "_PAS.png")
            WSI_array = np.array(WSI_pil)

            for roi in rois_list:
                roi_id = roi["id"]

                x_roi_list = [coord[0] for coord in roi["geometry"]["coordinates"][0]]
                y_roi_list = [coord[1] for coord in roi["geometry"]["coordinates"][0]]
                xmax = max(x_roi_list)
                xmin = min(x_roi_list)
                ymax = max(y_roi_list)
                ymin = min(y_roi_list)

                WSI_roi = WSI_array[ymin:ymax, xmin:xmax, :3]
                GT_type_map = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint16)
                GT_inst_map = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint16)
                i = 0
                for cell in cells_gt_list:
                    if cell["properties"]["objectType"] == "cell":
                        id_cell = cell["id"]
                        list_coord_cell = cell["geometry"]["coordinates"][0]
                        properties = cell["properties"]
                        if "classification" in properties.keys():
                            name = properties["classification"]["name"]
                        else:
                            name = "Outside"
                        x1 = list_coord_cell[0][0]
                        y1 = list_coord_cell[0][1]
                        if xmin < x1 < xmax and ymin < y1 < ymax:
                            i += 1
                            new_list_coord_cell = []
                            new_list_coord_nuclear = []
                            for coord in list_coord_cell:
                                new_coord_cell = [coord[0] - xmin, coord[1] - ymin]
                                new_list_coord_cell.append(new_coord_cell)
                            poly = np.array(new_list_coord_cell[:-1])
                            rr, cc = polygon(poly[:, 0], poly[:, 1], (GT_inst_map.shape[1], GT_inst_map.shape[0]))
                            GT_inst_map[cc, rr] = i
                            GT_type_map[cc, rr] = TYPE_NUCLEI_DICT_inv[name]
                path_number = 0
                for x in range(0, WSI_roi.shape[0], patch_size[0]):
                    for y in range(0, WSI_roi.shape[1], patch_size[1]):
                        path_number += 1
                        WSI_patch = WSI_roi[x:x + patch_size[0], y:y + patch_size[1]]
                        GT_inst_map_patch = GT_inst_map[x:x + patch_size[0], y:y + patch_size[1]]
                        GT_type_map_patch = GT_type_map[x:x + patch_size[0], y:y + patch_size[1]]
                        # rand = random.randrange(5)
                        rand = 0
                        if GT_inst_map_patch.shape == patch_size and rand == 0:
                            # print(WSI_patch.shape)
                            WSI_patch_pil = Image.fromarray(WSI_patch)
                            WSI_patch_pil.save(images_path + image_name + "_" + str(roi_id) + "_" + str(path_number) + ".png")

                            outdict = {"inst_map": GT_inst_map_patch, "type_map": GT_type_map_patch}
                            np.save(labels_path + image_name + "_" + str(roi_id) + "_" + str(path_number) + ".npy", outdict)
                            # plt.imshow(WSI_patch)
                            # plt.show()
                            # plt.imshow(GT_inst_map_patch)
                            # plt.show()
                            # plt.imshow(GT_type_map_patch)
                            # plt.show()
                # plt.imshow(WSI_roi)
                # plt.show()
                # plt.imshow(GT_inst_map)
                # plt.show()
                # plt.imshow(GT_type_map)
                # plt.show()

