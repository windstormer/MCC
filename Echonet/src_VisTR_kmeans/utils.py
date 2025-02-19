from skimage import io
from skimage import img_as_ubyte
import cv2
import numpy as np
import os
import torch

def draw_image_index(save_dir, case_id, input_image, ground_truth, pred, index):
    # index = index.item()
    np_input_image = input_image.permute(0,2,3,1).numpy()
    np_ground_truth = ground_truth.permute(0,2,3,1).squeeze(3).numpy()
    np_pred = pred.permute(0,2,3,1).squeeze(3).numpy()
    if not os.path.exists(os.path.join(save_dir, case_id, "input")):
        os.makedirs(os.path.join(save_dir, case_id, "input"))
    if not os.path.exists(os.path.join(save_dir, case_id, "gt")):
        os.makedirs(os.path.join(save_dir, case_id, "gt"))
    if not os.path.exists(os.path.join(save_dir, case_id, "pred")):
        os.makedirs(os.path.join(save_dir, case_id, "pred"))
    io.imsave(os.path.join(save_dir, case_id, "input", f"input_l_{index[0]}.png"), img_as_ubyte(np_input_image[0]), check_contrast=False)
    io.imsave(os.path.join(save_dir, case_id, "gt", f"gt_l_{index[0]}.png"), img_as_ubyte(np_ground_truth[0].astype(np.float32)), check_contrast=False)
    io.imsave(os.path.join(save_dir, case_id, "pred", f"pred_l_{index[0]}.png"), img_as_ubyte(np_pred[0].astype(np.float32)), check_contrast=False)
    io.imsave(os.path.join(save_dir, case_id, "input", f"input_s_{index[1]}.png"), img_as_ubyte(np_input_image[1]), check_contrast=False)
    io.imsave(os.path.join(save_dir, case_id, "gt", f"gt_s_{index[1]}.png"), img_as_ubyte(np_ground_truth[1].astype(np.float32)), check_contrast=False)
    io.imsave(os.path.join(save_dir, case_id, "pred", f"pred_s_{index[1]}.png"), img_as_ubyte(np_pred[1].astype(np.float32)), check_contrast=False)
    for i in range(2, len(np_input_image)):
        io.imsave(os.path.join(save_dir, case_id, "input", f"input_{index[i]}.png"), img_as_ubyte(np_input_image[i]), check_contrast=False)
        io.imsave(os.path.join(save_dir, case_id, "pred", f"pred_{index[i]}.png"), img_as_ubyte(np_pred[i].astype(np.float32)), check_contrast=False)

def img_fusion(image, heatmap):
    cam = heatmap + np.float32(image)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return image, cam

def load_key_file(key_path):
    out_dict = {}
    file = open(key_path, 'r')
    file.readline()
    ll = file.readlines()
    for line in ll:
        case_id = line.split(":")[0]
        frame_list = line.split(":")[1]
        cluster_list = frame_list.split("/")
        out_dict[case_id] = []
        for cluster in cluster_list:
            chosen_frame_list = cluster.split(",")
            out_dict[case_id].append([int(f) for f in chosen_frame_list])
    return out_dict

def draw_image(save_dir, case_id, input_image, ground_truth, pred):
    np_input_image = input_image.permute(0,2,3,1).numpy()
    np_ground_truth = ground_truth.permute(0,2,3,1).squeeze(3).numpy()
    np_pred = pred.permute(0,2,3,1).squeeze(3).numpy()
    if not os.path.exists(os.path.join(save_dir, case_id, "input")):
        os.makedirs(os.path.join(save_dir, case_id, "input"))
    if not os.path.exists(os.path.join(save_dir, case_id, "gt")):
        os.makedirs(os.path.join(save_dir, case_id, "gt"))
    if not os.path.exists(os.path.join(save_dir, case_id, "pred")):
        os.makedirs(os.path.join(save_dir, case_id, "pred"))
    for i in range(np_input_image.shape[0]):
        io.imsave(os.path.join(save_dir, case_id, "input", f"input_{i}.png"), img_as_ubyte(np_input_image[i]), check_contrast=False)
        io.imsave(os.path.join(save_dir, case_id, "gt", f"gt_{i}.png"), img_as_ubyte(np_ground_truth[i].astype(np.float32)), check_contrast=False)
        io.imsave(os.path.join(save_dir, case_id, "pred", f"pred_{i}.png"), img_as_ubyte(np_pred[i].astype(np.float32)), check_contrast=False)