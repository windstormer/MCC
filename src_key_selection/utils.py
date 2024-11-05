from skimage import io
from skimage import img_as_ubyte
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F

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
        io.imsave(os.path.join(save_dir, case_id, "input", f"input_{i}.png"), img_as_ubyte(np_input_image[0]), check_contrast=False)
        io.imsave(os.path.join(save_dir, case_id, "gt", f"gt_{i}.png"), img_as_ubyte(np_ground_truth[0].astype(np.float32)), check_contrast=False)
        io.imsave(os.path.join(save_dir, case_id, "pred", f"pred_{i}.png"), img_as_ubyte(np_pred[0].astype(np.float32)), check_contrast=False)

def img_fusion(image, heatmap):
    cam = heatmap + np.float32(image)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return image, cam

def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C
