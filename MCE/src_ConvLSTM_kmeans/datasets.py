import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os, glob
from utils import load_key_file

class ImageDataset(Dataset):
    def __init__(self, data, img_size, key_file):
        self.data = data
        # self.transform = transforms.ToTensor()
        self.transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
        self.key_file = key_file
        if self.key_file is not None:
            self.chosen_idx = load_key_file(key_file)

        ## origin augmentation


    def __getitem__(self, index):
        case_path = self.data[index]
        case_id = case_path.split(os.path.sep)[-1]
        img_per_case = []
        label_per_case = []
        img_path_list = glob.glob(os.path.join(case_path, "*.png"))
        img_path_list.sort()
        for img_path in img_path_list:
            label_path = img_path.replace("images", "labels")
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            img_per_case.append(img.unsqueeze(0))

            label = Image.open(label_path)
            label = self.transform(label)
            label_per_case.append(label.unsqueeze(0))

        img_per_case = torch.cat(img_per_case, dim=0)
        label_per_case = torch.cat(label_per_case, dim=0)

        if self.key_file is not None:
            chosen_idx = self.chosen_idx[case_id]
            return img_per_case, label_per_case.float(), chosen_idx, case_id
        else:
            return img_per_case, label_per_case.float(), case_id
    
    def __len__(self):
        return len(self.data)

