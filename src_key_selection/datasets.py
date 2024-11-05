import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os, glob
class ImageDataset(Dataset):
    def __init__(self, data, img_size):
        self.data = data
        # self.transform = transforms.ToTensor()
        self.transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
        ## origin augmentation


    def __getitem__(self, index):
        case_path = self.data[index]
        case_id = case_path.split(os.path.sep)[-1]
        img_per_case = []
        img_path_list = glob.glob(os.path.join(case_path, "*.png"))
        img_path_list.sort()
        for img_path in img_path_list:
            img = Image.open(img_path)
            img = self.transform(img)
            img_per_case.append(img.unsqueeze(0))

        img_per_case = torch.cat(img_per_case, dim=0)


        return img_per_case, case_id
    
    def __len__(self):
        return len(self.data)
