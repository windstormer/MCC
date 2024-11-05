import torch
import os
from tqdm import tqdm
import torch.nn as nn

from utils import *
from sklearn.cluster import KMeans

class Engine:
    def __init__(
        self, 
        args,
        loader,
        logger,
        save_dir
        ):
        self.save_dir = save_dir
        self.loader = loader
        # self.loss = nn.BCEWithLogitsLoss()
        self.logger = logger
    
    def run(self):
        train_bar = tqdm(self.loader)
        for data, case_id in train_bar:
            # print(case_id[0])
            data = data.squeeze(0)
            feature = data.flatten(1)
            sim = cos_simi(feature, feature)
            avg_sim = torch.mean(sim, axis=1)
            print(case_id[0], avg_sim)
            max_values, max_idx = torch.topk(avg_sim, k=3, axis=-1)
            min_values, min_idx = torch.topk(avg_sim, k=3, axis=-1, largest=False)
            
            self.logger.log(f"{case_id[0]}:{max_idx[0]},{max_idx[1]},{max_idx[2]}")

    def run_kmeans(self):
        train_bar = tqdm(self.loader)
        for data, case_id in train_bar:
            cluster = {0:[], 1:[], 2:[]}
            
            # print(case_id[0])
            data = data.squeeze(0)
            feature = data.flatten(1)
            model = KMeans(n_clusters=3, n_init='auto', random_state=1)
            model.fit(feature)
            for idx, l in enumerate(model.labels_):
                cluster[l].append(str(idx))
                # print(l)
            c0 = ",".join(cluster[0])
            c1 = ",".join(cluster[1])
            c2 = ",".join(cluster[2])
            self.logger.log(f"{case_id[0]}:{c0}/{c1}/{c2}")


        