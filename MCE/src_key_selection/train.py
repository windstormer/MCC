
import argparse
import os, pickle, glob
from datetime import datetime
import logger
from args_utils import *
from torch.utils.data import DataLoader
from datasets import *
from torch.utils.data.sampler import SubsetRandomSampler
from models import build_model
from engine import Engine

def set_loader(args):
    # normal_path = glob.glob(os.path.join(args.src_dir, "training", "normal", "*.jpg"))
    data_path = glob.glob(os.path.join(args.src_dir, "images", args.view, "*"))
    data_path.sort()
    train_path = data_path[:70]
    test_path = data_path[70:]

    print("============== Model Setup ===============")

    dataset = ImageDataset(train_path, args.img_size)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                              pin_memory=False, drop_last=True)

    return loader


if __name__ == '__main__':
    args = create_argparser().parse_args()

    now = datetime.now()
    date_time = now.strftime("%m%d-%H%M%S")
    model_name = f"key_selection_{args.view}_{args.task}"
    save_dir = os.path.join(args.save_dir, model_name)
    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(os.path.join(save_dir))

    log_dir = os.path.join(save_dir)
    logger.configure(dir=log_dir)

    loader = set_loader(args)

    Engine(args, loader, logger, save_dir).run_kmeans()

