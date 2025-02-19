
import argparse
import os, pickle, glob
from datetime import datetime
import logger
from args_utils import *
from torch.utils.data import DataLoader
from datasets import *
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from models import build_model
from engine import Engine

def set_loader(args):
    # normal_path = glob.glob(os.path.join(args.src_dir, "training", "normal", "*.jpg"))
    data_path = glob.glob(os.path.join(args.src_dir, "images", args.view, "*"))
    data_path.sort()
    logger.log("Num of All Data:", len(data_path))
    train_path = data_path[:70]
    test_path = data_path[70:]

    logger.log("Num of Training Case:", len(train_path))
    logger.log("Num of Test Case:", len(test_path))

    print("============== Model Setup ===============")

    train_index, val_index = train_test_split(range(len(train_path)), test_size=0.1)
    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(val_index)

    key_path = os.path.join(args.save_dir, f"key_selection_{args.view}_{args.key}", "log.txt")
    
    dataset = ImageDataset(train_path, args.img_size, key_path)
    test_dataset = ImageDataset(test_path, args.img_size, None)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                              pin_memory=False, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                            pin_memory=True, drop_last=True, sampler=test_sampler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                            pin_memory=True, drop_last=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    args = create_argparser().parse_args()
    gpu_id = args.gid
    if gpu_id == '':
        gpu_id = ",".join([str(g) for g in np.arange(th.cuda.device_count())])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    now = datetime.now()
    date_time = now.strftime("%m%d-%H%M%S")
    if args.key is not None:
        model_name = f"{args.model}_{args.view}_{date_time}_key_{args.key}"
    else:
        model_name = f"{args.model}_{args.view}_{date_time}"
    save_dir = os.path.join(args.save_dir, model_name)
    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(os.path.join(save_dir))

    log_dir = os.path.join(save_dir)
    logger.configure(dir=log_dir)
    logger.logkvs(vars(args))
    logger.dumpkvs()

    train_loader, val_loader, test_loader = set_loader(args)
    model = build_model(args)

    Engine(args, gpu_id, train_loader, val_loader, test_loader, model, logger, save_dir).run()

