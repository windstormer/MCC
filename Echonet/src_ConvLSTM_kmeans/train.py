
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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    args = create_argparser().parse_args()
    gpu_id = args.gid
    if gpu_id == '':
        gpu_id = ",".join([str(g) for g in np.arange(th.cuda.device_count())])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    now = datetime.now()
    date_time = now.strftime("%m%d-%H%M%S")
    model_name = f"{args.model}_{date_time}"
    save_dir = os.path.join(args.save_dir, model_name)
    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(os.path.join(save_dir))

    log_dir = os.path.join(save_dir)
    logger.configure(dir=log_dir)
    logger.logkvs(vars(args))
    logger.dumpkvs()
    
    key_path = os.path.join(args.save_dir, f"key_selection", "log.txt")

    dataset = Echo(root=args.src_dir, key_file=key_path, split='train', max_length=150, target_type=["LargeTrace", "SmallTrace"])
    val_dataset = Echo(root=args.src_dir, key_file=key_path, split='val', max_length=150, target_type=["LargeTrace", "SmallTrace"])
    test_dataset = Echo(root=args.src_dir, split='test', max_length=150, target_type=["LargeTrace", "SmallTrace", "LargeIndex", "SmallIndex"])
    

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2,
                            pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2,
                            pin_memory=True, drop_last=True)

    model = build_model(args)

    Engine(args, gpu_id, train_loader, val_loader, test_loader, model, logger, save_dir).run()

