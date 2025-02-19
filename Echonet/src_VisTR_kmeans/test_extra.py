
import argparse
import os, pickle, glob
from datetime import datetime
import logger
from args_utils import *
from torch.utils.data import DataLoader
from datasets import *
from models import *
from engine import Engine


if __name__ == '__main__':
    args = create_argparser().parse_args()
    gpu_id = args.gid
    if gpu_id == '':
        gpu_id = ",".join([str(g) for g in np.arange(th.cuda.device_count())])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    
    model_name = f"{args.model}_{args.load_model_name}"
    load_model_name = args.load_model_name
    save_dir = os.path.join(args.save_dir, model_name)
    if not os.path.exists(os.path.join(save_dir)):
        raise RuntimeError(f"load model not exist, {load_model_name}")

    data = glob.glob(os.path.join(args.src_dir, "image", "*"))
    test_dataset = ImageDataset(data)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2,
                            pin_memory=True, drop_last=True)
    model = build_model(args)

    load_model_path = os.path.join(save_dir, "model", "encoder.pth")
    log_path = os.path.join(save_dir, "log_extra.txt")

    test_results = Engine(args, gpu_id, None, None, test_loader, model, None, save_dir).test(save_dir, load_model_path)

    log_file = open(log_path, "w+")
    now = datetime.now()
    date_time = now.strftime("%m%d-%H%M%S")
    log_file.writelines(f"\nTest at {date_time}\n")
    log_file.writelines(f"Test Dice: {test_results['Dice'][0]:.3f}+-{test_results['Dice'][1]:.3f}, IOU: {test_results['IoU'][0]:.3f}+-{test_results['IoU'][1]:.3f}, HD95: {test_results['HD95'][0]:.3f}+-{test_results['HD95'][1]:.3f}\n")    
    log_file.close()


