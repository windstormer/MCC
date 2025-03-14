
import argparse

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def create_argparser():
    defaults = dict(
        data_dir="",
        lr=1e-4,
        weight_decay=1e-6,
        batch_size=1,
        img_size=112
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default='', type=str,
                        help='source dir of dataset ')
    parser.add_argument("--save_dir", default='../results/', type=str,
                        help='where to store results and logger')
    parser.add_argument("--gid",
                        type=str,
                        default ='0',
                        help="gpu id number")
    parser.add_argument("--epochs",
                        "-e",
                        type=int,
                        default=30,
                        help="num of epochs")
    parser.add_argument('--model',
                        type=str,
                        choices=["ConvLSTM"],
                        default='ConvLSTM',
                        help='model select [ConvLSTM]')
    parser.add_argument('--load_model_name',
                        type=str,
                        default='',
                        help='model version to be loaded')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=512,
                        help='dim of the hidden representation in the model')  

    add_dict_to_argparser(parser, defaults)
    return parser

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}