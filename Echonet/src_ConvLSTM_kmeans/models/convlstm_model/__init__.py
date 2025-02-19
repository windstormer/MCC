from .convlstm import ConvLSTM
# from .vistr_model import build_model

def build_model(args):
    model = ConvLSTM(args.img_size, 3, 1)
    return model