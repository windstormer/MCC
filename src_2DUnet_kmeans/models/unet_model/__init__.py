from .unet import Unet
# from .vistr_model import build_model

def build_model(args):
    model = Unet(3, 1)
    return model