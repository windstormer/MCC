from .unet_model import build_model as unet_build_model

def build_model(args):
    if args.model == "2DUnet":
        model = unet_build_model(args)
    else:
        raise RuntimeError(f"undefined model, {args.model}.")

    return model