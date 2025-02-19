from .convlstm_model import build_model as convlstm_build_model

def build_model(args):
    if args.model == "ConvLSTM":
        model = convlstm_build_model(args)
    else:
        raise RuntimeError(f"undefined model, {args.model}.")

    return model