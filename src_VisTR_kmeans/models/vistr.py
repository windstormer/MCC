"""
VisTR model and criterion classes.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import torch
import torch.nn.functional as F
from torch import nn

from .backbone import build_backbone
from .segmentation import VisTRsegm
from .transformer import build_transformer


class VisTR(nn.Module):
    """ This is the VisTR module that performs video object detection """
    def __init__(self, backbone, transformer, num_classes, num_frames):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         VisTR can detect in a video. For ytvos, we recommend 10 queries for each frame, 
                         thus 360 queries for 36 frames.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_frames = num_frames
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_frames, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples, length):
        """ It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_frames x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # moved the frame to batch dimension for computation efficiency
        
        features, pos = self.backbone(samples)
        pos = pos[-1]
        src = features[-1]
        src_proj = self.input_proj(src)
        n,c,h,w = src_proj.shape

        assert mask is not None
        src_proj = src_proj.reshape(n//length, length, c, h, w).permute(0,2,1,3,4).flatten(-2)
        mask = mask.reshape(n//length, length, h*w)
        pos = pos.permute(0,2,1,3,4).flatten(-2)
        hs = self.transformer(src_proj, mask, self.query_embed.weight[:length], pos)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        return out

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = torch.device('cuda')

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = VisTR(
        backbone,
        transformer,
        num_classes=1,
        num_frames=30
    )

    model = VisTRsegm(model)
    return model
