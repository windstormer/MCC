import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .clstm import ConvLSTMCell
from .resnet import *
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights

class ConvLSTM(nn.Module):
    def __init__(self, ori_size, in_channels, n_classes, base_model="resnet50"):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, base_model=base_model)
        self.decoder = RSIS(n_classes=n_classes)
        self.upsample_match = nn.UpsamplingBilinear2d(size=(ori_size))

    def forward(self, x):
        feats = self.encoder(x)

        prev_hidden_temporal = None
        for ii in range(len(feats)):
            out_mask, hidden = self.decoder(feats, None, prev_hidden_temporal)
            hidden_tmp = []
            for ss in range(len(hidden)):
                hidden_tmp.append(hidden[ss][0].data)
            prev_hidden_temporal = hidden_tmp
        out_mask = self.upsample_match(out_mask)
        return out_mask
        



class Encoder(nn.Module):
    def __init__(self, in_channels, base_model):
        super().__init__()
        self.in_channels = in_channels

        if base_model == 'resnet18':
            self.base = ResNet18()
            self.base.load_state_dict(models.resnet18(weights=ResNet18_Weights.DEFAULT).state_dict())
            skip_dims_in = [512,256,128,64,64]
        elif base_model == 'resnet34':
            self.base = ResNet34()
            self.base.load_state_dict(models.resnet34(weights=ResNet34_Weights.DEFAULT).state_dict())
            skip_dims_in = [512,256,128,64,64]
        elif base_model == 'resnet50':
            self.base = ResNet50()
            self.base.load_state_dict(models.resnet50(weights=ResNet50_Weights.DEFAULT).state_dict())
            skip_dims_in = [2048,1024,512,256,64]
        elif base_model == 'resnet101':
            self.base = ResNet101()
            self.base.load_state_dict(models.resnet101(weights=ResNet101_Weights.DEFAULT).state_dict())
            skip_dims_in = [2048,1024,512,256,64]

        self.hidden_size = 128
        self.kernel_size = 3
        self.padding = 0 if self.kernel_size == 1 else 1

        self.sk5 = nn.Conv2d(skip_dims_in[0], int(self.hidden_size), self.kernel_size, padding=self.padding)
        self.sk4 = nn.Conv2d(skip_dims_in[1], int(self.hidden_size), self.kernel_size, padding=self.padding)
        self.sk3 = nn.Conv2d(skip_dims_in[2], int(self.hidden_size / 2), self.kernel_size, padding=self.padding)
        self.sk2 = nn.Conv2d(skip_dims_in[3], int(self.hidden_size / 4), self.kernel_size, padding=self.padding)

        self.bn5 = nn.BatchNorm2d(int(self.hidden_size))
        self.bn4 = nn.BatchNorm2d(int(self.hidden_size))
        self.bn3 = nn.BatchNorm2d(int(self.hidden_size / 2))
        self.bn2 = nn.BatchNorm2d(int(self.hidden_size / 4))

    def forward(self, x):
        x5, x4, x3, x2, x1 = self.base(x)

        x5_skip = self.bn5(self.sk5(x5))
        x4_skip = self.bn4(self.sk4(x4))
        x3_skip = self.bn3(self.sk3(x3))
        x2_skip = self.bn2(self.sk2(x2))

        return x5_skip, x4_skip, x3_skip, x2_skip

class RSIS(nn.Module):
    """
    The recurrent decoder
    """

    def __init__(self, n_classes):

        super(RSIS, self).__init__()
        self.hidden_size = 128
        self.kernel_size = 3
        padding = 0 if self.kernel_size == 1 else 1

        self.dropout = 0.1
        self.skip_mode = 'concat'

        # convlstms have decreasing dimension as width and height increase
        skip_dims_out = [self.hidden_size, int(self.hidden_size / 2),
                         int(self.hidden_size / 4), int(self.hidden_size / 8)]

        # skip_dims_out = [self.hidden_size, int(self.hidden_size / 2),
        #                  int(self.hidden_size / 4), int(self.hidden_size / 8), int(self.hidden_size / 16)]

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 5 is the number of deconv steps that we need to reach image size in the output
        for i in range(len(skip_dims_out)):
            if i == 0:
                clstm_in_dim = self.hidden_size
            else:
                clstm_in_dim = skip_dims_out[i - 1]
                if self.skip_mode == 'concat':
                    clstm_in_dim *= 2

            clstm_i = ConvLSTMCell(clstm_in_dim, skip_dims_out[i], self.kernel_size, padding=padding)
            self.clstm_list.append(clstm_i)

        self.conv_out = nn.Conv2d(skip_dims_out[-1], n_classes, self.kernel_size, padding=padding)



    def forward(self, skip_feats, prev_state_spatial, prev_hidden_temporal):

        clstm_in = skip_feats[0]
        skip_feats = skip_feats[1:]
        hidden_list = []

        for i in range(len(skip_feats) + 1):

            # hidden states will be initialized the first time forward is called
            if prev_state_spatial is None:
                if prev_hidden_temporal is None:
                    state = self.clstm_list[i](clstm_in, None, None)
                else:
                    state = self.clstm_list[i](clstm_in, None, prev_hidden_temporal[i])
            else:
                # else we take the ones from the previous step for the forward pass
                if prev_hidden_temporal is None:
                    state = self.clstm_list[i](clstm_in, prev_state_spatial[i], None)

                else:
                    state = self.clstm_list[i](clstm_in, prev_state_spatial[i], prev_hidden_temporal[i])

            hidden_list.append(state)
            hidden = state[0]

            if self.dropout > 0:
                hidden = nn.Dropout2d(self.dropout)(hidden)

            # apply skip connection
            if i < len(skip_feats):

                skip_vec = skip_feats[i]
                upsample = nn.UpsamplingBilinear2d(size=(skip_vec.size()[-2], skip_vec.size()[-1]))
                hidden = upsample(hidden)
                # skip connection
                if self.skip_mode == 'concat':
                    clstm_in = torch.cat([hidden, skip_vec], 1)
                elif self.skip_mode == 'sum':
                    clstm_in = hidden + skip_vec
                elif self.skip_mode == 'mul':
                    clstm_in = hidden * skip_vec
                elif self.skip_mode == 'none':
                    clstm_in = hidden
                else:
                    raise Exception('Skip connection mode not supported !')
            else:
                upsample = nn.UpsamplingBilinear2d(size=(hidden.size()[-2] * 2, hidden.size()[-1] * 2))
                hidden = upsample(hidden)
                clstm_in = hidden

        out_mask = self.conv_out(clstm_in)
        # classification branch

        return out_mask, hidden_list