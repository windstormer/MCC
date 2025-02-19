import torch
from torch import nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, padding):
        super(ConvLSTMCell,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + 2*hidden_size, 4 * hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_state_spatial, hidden_state_temporal):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state_spatial is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state_spatial = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )

                
        if hidden_state_temporal is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            hidden_state_temporal = Variable(torch.zeros(state_size)).cuda()



        prev_hidden_spatial, prev_cell_spatial = prev_state_spatial

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_hidden_spatial, hidden_state_temporal], 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell_spatial) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        state = [hidden,cell]

        return state