import torch
import torch.nn as nn

class DepthNet(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, encoder, decoder, min_depth = 0.1, max_depth=100, med_scale = 17.769):
        super(DepthNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.med_scale = med_scale

    def disp_to_depth(self, disp):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth
    
    def forward(self, input_image):
        features = self.encoder(input_image)
        disp = self.decoder(features)

        depth = self.disp_to_depth(disp) * self.med_scale

        return depth