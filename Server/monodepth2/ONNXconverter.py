from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
from options import MonodepthOptions
import networks
import torch.nn as nn
import PIL.Image as pil
from torchvision import transforms
import torch_tensorrt

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

from typing import List

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

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set"""

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
    decoder_dict = torch.load(decoder_path, map_location=torch.device('cpu'))

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    depth_model_dict = depth_decoder.decoder.state_dict()

    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.decoder.load_state_dict({k[8:]: v for k, v in decoder_dict.items() if k[8:] in depth_model_dict})

    model = DepthNet(encoder, depth_decoder)

    model.eval()

    sample_input = torch.tensor(torch.randn(1, 3, 256, 512)).cuda()
    model.cuda()
    
    # SCRIPTED_MODEL
    scripted_model = torch.jit.script(model, example_inputs={model: [(sample_input)]})
    
    # ONNX MODEL
    torch.onnx.export(
          scripted_model, 
          sample_input, 
          "/content/drive/MyDrive/MinhHuy/JetsonNano/ckpts/monodepth2_oxford_day.onnx", 
          verbose=True,
          input_names = ['input'],
          output_names = ['output']) 

    # TORCH - TENSORRT
    # 1. Converting nn.Module into TensorRT-Torch script module
    model_name = "resnet{}_infp32_modelfpmix".format(opt.num_layers)
    inputs = [
      torch_tensorrt.Input(
          min_shape=[1, 3, 128, 256],
          opt_shape=[1, 3, 256, 512],
          max_shape=[32, 3, 384, 768],
          dtype=torch.float
      )]
    enabled_precisions = {torch.float, torch.half}      
    trt_ts_module = torch_tensorrt.ts.compile(
        scripted_model, inputs=inputs, enabled_precisions=enabled_precisions
    )

    # 2. Save TensorRT-Torch script module
    torch.jit.save(trt_ts_module, 
        "/workspace/source/ckpts/{}.ts".format(model_name)
    )

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
