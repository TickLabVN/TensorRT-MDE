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

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set"""

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    # encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
    # decoder_dict = torch.load(decoder_path, map_location=torch.device('cpu'))

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    depth_model_dict = depth_decoder.decoder.state_dict()

    # encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    # depth_decoder.decoder.load_state_dict({k[8:]: v for k, v in decoder_dict.items() if k[8:] in depth_model_dict})

    model = networks.DepthNet(encoder, depth_decoder)
    model.eval()
    model.cuda()

    
    # SCRIPTED_MODEL
    sample_input = torch.tensor(torch.randn(1, 3, 256, 512)).cuda()
    scripted_model = torch.jit.script(model, example_inputs={model: [(sample_input)]})

    # TORCH - TENSORRT
    inputs = [
      torch_tensorrt.Input(
          min_shape=[1, 3, 128, 256],
          opt_shape=[1, 3, 256, 512],
          max_shape=[32, 3, 384, 768],
          dtype=torch.float
      )]
    
    if opt.precision == "fp16":
        enabled_precisions = {torch.half}
    elif opt.precision == "fp32":
        enabled_precisions = {torch.float}
    else:
        enabled_precisions = {torch.float, torch.half}
    
    trt_ts_module = torch_tensorrt.ts.compile(
        scripted_model, inputs=inputs, enabled_precisions=enabled_precisions
    )
    torch.jit.save(trt_ts_module, os.path.join(opt.log_dir, "{}.ts".format(opt.model_name)))

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
