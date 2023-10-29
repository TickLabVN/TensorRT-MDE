import torch
import torch_tensorrt
import time
import numpy as np
import torch.backends.cudnn as cudnn
import PIL.Image as pil
from torchvision import transforms


def save_depth(depth, name):
    depth = depth.reshape((1, 1, 256, 512))
    depth = 1 / (depth + 1e-9)
    scaled_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    scaled_depth = np.squeeze(scaled_depth)
    scaled_depth = np.repeat(scaled_depth[None, :, :], 3, 0)
    scaled_depth = np.transpose(scaled_depth, (1, 2, 0))
    scaled_depth = np.clip(scaled_depth * 255, 0, 255).astype(np.uint8)
    scaled_depth = pil.fromarray(scaled_depth)
    scaled_depth.save("/workspace/source/output/tensorrt_fp32/{:010d}.jpg".format(name))
    
def load_input(image_path):
    input_image = pil.open(image_path).convert('RGB')
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    input_image = input_image
    return input_image

cudnn.benchmark = True

def benchmark(model, use_cuda = True, input_shape=(1024, 3, 512, 512), dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    if use_cuda:
        input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            pred_loc  = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print('Average throughput: %.2f images/second'%(input_shape[0]/np.mean(timings)))

ts_path = "/workspace/source/ckpts/resnet101_jit_cuda.pt"
tensorrt_path = "/workspace/source/ckptsv2/resnet18_infp32_modelfp32.ts"

trt_model = torch.jit.load(tensorrt_path)
# benchmark(trt_model, use_cuda = True, input_shape=(4, 3, 256, 512), nruns=100, dtype="fp32")

with open("/workspace/source/filenames/day_val_451.txt", "r") as file:
    img_names = file.readlines()
img_names = [int(img_name) for img_name in img_names]
    
# preds = []
print(">> Inference")
for name in img_names:
    print("{:010d}".format(name))
    input = load_input("/workspace/source/data/rgb/{:010d}.png".format(name))
    # input = input.half()
    output = trt_model(input.cuda())
    output = output.detach().cpu().numpy()
    save_depth(output, name)