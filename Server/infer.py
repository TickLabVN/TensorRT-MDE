import torch
import torch_tensorrt
import time
import numpy as np
import torch.backends.cudnn as cudnn
import PIL.Image as pil
from torchvision import transforms
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_dir', type=str,
                        help='path to folder images, for inference mode')
    parser.add_argument('--output_dir', type=str,
                        help='path to save images, for inference mode')
    parser.add_argument('--load_weights_path', type=str,
                        help='Path contain tensorrt ckpts', required=True)
    parser.add_argument('--benchmark',
                        help='Benchmark mode', 
                        action="store_true")

    return parser.parse_args()

def save_depth(depth, name, out_dir):
    depth = depth.reshape((1, 1, 256, 512))
    depth = 1 / (depth + 1e-9)
    scaled_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    scaled_depth = np.squeeze(scaled_depth)
    scaled_depth = np.repeat(scaled_depth[None, :, :], 3, 0)
    scaled_depth = np.transpose(scaled_depth, (1, 2, 0))
    scaled_depth = np.clip(scaled_depth * 255, 0, 255).astype(np.uint8)
    scaled_depth = pil.fromarray(scaled_depth)
    scaled_depth.save(os.path.join(out_dir, "{:010d}.jpg".format(name)))
    
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

if __name__ == "__main__":
    args = parse_args()    
    tensorrt_path = args.load_weights_path
    trt_model = torch.jit.load(tensorrt_path)
    
    if args.benchmark:
        benchmark(trt_model, use_cuda = True, input_shape=(1, 3, 256, 512), nruns=100, dtype="fp32")
    else:
        with open("splits/day_val_451.txt", "r") as file:
            img_names = file.readlines()
        img_names = [int(img_name) for img_name in img_names]
            
        for name in img_names:
            input = load_input(os.path.join(args.image_dir, "{:010d}.png".format(name)))
            output = trt_model(input.cuda())
            output = output.detach().cpu().numpy()
            save_depth(output, name)