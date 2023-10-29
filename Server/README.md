# Accelerating MDE with TensorRT

## Convert Pytorch to TensorRT

You can download DepthNet's ckpt from [here](https://drive.google.com/file/d/1R1tzpvCdZ1c_3kyIWLPR0YZdpl__yIlY/view?usp=sharing).

```
python3 converter.py \
        --load_weights_folder ./ckpts \
        --log_dir ./log\
        --model_name model\
        --num_layers 18\
        --precision fp16
```

## Benchmark & Inference

### Benchmark

```
python3 infer.py \
        --load_weights_path ./log/model.ts \
        --benchmark
```

### Inference:

You can download sample data from [here](https://drive.google.com/file/d/1Ap-kwj0zZhIXrze-q1kDp8dynamQrRlP/view?usp=sharing).

```
python3 infer.py \
        --load_weights_path ./log/model.ts \
        --image_dir data\
        --output_dir vis
```

## Citation

```
@article{ICCV 19,
title={Digging into self-supervised monocular depth estimation},
author={Clément Godard, Oisin Mac Aodha, Michael Firman, Gabriel Brostow},
conference={2019 IEEE/CVF International Conference on Computer Vision },
year={2019}
}
```
