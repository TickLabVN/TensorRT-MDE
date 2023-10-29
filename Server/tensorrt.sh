/usr/src/tensorrt/bin/trtexec \
        --loadEngine="/workspace/source/engine/jetson.trt"\
        --warmUp=1\
        --iterations=1\
        --threads\
        --workspace=1024\
        --useSpinWait\
        --loadInputs='input:/workspace/source/data/dat/0000030910.dat'\
        --exportOutput='/workspace/source/output/tensorrt_fp32/json/0000030910.json'\
        --best\
        --verbose
        #--saveEngine=/workspace/source/engine/jetson.trt\
        #--onnx=/workspace/source/ckpts/jetson.onnx\

# FILES="/workspace/source/data/dat/*"
# for f in $FILES
# do
#     out_file = ${f:27:10}
# done
        #--loadEngine="/workspace/source/engine/model.trt"\
        #--streams=1\
        #--buildOnly\
        # --shapes=inputs:2x3x256x512
        # --minShapes=inputs:1x3x128x256\
        # --optShapes=inputs:1x3x256x512\
        # --maxShapes=inputs:4x3x384x768\
        #--saveEngine=/workspace/source/engine/model.engine\

        #--exportOutput=`/workspace/source/output/$out_file.json`