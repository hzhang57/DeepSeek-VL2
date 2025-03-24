#!/bin/bash
MODEL_PATH="deepseek-ai/deepseek-vl2-tiny"
CONV_MODE=vicuna_v1


export CUDA_VISIBLE_DEVICES=1
CKPT=result_zs

NUM_FRAMES=8

python inference_test/evaluate_nextqa_random.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 20 \
    --num_video_frames $NUM_FRAMES \
    --question-file ./dataset/nextqa/scripts/nextqa_val_Query_Video_v3.3.json \
    --image-folder ./dataset/nextqa/NExTVideo \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode $CONV_MODE > random_NextQA_val_x{$NUM_FRAMES}_v3.3.txt

