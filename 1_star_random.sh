#!/bin/bash
MODEL_PATH="deepseek-ai/deepseek-vl2-tiny"
#MODEL_PATH="deepseek-ai/deepseek-vl2-small"
#MODEL_PATH="deepseek-ai/deepseek-vl2"
CONV_MODE=vicuna_v1


export CUDA_VISIBLE_DEVICES=1
CKPT=result_zs

NUM_FRAMES=8

python inference_test/evaluate_star_random.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 20 \
    --num_video_frames $NUM_FRAMES \
    --question-file ./dataset/star/sft_annots_video_v3.3/STAR_val_NEAT_Query_Video_v3.3.json \
    --image-folder ./dataset/star/charadesv1_480/video \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode $CONV_MODE > random_STAR_val_NEAT_x{$NUM_FRAMES}_v3.3.txt



