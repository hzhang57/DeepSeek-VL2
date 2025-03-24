#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
CKPT=result_zs
CONV_MODE=vicuna_v1
NUM_FRAMES=8

#MODEL_PATH="deepseek-ai/deepseek-vl2-tiny"
#MODEL_PATH="deepseek-ai/deepseek-vl2-small"
MODEL_PATH="deepseek-ai/deepseek-vl2"

python inference_test/evaluate_star_video.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 20 \
    --num_video_frames $NUM_FRAMES \
    --question-file ./dataset/star/sft_annots_video_v3.3/STAR_val_NEAT_Query_Video_v3.3.json \
    --image-folder ./dataset/star/charadesv1_480/video \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode $CONV_MODE > seekvl2_STAR_val_NEAT_Query_Video_x{$NUM_FRAMES}_v3.3.txt



python inference_test/evaluate_star_video.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 20 \
    --num_video_frames $NUM_FRAMES \
    --question-file ./dataset/star/sft_annots_video_v3.3/STAR_val_NEAT_Option_Video_v3.3.json \
    --image-folder ./dataset/star/charadesv1_480/video \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode $CONV_MODE > seekvl2_STAR_val_NEAT_Option_Video_x{$NUM_FRAMES}_v3.3.txt


MODEL_PATH="deepseek-ai/deepseek-vl2-small"
python inference_test/evaluate_star_video.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 20 \
    --num_video_frames $NUM_FRAMES \
    --question-file ./dataset/star/sft_annots_video_v3.3/STAR_val_NEAT_Query_Video_v3.3.json \
    --image-folder ./dataset/star/charadesv1_480/video \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode $CONV_MODE > seekvl2_small_STAR_val_NEAT_Query_Video_x{$NUM_FRAMES}_v3.3.txt

python inference_test/evaluate_star_video.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 20 \
    --num_video_frames $NUM_FRAMES \
    --question-file ./dataset/star/sft_annots_video_v3.3/STAR_val_NEAT_Option_Video_v3.3.json \
    --image-folder ./dataset/star/charadesv1_480/video \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode $CONV_MODE > seekvl2_small_STAR_val_NEAT_Option_Video_x{$NUM_FRAMES}_v3.3.txt


MODEL_PATH="deepseek-ai/deepseek-vl2-tiny"
python inference_test/evaluate_star_video.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 20 \
    --num_video_frames $NUM_FRAMES \
    --question-file ./dataset/star/sft_annots_video_v3.3/STAR_val_NEAT_Query_Video_v3.3.json \
    --image-folder ./dataset/star/charadesv1_480/video \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode $CONV_MODE > seekvl2_tiny_STAR_val_NEAT_Query_Video_x{$NUM_FRAMES}_v3.3.txt

python inference_test/evaluate_star_video.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 20 \
    --num_video_frames $NUM_FRAMES \
    --question-file ./dataset/star/sft_annots_video_v3.3/STAR_val_NEAT_Option_Video_v3.3.json \
    --image-folder ./dataset/star/charadesv1_480/video \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode $CONV_MODE > seekvl2_tiny_STAR_val_NEAT_Option_Video_x{$NUM_FRAMES}_v3.3.txt
