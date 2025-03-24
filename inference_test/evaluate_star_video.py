import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from PIL import Image
import math

from torch.utils.data import Dataset, DataLoader

import requests
from io import BytesIO
import re

from decord_func import decord_video_given_start_end_seconds
from pathlib import Path

import re
from eval_utils import parse_choice
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

import os
from typing import List, Literal

def parse_first_number(text):
    """
    This function takes a string as input and returns the first number it finds in the string.
    If no number is found, it returns None.
    
    :param text: The input string to search for a number
    :return: The first number found as a string, or None if no number is found
    """
    # Use regular expression to find the first number in the string
    match = re.search(r'\d+', text)
    
    # Return the matched number if found, otherwise return None
    return match.group() if match else None



class TypeAccuracy(object):
    def __init__(self, type_name):
        self.correct = 0
        self.total = 10e-9
        self.type_name = type_name

    def update(self, gt, pred):
        self.total += 1
        if "{}".format(pred) in gt:
            self.correct += 1

    def get_accuracy(self):
        return 1.0*self.correct / self.total

    def print_accuracy(self):
        print("{} Accuracy: {:.4f} | {}/{}".format(
                self.type_name,
                self.get_accuracy(),
                self.correct,
                self.total
            ))

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out



class vllm_deepseek_v2:
     def __init__(self, model_path):
         # specify the path to the model
         model = model_path
         vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
         tokenizer = vl_chat_processor.tokenizer
         vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
         vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

         self.vl_gpt = vl_gpt
         self.tokenizer = tokenizer
         self.vl_chat_processor = vl_chat_processor

     def predict(self, images, qs):
         # multiple images/interleaved image-text
         conversation = [
            {
                "role": "<|User|>",
                "content": "{}".format(qs),
                #"images": [
                #    "images/hack_{}.jpeg".format(i) for i in range(len(images))
                #        ],
            },
            {"role": "<|Assistant|>", "content": ""}
            ]
         # load images and prepare for inputs
         prepare_inputs = self.vl_chat_processor(
                 conversations=conversation,
                 images=images,
                 force_batchify=True,
                 system_prompt=""
                 ).to(self.vl_gpt.device)
                 # run image encoder to get the image embeddings
         inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
         # run the model to get the response
         outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
            )
         answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
         answer = answer.replace("<｜end▁of▁sentence｜>", "") # Remove end token
         #image_placeholder = "<image>"*len(images)
         #qs = qs.replace("<video>\n", image_placeholder)
         return answer

def main(args):
    # Load Model
    #disable_torch_init()
    ### Load Model
    vllm = vllm_deepseek_v2(args.model_path)

    # Load Questions
    annotations = json.load(open(os.path.expanduser(args.question_file), "r"))

    # Overall Accuracy for All Questions
    correct = 0
    total = 0

    global_acc = TypeAccuracy("Global")
    qa1_acc = TypeAccuracy("Interact")
    qa2_acc = TypeAccuracy("Sequence")
    qa3_acc = TypeAccuracy("Predict")
    qa4_acc = TypeAccuracy("Feasibility")

    ii = 0
    for line in tqdm(annotations, total=len(annotations)):
        #if ii > 100:
        #    break
        #ii+=1
        # Q-A Pair
        idx = line["id"]
        quest_type = line["quest_type"]
        conversations = line["conversations"]
        qs = conversations[0]["value"]
        gt_answers   = conversations[1]["value"]
        index2ans = line["index2ans"]
        all_choices = line["all_choices"]
        
        use_image = False
        with torch.inference_mode():
            if args.num_video_frames > 0:
                use_image = True
                # Load Image
                video_path = os.path.join(args.image_folder, line["video"])
                
                if "start_secs" in line:
                    start_secs = line['start_secs']
                    end_secs = line['end_secs']
                    frames, frame_indices =  decord_video_given_start_end_seconds(video_path, 
                        start_secs=start_secs, end_secs=end_secs,
                        num_video_frames=args.num_video_frames)
                    print("st-ed {}-{}".format(start_secs, end_secs))
                else:
                    frames, frame_indices =  decord_video_given_start_end_seconds(video_path,
                        num_video_frames=args.num_video_frames)
                print(frames.shape)
                images =[  Image.fromarray(x).convert('RGB') for x in frames ]
                #images =[  Image.fromarray(x.astype('uint8')) for x in frames ]

                n_images = len(images)
                print(images[0].size, n_images)

                    
            #print("HEHREH ", qs)
            qs = qs + "Return only the index of the correct answer (e.g. 0, 1, 2, 3)"
            img_placehoder = '<image>\n' * n_images
            qs = qs.replace("<video>\n", img_placehoder)

            if use_image:
                outputs = vllm.predict(images, qs)

        # Decode output
        outputs = outputs.strip()
        total += 1
        answer_id = parse_choice(outputs, all_choices, index2ans)
        global_acc.update(gt_answers, answer_id)
        print("{}:\n{}".format(idx, qs))
        #print("Global Accu{:.4f}.\nGT: {}\nAI: {}".format(correct*1.0/total, gt_answers, outputs))
        print("GT: {}\nAI: {}".format(gt_answers, outputs))
        if "Interaction" in quest_type:
            qa1_acc.update(gt_answers, answer_id)
        elif "Sequence" in quest_type:
            qa2_acc.update(gt_answers, answer_id)
        elif "Prediction" in quest_type:
            qa3_acc.update(gt_answers, answer_id)
        elif "Feasibility" in quest_type:
            qa4_acc.update(gt_answers, answer_id)
        else:
            print(f"Unknown Type: {idx}")
        # print each type accuracy
        print("-----"*5)
        qa1_acc.print_accuracy()
        qa2_acc.print_accuracy()
        qa3_acc.print_accuracy()
        qa4_acc.print_accuracy()
        print("-----"*5)
        # average over type
        avg_acc = (qa1_acc.get_accuracy() + qa2_acc.get_accuracy() + qa3_acc.get_accuracy() + qa4_acc.get_accuracy() ) / 4.0
        print("Average Acc over Type: {:.4f}".format(avg_acc))

    print("Process Finished")

def parse_answer(outputs):
    if "Answer is:" in outputs:
    # with graph
        outputs = outputs.split("Answer is: ")[-1]
    if "answer is " in outputs:
    # with graph
        outputs = outputs.split("answer is ")[-1].strip(".")
    # remove graph
    answer_id = outputs[0]
    try:
        answer_id = int(answer_id)
        return answer_id
    except:
        return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_video_frames", type=int, default=1)
    #parser.add_argument("--tokenizer_model_max_length", type=int, default=8192)
    args = parser.parse_args()
    main(args)
