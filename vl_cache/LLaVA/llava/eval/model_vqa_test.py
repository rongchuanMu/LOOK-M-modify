import os
import sys
sys.path.insert(0, '/home/zxwang/module/llava_16/LLaVA/')
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import pandas as pd
from PIL import Image
import io

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,attn_implementation='eager',torch_dtype='bfloat16')


    ans_file = open(os.path.expanduser(args.answers_file), "w")
    for file in os.listdir(args.questions_folder):
        df_question = pd.read_parquet(os.path.join(args.questions_folder, file))
        # 按列获取内容
        questionIds = df_question['questionId'].tolist()
        quetions = df_question['question'].tolist()
        images_bytes = df_question['image'].tolist()
        images = []
        for i, (img_dict, qs, idx) in enumerate(zip(images_bytes,quetions, questionIds)):
            # 读取和保存图片
            byte_data = img_dict['bytes']
            image = Image.open(io.BytesIO(byte_data))
            image_path = os.path.join(args.image_folder, f"image_{idx}.png")
            if not os.path.exists(image_path):
                image.save(image_path)

            # 进行推理
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_tensor = process_images([image], image_processor, model.config)[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=100,
                    use_cache=True,
                    alpha_sparsity=args.alpha_sparsity)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
            print("questionId:", idx)
            print("question:", qs)
            print("outputs:", outputs)
            exit()
        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/zxwang/huggingface/llava-v1.6-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/zxwang/module/llava_16/dataset/DocVQA/DocVQA/image")
    parser.add_argument("--questions-folder", type=str, default="/home/zxwang/module/llava_16/dataset/DocVQA/DocVQA/test")
    parser.add_argument("--answers-file", type=str, default="/home/zxwang/module/llava_16/dataset/DocVQA/answer_docvqa_test.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--alpha-sparsity", type=int, default=0.9)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
