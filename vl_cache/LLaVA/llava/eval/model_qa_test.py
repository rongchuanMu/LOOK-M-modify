import os
import sys
# sys.path.append('/home/zxwang/module/llava_16/LLaVA')

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
import torch

import json
from tqdm import tqdm
import shortuuid
import pandas as pd
from PIL import Image
import io

from llava.conversation import default_conversation
from llava.utils import disable_torch_init


@torch.inference_mode()
def eval_model(model_name, questions_folder, answers_file, image_folder):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16).cuda()


    ans_file = open(os.path.expanduser(answers_file), "w")
    for file in os.listdir(questions_folder):
        df_question = pd.read_parquet(os.path.join(questions_folder, file))
        # 按列获取内容
        questionIds = df_question['questionId'].tolist()
        quetions = df_question['question'].tolist()
        images_bytes = df_question['image'].tolist()
        images = []
        for i, (img_dict, qs, idx) in enumerate(zip(images_bytes,quetions, questionIds)):
            # 读取和保存图片
            byte_data = img_dict['bytes']
            image = Image.open(io.BytesIO(byte_data))
            image_path = os.path.join(image_folder, f"image_{idx}.png")
            if not os.path.exists(image_path):
                image.save(image_path)

            # 进行推理
            conv = default_conversation.copy()
            conv.append_message(conv.roles[0], qs)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])
            input_ids = torch.as_tensor(inputs.input_ids).cuda()
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                use_cache=True,
                temperature=0.4,
                max_new_tokens=1024,)
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            try:
                index = outputs.index(conv.sep, len(prompt))
            except ValueError:
                outputs += conv.sep
                index = outputs.index(conv.sep, len(prompt))

            outputs = outputs[len(prompt) + len(conv.roles[1]) + 2:index].strip()
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                        "text": outputs,
                                        "answer_id": ans_id,
                                        "model_id": model_name,
                                        "metadata": {}}) + "\n")
            ans_file.flush()
        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/home/zxwang/huggingface/llava-v1.6-mistral-7b")
    parser.add_argument("--question-folder", type=str, default="/home/zxwang/module/llava_16/dataset/DocVQA/DocVQA/test")
    parser.add_argument("--answers-file", type=str, default="/home/zxwang/module/llava_16/dataset/DocVQA/answer_docvqa_test.jsonl")
    parser.add_argument("--image-folder", type=str, default="/home/zxwang/module/llava_16/dataset/DocVQA/DocVQA/image")
    args = parser.parse_args()

    eval_model(args.model_name, args.question_folder, args.answers_file, args.image_folder)
