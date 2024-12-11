from workers.baseworker import *
import sys
from PIL import Image
import torch
from transformers import AutoTokenizer
import os
sys.path.insert(0, '/home/rcmu/read_papers/LOOK-M-main/LLaVA-mix_merge_v1')

######################## Multi-image application ########################

class LLaVA(BaseWorker):
    def init_components(self, config):
        from llava.model.builder import load_pretrained_model
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.model.kv_token_merge.modify_llama import H2OLlamaAttention_drop, \
                                                            WeightedLlamaAttention_drop, \
                                                            PivotMergeLlamaAttention_drop, \
                                                            TextH2OLlamaAttention_drop, \
                                                            TextWeightedLlamaAttention_drop, \
                                                            TextPivotLlamaAttention_drop, \
                                                            PoolingWindowLlamaAttention_drop, \
                                                            AVGMergeLlamaAttention_drop, \
                                                            MeanH2OLlamaAttention_drop, \
                                                            TextAVGMergeLlamaAttention_drop
        self.tokenizer, self.model, self.processor, context_len = load_pretrained_model(
            model_path=config.model_dir,
            model_base=None,
            model_name=config.model_dir,
            device_map='cuda',
            kv_mode=config.kv_mode,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
        )
        self.kv_mode = config.kv_mode
        if getattr(self.model.config, 'mm_use_im_start_end', False):
            self.single_img_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            self.single_img_tokens = DEFAULT_IMAGE_TOKEN

        self.conv_temp = conv_templates["llava_llama_2"]
        stop_str = self.conv_temp.sep if self.conv_temp.sep_style != SeparatorStyle.TWO else self.conv_temp.sep2
        self.keywords = [stop_str]

        self.model.eval()
        choices=["origin", "h2o", "weighted_merge", "pivot_merge", "text_prior_h2o", "text_prior_weighted_merge", "text_prior_pivot_merge"]
        self.TAGET_MODULE = {
            "llama": None,
            "origin": None,
            "h2o": H2OLlamaAttention_drop,
            "weighted_merge": WeightedLlamaAttention_drop,
            "pivot_merge": PivotMergeLlamaAttention_drop,
            "text_prior_h2o": TextH2OLlamaAttention_drop,
            "text_prior_weighted_merge": TextWeightedLlamaAttention_drop,
            "text_prior_pivot_merge": TextPivotLlamaAttention_drop,
            "snapkv": PoolingWindowLlamaAttention_drop,
            "avg_merge": AVGMergeLlamaAttention_drop,
            "mean_h2o": MeanH2OLlamaAttention_drop,
            "text_prior_avg_merge": TextAVGMergeLlamaAttention_drop,
        }

    def clean_cache(self):
        if self.kv_mode == "origin":
            return
        for name, m in self.model.named_modules():
            if isinstance(m, self.TAGET_MODULE[self.kv_mode]):
                m._clean_cache()

    def forward(self, questions, image_paths, device, gen_kwargs):
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria

        answers = []
        for question,images_path in zip(questions, image_paths):
            conv = self.conv_temp.copy()

            # Multi-image
            if images_path == []:
                image_tensor = None
            else:
                image_tensor = process_images(   # 最终形如[5, 3, 336, 336]
                    [Image.open(image_path).convert('RGB') for image_path in images_path],
                    self.processor, self.model.config
                ).to(device)

            question = question.replace('<ImageHere><ImageHere>', '<ImageHere>\n<ImageHere>\n') # NOTE: handle the special cases in CLEVR-Change dataset
            input_prompt = question.replace('<ImageHere>', self.single_img_tokens)

            conv.append_message(conv.roles[0], input_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt=prompt, 
                tokenizer=self.tokenizer, 
                image_token_index=IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    use_cache=True,
                    stopping_criteria=[KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)],
                    **gen_kwargs
                )
            self.clean_cache()
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            answers.append(answer)

        return answers

class InternVL(BaseWorker):
    def init_components(self, config):
        sys.path.insert(0, '/users/PAS2473/brucewan666/Faster-LLaVA/InternVL/internvl_chat_llava')
        from llava.model.builder import load_pretrained_model
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.model.kv_token_merge.modify_llama import H2OLlamaAttention_drop, \
                                                            WeightedLlamaAttention_drop, \
                                                            PivotMergeLlamaAttention_drop, \
                                                            TextH2OLlamaAttention_drop, \
                                                            TextWeightedLlamaAttention_drop, \
                                                            TextPivotLlamaAttention_drop, \
                                                            PoolingWindowLlamaAttention_drop, \
                                                            AVGMergeLlamaAttention_drop, \
                                                            MeanH2OLlamaAttention_drop
        self.tokenizer, self.model, self.processor, context_len = load_pretrained_model(
            model_path=config.model_dir,
            model_base=None,
            model_name=config.model_dir,
            device_map='cuda',
            kv_mode=config.kv_mode,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
        )
        self.kv_mode = config.kv_mode
        if getattr(self.model.config, 'mm_use_im_start_end', False):
            self.single_img_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            self.single_img_tokens = DEFAULT_IMAGE_TOKEN

        self.conv_temp = conv_templates["internlm2-chat"]
        stop_str = self.conv_temp.sep if self.conv_temp.sep_style != SeparatorStyle.TWO else self.conv_temp.sep2
        self.keywords = [stop_str]

        self.model.eval()
        choices=["origin", "h2o", "weighted_merge", "pivot_merge", "text_prior_h2o", "text_prior_weighted_merge", "text_prior_pivot_merge"]
        self.TAGET_MODULE = {
            "llama": None,
            "origin": None,
            "h2o": H2OLlamaAttention_drop,
            "weighted_merge": WeightedLlamaAttention_drop,
            "pivot_merge": PivotMergeLlamaAttention_drop,
            "text_prior_h2o": TextH2OLlamaAttention_drop,
            "text_prior_weighted_merge": TextWeightedLlamaAttention_drop,
            "text_prior_pivot_merge": TextPivotLlamaAttention_drop,
            "snapkv": PoolingWindowLlamaAttention_drop,
            "avg_merge": AVGMergeLlamaAttention_drop,
            "mean_h2o": MeanH2OLlamaAttention_drop,
        }

    def clean_cache(self):
        if self.kv_mode == "origin":
            return
        for name, m in self.model.module.named_modules():
            if isinstance(m, self.TAGET_MODULE[self.kv_mode]):
                m._clean_cache()

    def forward(self, questions, image_paths, device, gen_kwargs):
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria

        answers = []
        for question,images_path in zip(questions, image_paths):
            conv = self.conv_temp.copy()

            # Multi-image
            if images_path == []:
                image_tensor = None
            else:
                image_tensor = process_images(
                    [Image.open(image_path).convert('RGB') for image_path in images_path],
                    self.processor, self.model.config
                ).to(device)

            question = question.replace('<ImageHere><ImageHere>', '<ImageHere>\n<ImageHere>\n') # NOTE: handle the special cases in CLEVR-Change dataset
            input_prompt = question.replace('<ImageHere>', self.single_img_tokens)

            conv.append_message(conv.roles[0], input_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt=prompt, 
                tokenizer=self.tokenizer, 
                image_token_index=IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    use_cache=True,
                    stopping_criteria=[KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)],
                    **gen_kwargs
                )
            self.clean_cache()
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            answers.append(answer)
        return answers

class MobileVLM(BaseWorker):
    def init_components(self, config):
        sys.path.insert(0, '/users/PAS2473/brucewan666/Faster-LLaVA/MobileVLM/mobilevlm')
        
        from model.mobilevlm import load_pretrained_model
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.model.kv_token_merge.modify_llama import H2OLlamaAttention_drop, \
                                                            WeightedLlamaAttention_drop, \
                                                            PivotMergeLlamaAttention_drop, \
                                                            TextH2OLlamaAttention_drop, \
                                                            TextWeightedLlamaAttention_drop, \
                                                            TextPivotLlamaAttention_drop, \
                                                            PoolingWindowLlamaAttention_drop, \
                                                            AVGMergeLlamaAttention_drop, \
                                                            MeanH2OLlamaAttention_drop
        self.tokenizer, self.model, self.processor, context_len = load_pretrained_model(
            model_path=config.model_dir,
            device_map='cuda',
            kv_mode=config.kv_mode,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
        )
        self.kv_mode = config.kv_mode
        if getattr(self.model.config, 'mm_use_im_start_end', False):
            self.single_img_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            self.single_img_tokens = DEFAULT_IMAGE_TOKEN

        self.conv_temp = conv_templates["v1"]
        stop_str = self.conv_temp.sep if self.conv_temp.sep_style != SeparatorStyle.TWO else self.conv_temp.sep2
        self.keywords = [stop_str]

        self.model.eval()
        choices=["origin", "h2o", "weighted_merge", "pivot_merge", "text_prior_h2o", "text_prior_weighted_merge", "text_prior_pivot_merge"]
        self.TAGET_MODULE = {
            "llama": None,
            "origin": None,
            "h2o": H2OLlamaAttention_drop,
            "weighted_merge": WeightedLlamaAttention_drop,
            "pivot_merge": PivotMergeLlamaAttention_drop,
            "text_prior_h2o": TextH2OLlamaAttention_drop,
            "text_prior_weighted_merge": TextWeightedLlamaAttention_drop,
            "text_prior_pivot_merge": TextPivotLlamaAttention_drop,
            "snapkv": PoolingWindowLlamaAttention_drop,
            "avg_merge": AVGMergeLlamaAttention_drop,
            "mean_h2o": MeanH2OLlamaAttention_drop,
        }

    def clean_cache(self):
        if self.kv_mode == "origin":
            return
        for name, m in self.model.module.named_modules():
            if isinstance(m, self.TAGET_MODULE[self.kv_mode]):
                m._clean_cache()

    def forward(self, questions, image_paths, device, gen_kwargs):
        sys.path.insert(0, '/users/PAS2473/brucewan666/Faster-LLaVA/MobileVLM')
        from mobilevlm.constants import IMAGE_TOKEN_INDEX
        from mobilevlm.utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria

        answers = []
        for question,images_path in zip(questions, image_paths):
            conv = self.conv_temp.copy()

            # Multi-image
            if images_path == []:
                image_tensor = None
            else:
                image_tensor = process_images(
                    [Image.open(image_path).convert('RGB') for image_path in images_path],
                    self.processor, self.model.config
                ).to(device)

            question = question.replace('<ImageHere><ImageHere>', '<ImageHere>\n<ImageHere>\n') # NOTE: handle the special cases in CLEVR-Change dataset
            input_prompt = question.replace('<ImageHere>', self.single_img_tokens)

            conv.append_message(conv.roles[0], input_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt=prompt, 
                tokenizer=self.tokenizer, 
                image_token_index=IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    use_cache=True,
                    stopping_criteria=[KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)],
                    **gen_kwargs
                )
            self.clean_cache()
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            answers.append(answer)
        return answers


class LLaVA_OV(BaseWorker):
    def init_components(self, config):
        from llava.model.builder import load_pretrained_model
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.model.kv_token_merge.modify_llama import H2OLlamaAttention_drop, \
                                                            WeightedLlamaAttention_drop, \
                                                            PivotMergeLlamaAttention_drop, \
                                                            TextH2OLlamaAttention_drop, \
                                                            TextWeightedLlamaAttention_drop, \
                                                            TextPivotLlamaAttention_drop, \
                                                            PoolingWindowLlamaAttention_drop, \
                                                            AVGMergeLlamaAttention_drop, \
                                                            MeanH2OLlamaAttention_drop, \
                                                            TextAVGMergeLlamaAttention_drop
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        # self.tokenizer, self.model, self.processor, context_len = load_pretrained_model(
        #     model_path=config.model_dir,
        #     model_base=None,
        #     model_name=config.model_dir,
        #     device_map='cuda',
        #     kv_mode=config.kv_mode,
        #     hh_ratio=config.hh_ratio,
        #     recent_ratio=config.recent_ratio,
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        self.model =  LlavaQwenForCausalLM.from_pretrained(config.model_dir, low_cpu_mem_usage=True, attn_implementation="eager", **kwargs)
        """直接粘过来了"""
        is_multimodal = False

        if "llava" in config.model_dir.lower() or is_multimodal:
            mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
            if mm_use_im_patch_token:
                self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))

            vision_tower = self.model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model(device_map=kwargs["device_map"])
            if kwargs['device_map'] != "auto":
                vision_tower.to(device="cuda", dtype=torch.float16)
            self.image_processor = vision_tower.image_processor

        if hasattr(self.model.config, "max_sequence_length"):
            self.context_len = self.model.config.max_sequence_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.context_len = self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "tokenizer_model_max_length"):
            self.context_len = self.model.config.tokenizer_model_max_length
        else:
            self.context_len = 2048
        
        self.kv_mode = config.kv_mode

        if self.kv_mode != "origin":  # 目前只做了origin的
           raise NotImplementedError("ov模型现在只实现了origin模式")
        
        if getattr(self.model.config, 'mm_use_im_start_end', False):
            self.single_img_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            self.single_img_tokens = DEFAULT_IMAGE_TOKEN

        self.conv_temp = conv_templates["qwen_1_5"]
        stop_str = self.conv_temp.sep if self.conv_temp.sep_style != SeparatorStyle.TWO else self.conv_temp.sep2
        self.keywords = [stop_str]

        self.model.eval()
        choices=["origin", "h2o", "weighted_merge", "pivot_merge", "text_prior_h2o", "text_prior_weighted_merge", "text_prior_pivot_merge"]
        self.TAGET_MODULE = {
            "llama": None,
            "origin": None,
            "h2o": H2OLlamaAttention_drop,
            "weighted_merge": WeightedLlamaAttention_drop,
            "pivot_merge": PivotMergeLlamaAttention_drop,
            "text_prior_h2o": TextH2OLlamaAttention_drop,
            "text_prior_weighted_merge": TextWeightedLlamaAttention_drop,
            "text_prior_pivot_merge": TextPivotLlamaAttention_drop,
            "snapkv": PoolingWindowLlamaAttention_drop,
            "avg_merge": AVGMergeLlamaAttention_drop,
            "mean_h2o": MeanH2OLlamaAttention_drop,
            "text_prior_avg_merge": TextAVGMergeLlamaAttention_drop,
        }

    def clean_cache(self):
        if self.kv_mode == "origin":
            return
        for name, m in self.model.named_modules():
            if isinstance(m, self.TAGET_MODULE[self.kv_mode]):
                m._clean_cache()

    def forward(self, questions, image_paths, device, gen_kwargs):
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria

        answers = []
        for question,images_path in zip(questions, image_paths):
            conv = self.conv_temp.copy()

            # Multi-image
            if images_path == []:
                image_tensor = None
            else:
                image_tensor = process_images(   # 最终形如[5, 3, 336, 336] 或 [5, 2, 3, 384, 384] 或 一个list，每个元素都是一个[n, 3, 384, 384]而且不同元素的n不同
                    [Image.open(image_path).convert('RGB') for image_path in images_path],
                    self.image_processor, self.model.config
                ).to(device)

            question = question.replace('<ImageHere><ImageHere>', '<ImageHere>\n<ImageHere>\n') # NOTE: handle the special cases in CLEVR-Change dataset
            input_prompt = question.replace('<ImageHere>', self.single_img_tokens)

            conv.append_message(conv.roles[0], input_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(   # 除了图片的tokenizer成id，图片位置用-200占位。本质上就是[bsz, seq_len]。尽管可能不能叫做seq_len，因为图片只是占位，还没有“弹开”
                prompt=prompt, 
                tokenizer=self.tokenizer, 
                image_token_index=IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    use_cache=True,
                    stopping_criteria=[KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)],
                    **gen_kwargs
                )
            self.clean_cache()
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            answers.append(answer)

        return answers