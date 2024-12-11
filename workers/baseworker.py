from typing import Any


class BaseWorker():
    def __init__(self, config) -> None:
        self.init_components(config)
        self.gen_kwargs = config.get('gen_kwargs', {})
        self.model_id = config.model_name

    def init_components(self) -> None:  # 等待子类实现
        '''
        Initialize model and processor, and anything needed in forward
        '''
        raise NotImplementedError

    @classmethod
    def from_config(cls, **kwargs):  # 从配置创建类实例  cls默认为 <class 'workers.model_workers.LLaVA'>
        return cls(**kwargs)

    def forward(self, questions: list[str], image_paths: list[str, list], device, gen_kwargs) -> list[str]:  # 根据问题和图片路径生成回答，等待子类实现

        raise NotImplementedError

    def __call__(self, device, **kwargs: Any) -> Any:
        for k in ['question', 'image_path']:
            assert k in kwargs, f'the key {k} is missing'  # kwargs就是传入的数据，包括question、image_path、gt_response、sample_id
        questions = kwargs['question']
        image_paths = kwargs['image_path']
        answers = self.forward(
            questions=questions,
            image_paths=image_paths,
            device=device,
            gen_kwargs=self.gen_kwargs,
        )
        outputs = self.collate_batch_for_output(kwargs, answers=answers, prompts=questions)
        return outputs

    def collate_batch_for_output(self, batch, answers, prompts):  # 将模型的输出整理成一个字典

        ret = []
        len_batch = len(batch['id'])
        assert len(answers) == len(prompts) == len_batch

        for i in range(len_batch):
            new = {
                'sample_id': batch['id'][i], # modify the key
                'image': batch['image_path'][i],
                **{
                    k: v[i]
                    for k, v in batch.items() if k not in ('id', 'image_path')
                },
                'gen_model_id': self.model_id,
                'pred_response': answers[i],
                'gen_kwargs': dict(self.gen_kwargs), # omegaconf -> dict
            }

            ret.append(new)

        return ret
