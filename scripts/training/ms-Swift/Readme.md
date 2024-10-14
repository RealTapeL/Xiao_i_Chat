# SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning)
SWIFT支持350+ LLM和100+ MLLM（多模态大模型）的训练(预训练、微调、对齐)、推理、评测和部署。开发者可以直接将我们的框架应用到自己的Research和生产环境中，实现模型训练评测到应用的完整链路。我们除支持了PEFT提供的轻量训练方案外，也提供了一个完整的Adapters库以支持最新的训练技术，如NEFTune、LoRA+、LLaMA-PRO等，这个适配器库可以脱离训练脚本直接使用在自己的自定流程中。
为方便不熟悉深度学习的用户使用，我们提供了一个Gradio的web-ui用于控制训练和推理，并提供了配套的深度学习课程和最佳实践供新手入门。 

现在我们项目使用本项目自定义[数据集](https://github.com/RealTapeL/Xiao_i_Chat/tree/main/data)，并将其转化成合适的Alpaca格式，使用SWIFT进行微调.

## Swift微调框架的安装与使用
### 环境准备
GPU设备: 4090, 3090, A100等显存>24G的显卡均可.

项目Swift微调使用魔搭社区提供的基于英特尔CPU的免费计算资源，使用GPU环境（8核 32GB 显存24G）；

SWIFT在Python环境中运行。请确保您的Python版本高于3.8。

这里我们对实验环境进行安装，其中包含了虚拟环境的创建、ms-swift以及相关依赖的安装。
```
# 设置pip全局镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# 如果你想要使用deepspeed.
pip install deepspeed -U

# 环境对齐 (通常不需要运行. 如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```
### 微调，启动！😻
```
# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main
)

model_type = ModelType.qwen_7b_chat
sft_args = SftArguments(
    model_type=model_type,
    dataset=[f'{DatasetName.blossom_math_zh}#2000'],
    output_dir='output')
result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')
torch.cuda.empty_cache()

infer_args = InferArguments(
    ckpt_dir=best_model_checkpoint,
    load_dataset_config=True)
# merge_lora(infer_args, device_map='cpu')
result = infer_main(infer_args)
torch.cuda.empty_cache()

app_ui_main(infer_args)

```
