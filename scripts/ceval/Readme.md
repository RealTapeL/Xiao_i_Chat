## 模型效果评测

### 模型生成效果评测
为了全面评估相关模型的性能，建议用户根据自身关注的任务特性，进行针对性的模型测试，以筛选出最适合特定任务需求的模型。[Fastchat Chatbot Arena](https://lmarena.ai/?arena)，推出了模型在线对战平台，可浏览和评测模型回复质量。对战平台提供了胜率、Elo评分等评测指标，并且可以查看两两模型的对战胜率等结果。

## 模型客观效果评测
### C-Eval
[C-Eval](https://github.com/hkust-nlp/ceval) 是一个全面的中文基础模型评估套件。它由 13948 道多项选择题组成，涵盖 52 个不同的学科和 4 个难度级别
在抱抱脸Hugging Face上下载评测数据集，并解压
```
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam.zip -d data
```
### 运行预测脚本
```
cd scripts/ceval
python eval.py \
    --model_path ${model_path} \
    --few_shot False \
    --with_prompt False\
    --n_times 1 \
    --ntrain 5 \
    --do_save_csv False \
    --do_test False \
    --output_dir ${output_path} \
```
