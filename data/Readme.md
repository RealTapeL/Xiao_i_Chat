# 本项目数据集构建
> 具体内容不一一介绍，点进json文件看即可，多看多做是最重要的。数据集会一直更新。
## 指令集构建 —— Alpaca 格式
本文所有程序、脚本与模型仅支持支持Alpaca格式的数据集,正确格式内容如下
```
[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
  }
]
```
在指令监督微调时，`instruction` 列对应的内容会与 `input` 列对应的内容拼接后作为人类指令，即人类指令为 `instruction\ninput`。而 `output` 列对应的内容为模型回答。

如果指定，`system` 列对应的内容将被作为系统提示词。

`history` 列是由多个字符串二元组构成的列表，分别代表历史消息中每轮对话的指令和回答。注意在指令监督微调时，历史消息中的回答内容也会被用于模型学习。

## 单轮对话数据的格式转换
使用以下程序将单轮对话数据集转换成 alpaca 格式
```
import json
import re

# 选择要格式转换的数据集
file_name = "single_turn_dataset_1.json"
#file_name = "single_turn_dataset_2.json"

system_prompt = "如果要添加系统提示词，请放在这里"

with open(f'../{file_name}', 'rt', encoding='utf-8') as file:
    data = json.load(file)

converted_data = [{"instruction": item["prompt"], 
                   "input": "", 
                   "output": item["completion"],
                   "system": system_prompt
                  } for item in data]

for i in range(len(converted_data)):

    # 数据清洗-去掉特殊符号
    if "🐳" in converted_data[i]["output"]:
        converted_data[i]["output"] = converted_data[i]["output"].replace("🐳", "")
        
    # 数据清洗-去掉“你好，我是红烧肉”，会影响大模型的自我认知
    if '好，我是' in converted_data[i]["output"]:
        converted_data[i]["output"] = converted_data[i]["output"].strip()
        intro_pattern = r"^[^\n]+\n"
        converted_data[i]["output"] = re.sub(intro_pattern, "", converted_data[i]["output"]).strip() 

with open(f'./processed/{file_name}', 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)
print(f'./processed/{file_name} Done')
```
## 多轮对话数据的格式转换
使用以下程序将多轮对话转换成 alpaca 格式
```
from tqdm import tqdm
import json

# 选择要格式转换的数据集
file_name = "data.json"
#file_name = "data_pro.json"
#file_name = "multi_turn_dataset_1.json"
#file_name = "multi_turn_dataset_2.json"
#file_name = "aiwei.json"

system_prompt = "如果要添加系统提示词，请放在这里"

with open(f'../{file_name}', 'rt', encoding='utf-8') as file:
    data = json.load(file)

# 遍历原始数据，进行格式转换

# 转换后的数据格式
converted_data = []
for item in tqdm(data):
    conversation = item['conversation']
    history = [(c['input'], c['output']) for c in conversation[:-1]]
    last_item = conversation[-1]
    converted_data.append({
        "instruction": last_item['input'],
        "input": "",
        "output": last_item['output'],
        "system": system_prompt,
        "history": history
    })
    # 将转换后的数据转换为JSON格式
    converted_json = json.dumps(converted_data, ensure_ascii=False, indent=4)

with open(f'./processed/{file_name}', 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)
```
## 角色扮演数据的格式转换
代码同上，根据原数据集是单轮对话还是多轮对话来选择。注意设置各个角色的“system_prompt”。
