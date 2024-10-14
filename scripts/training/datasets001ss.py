from tqdm import tqdm
import json

# 选择要格式转换的数据集
file_name = "/home/ps/llm-train-and-val/data/Emo/multi_turn_dataset_1.json"
file_name = "/home/ps/llm-train-and-val/data/Emo/scientist.json"
file_name = "/home/ps/llm-train-and-val/data/Emo/self_cognition_EmoLLM.json"
file_name = "/home/ps/llm-train-and-val/data/Emo/tiangou.json"
file_name = "/home/ps/llm-train-and-val/data/Emo/aiwei.json"
file_name = "/home/ps/llm-train-and-val/data/Emo/data.json"

with open(f'{file_name}', 'rt', encoding='utf-8') as file:
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
        #"system": system_prompt,
        #"history": history
    })
    # 将转换后的数据转换为JSON格式
    converted_json = json.dumps(converted_data, ensure_ascii=False, indent=4)

with open(f'{file_name}', 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)