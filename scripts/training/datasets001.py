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
