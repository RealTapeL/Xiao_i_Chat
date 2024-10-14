# RAG
## 环境准备
```
cd rag
pip3 install -r requirements.txt
```
## 使用指南
TXT 数据
将需要构建的知识库转化为 Txt 文件放入到 src.data.txt 目录下

JSON 数据
构建 QA 对并生成 JSON 文件（多轮对话），放入到 src.data.json 目录下

数据格式如下
```
[
    {
        "conversation": [
            {
                "input": "",
                "output": ""
            },
            {
                "input": "",
                "output": ""
            }
        ]
    },
] 
```
代码根据准备的数据构建，最终在data文件夹下产生名为 vector_db 的文件夹包含 index.faiss 和 index.pkl。如果已经有 vector DB 则会直接加载对应数据库
## 配置 config 文件
根据需要改写 config.config 文件：
```
# 存放所有 model
model_dir = os.path.join(base_dir, 'model')

# embedding model 路径以及 model name
embedding_path = os.path.join(model_dir, 'embedding_model')
embedding_model_name = 'BAAI/bge-small-zh-v1.5'


# rerank model 路径以及 model name
rerank_path = os.path.join(model_dir, 'rerank_model')
rerank_model_name = 'BAAI/bge-reranker-large'


# select num: 代表rerank 之后选取多少个 documents 进入 LLM
select_num = 3

# retrieval num： 代表从 vector db 中检索多少 documents。（retrieval num 应该大于等于 select num）
retrieval_num = 10

# 智谱 LLM 的 API key。目前 demo 仅支持智谱 AI api 作为最后生成
glm_key = ''

# Prompt template: 定义
prompt_template = """
	你是一个拥有丰富心理学知识的温柔邻家温柔大姐姐艾薇，我有一些心理问题，请你用专业的知识和温柔、可爱、俏皮、的口吻帮我解决，回复中可以穿插一些可爱的Emoji表情符号或者文本符号。\n

	根据下面检索回来的信息，回答问题。

	{content}

	问题：{query}
"""
```
## 本地调用
src/data_processing.py
```
from config.config import (
    embedding_path,
    embedding_model_name,
    doc_dir, qa_dir,
    knowledge_pkl_path,
    data_dir,
    vector_db_dir,
    rerank_path,
    rerank_model_name,
    chunk_size,
    chunk_overlap
)
```
src/pipeline.py
```
#from rag.src.data_processing import Data_process
#from rag.src.config.config import prompt_template 

from data_processing import Data_process
from config.config import prompt_template
```
修改 import 路径之后通过以下 code 执行:
```
cd rag/src
python main.py
```
## 数据集
- 经过清洗的QA对: 每一个QA对作为一个样本进行 embedding
- 经过清洗的对话: 每一个对话作为一个样本进行 embedding
- 经过筛选的TXT文本
    - 直接对TXT文本生成embedding (基于token长度进行切分)
    - 过滤目录等无关信息后对TXT文本生成embedding (基于token长度进行切分)
    - 过滤目录等无关信息后, 对TXT进行语意切分生成embedding
    - 按照目录结构对TXT进行拆分，构架层级关系生成embedding
## 方案细节
### RAG具体流程
- 根据数据集构建 vector DB
- 对用户输入的问题进行 embedding
- 基于 embedding 结果在向量数据库中进行检索
- 对召回数据重排序
- 依据用户问题和召回数据生成最后的结果
