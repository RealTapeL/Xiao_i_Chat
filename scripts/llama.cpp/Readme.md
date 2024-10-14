## [llama.cpp](https://github.com/ggerganov/llama.cpp)部署

运行前请确保环境：

系统应有make（MacOS/Linux自带）或cmake（Windows需自行安装）编译工具
建议使用Python 3.10以上编译和运行该工具

Step 1: 编译llama.cpp

llama.cpp在2024年4月30日对编译做出重大改动，请务必拉取最新仓库进行编译！
```
$ git clone https://github.com/ggerganov/llama.cpp
```
对llama.cpp项目进行编译，生成./main（用于推理）和./quantize（用于量化）二进制文件
```
$ make
```
Windows/Linux用户如需启用GPU推理，则推荐使用cuda编译
```
$ make LLAMA_CUDA=1
```

Step 2:生成量化版本(gguf格式)模型

目前llama.cpp已支持.safetensors文件以及Hugging Face格式.bin转换为FP16的GGUF格式
```
$ python convert_hf_to_gguf.py your_model_name
$ ./llama-quantize your_model_name/ggml-model-f16.gguf your_model_name/ggml-model-q4_0.gguf q4_0
```

Step 3: 加载并启动模型
## Linux、macOS等系统：
#### 对话模式（允许与模型持续交互）
```
./llama-cli -m /home/fw/results8.24/earthllm826/earth.gguf -p "You are a helpful assistant" -cnv --chat-template chatml
```
## Windows系统：
#### 对话模式（允许与模型持续交互）
```
./llama-cli.exe -m /home/fw/results8.24/earthllm826/earth.gguf -p "You are a helpful assistant" -cnv --chat-template chatml
```
