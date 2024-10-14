# <p align="center">Xiao_i_Chat-苏州经贸信息技术学院“小i”大模型</p>
本项目开源了基于LLaMA模型的指令精调的“小i”大模型，以进一步促进大模型在中文NLP社区的开放研究。这些模型在原版LLaMA的基础上扩充了特定领域（如：教学、医学、心理等）中文数据集并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。
**“小i”大模型**是一款能够支持 理解用户-支持用户-帮助用户的细分领域大模型，由**LLaMA模型**微调而来得到，欢迎大家star~⭐⭐，你们的支持是团队进步的最大动力！😋

[报告Bug](https://github.com/RealTapeL/Xiao_i_Chat/issues) [提出新想法](https://github.com/RealTapeL/Xiao_i_Chat/issues)

>开发前的配置要求
硬件：支持bf16精度且显存>24G的GPU；内存>16G
软件：Ubuntu22.04，python>=3.9

## 🐙数据集构建
本项目最重要的内容是开源了细分领域的数据集，请参考[数据集构建](https://github.com/RealTapeL/Xiao_i_Chat/tree/main/data)部分

## 🐨微调指南
基于lora（没错，本项目使用的依然是lora微调🤗），请查看[微调步骤](https://github.com/RealTapeL/Xiao_i_Chat/blob/main/scripts/training/Readme_sft.md)

想知道你微调的模型效果如何？请查看[评测指南]（https://github.com/RealTapeL/Xiao_i_Chat/tree/main/scripts/ceval）
### **One more thing~**
基于ms-swift的微调方法，请参考[指南](https://github.com/RealTapeL/Xiao_i_Chat/blob/main/scripts/training/ms-Swift/Readme.md)

**Only Apple can do!😋**

## 🦊如何部署微调的大模型？

基于llama.cpp的部署：请参照[指南](https://github.com/RealTapeL/Xiao_i_Chat/tree/main/scripts/llama.cpp)

基于变形金刚Transformers的部署，请参考[指南](https://github.com/RealTapeL/Xiao_i_Chat/tree/main/scripts/transformers)

## 问题反馈

如有疑问，请在GitHub Issue中提交。礼貌地提出问题，构建和谐的讨论社区。

- 在提交问题之前，请先查看FAQ能否解决问题，同时建议查阅以往的issue是否能解决你的问题。
- 提交问题请使用本项目设置的Issue模板，以帮助快速定位具体问题。

## todo~
设计词表，提升中文字词的覆盖程度，解决因混用词表带来的问题，以期进一步提升模型对中文文本的编解码效率

基于FlashAttention-2的高效注意力机制

基于PI和YaRN的超长上下文扩展技术

人类偏好对齐：通过基于人类反馈的强化学习和奖励模型实验，提升模型传递正确价值观的能力

投机采样加速效果评测、人类偏好对齐（RLHF）版本评测
