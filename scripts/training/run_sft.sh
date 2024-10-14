lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=path/to/hf/meta-llama-3-8b/or/llama-3-chinese-8b/dir/or/model_id
tokenizer_name_or_path=${pretrained_model}
dataset_dir=path/to/sft/data/dir
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8
max_seq_length=512
output_dir=output_dir
validation_file=validation_file_name

torchrun --nnodes 1 --nproc_per_node 1 run_clm_sft_with_peft.py \
    --model_name_or_path /home/ps/llm-train-and-val/models/medical/medical_hf \
    --tokenizer_name_or_path /home/ps/llm-train-and-val/models/medical/medical_hf \
    --dataset_dir /home/ps/llm-train-and-val/data/self \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --low_cpu_mem_usage \
    --do_eval \
    --seed $RANDOM \
    --bf16 \
    --num_train_epochs 5 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --max_seq_length 512 \
    --output_dir /home/ps/llm-train-and-val/models/medical/lora_self \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 64 \
    --lora_alpha 128 \
    --trainable "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"  \
    --lora_dropout 0.05 \
    --modules_to_save None \
    --torch_dtype bfloat16 \
    --validation_file /home/ps/llm-train-and-val/data/ruozhiba_qa2449_gpt4o.json \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False
