# !/bin/bash
set -x
set -e
source /opt/conda/bin/activate /share/chaofan/envs/reranker
cd /share/chaofan/code/reranker/finetune/finetune_mistral_fix

torchrun --nproc_per_node 8 \
run.py \
--output_dir ./run_15hard_distill_sep_layer_2e-4 \
--model_name_or_path /share/chaofan/models/Mistral-7B-v0.1 \
--train_data cfli/msmarco_train \
--learning_rate 2e-4 \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--query_max_len 32 \
--passage_max_len 192 \
--train_group_size 16 \
--logging_steps 1 \
--save_steps 100 \
--save_total_limit 50 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed /share/chaofan/code/stage/stage1.json \
--warmup_ratio 0.1 \
--bf16 \
--use_lora True \
--lora_rank 32 \
--lora_alpha 64 \
--loss_type 'only logits' \
--use_flash_attn False \
--target_modules q_proj k_proj v_proj o_proj down_proj up_proj gate_proj linear_head \
--token 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 \
--cache_dir /share/shared_models/ \
--cache_path /share/chaofan/cache/data_cache \
--padding_side right \
--start_layer 4 \
--layer_sep 1 \
--layer_wise True \
--compress_ratios 1 2 4 8 \
--compress_layers 4 8 12 16 20 24 28 \
--train_method distill_fix_layer_teacher

bash /share/chaofan/code/reranker/evaluate/evaluate_mistral_fix/run_sep_2e-4_distill.sh