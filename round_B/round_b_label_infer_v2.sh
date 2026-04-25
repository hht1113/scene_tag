python scripts/distillation/round_b_label_infer_v2.py \
    --model /mnt/pfs/qwen3/Qwen3-8B \
    --input scripts/distillation/round_a_results_v2.jsonl \
    --output scripts/distillation/round_b_label_results_v5.json \
    --prompt scripts/distillation/round_b_prompt_multilabel.txt \
    --tag-tree scripts/distillation/tag_definitions_3views.txt \
    --batch-size 32 \
    --max-model-len 20000 \
    --temperature 0.2