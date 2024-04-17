#!/bin/bash

# python build.py \
#   --model_dir ../Llama-2-13b-chat-hf \
#   --quant_ckpt_path <path to model.pt> \
#   --dtype float16 \
#   --use_gpt_attention_plugin float16 \
#   --use_gemm_plugin float16 \
#   --use_weight_only \
#   --weight_only_precision int4_awq \
#   --per_group \
#   --enable_context_fmha \
#   --max_batch_size 1 \
#   --max_input_len 3000 \
#   --max_output_len 1024 \
#   --output_dir <TRT engine folder>


# python convert_checkpoint.py --model_dir ../Llama-2-13b-chat-hf \
#                               --output_dir ./tllm_checkpoint_1gpu_fp16_wq \
#                               --dtype float16 \
#                               --use_weight_only \
#                               --weight_only_precision int4_awq \
#                               --per_group \

python examples/quantization/quantize.py \
  --model_dir ../Llama-2-13b-chat-hf \
  --dtype float16 \
  --qformat int4_awq \
  --awq_block_size 128 \
  --output_dir ./quantized_int4-awq \
  --calib_size 32

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16_wq \
            --output_dir ./tmp/llama/7B/trt_engines/weight_only/1-gpu/ \
            --use_gemm_plugin float16