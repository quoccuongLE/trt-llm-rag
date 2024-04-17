#!/bin/bash

python app.py \
    --trt_engine_path /media/Data1-HDD8/qcuong_le/test/TensorRT-LLM/tmp/llama/13B/trt_engines/weight_only/1-gpu/ \
    --trt_engine_name rank0.engine \
    --tokenizer_dir_path ../trt-llm-rag-windows/model/ \
    --data_dir ./dataset/
