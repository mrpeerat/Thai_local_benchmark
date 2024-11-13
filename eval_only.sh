#!/bin/bash
# MODEL_NAME='/home/peerat/LLaMA-Factory/gemma2-9b-wangchanx' # airesearch/LLaMa3-8b-WangchanX-sft-Demo, google/gemma-7b-it
# echo Eval on ${MODEL_NAME}
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8 python evaluation/main_nlu_prompt_batch.py tha ${MODEL_NAME} 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation/main_nlg_prompt_batch.py tha ${MODEL_NAME} 0 1

MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct' # airesearch/LLaMa3-8b-WangchanX-sft-Demo, google/gemma-7b-it
# echo Eval on ${MODEL_NAME}
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8 python evaluation/main_nlu_prompt_batch.py tha ${MODEL_NAME} 4
python evaluation/main_local_prompt_batch.py center ${MODEL_NAME} 0 1
python evaluation/main_local_prompt_batch.py south ${MODEL_NAME} 0 1
python evaluation/main_local_prompt_batch.py north ${MODEL_NAME} 0 1
python evaluation/main_local_prompt_batch.py east ${MODEL_NAME} 0 1

MODEL_NAME='meta-llama/Llama-3.1-70B-Instruct' # airesearch/LLaMa3-8b-WangchanX-sft-Demo, google/gemma-7b-it
# echo Eval on ${MODEL_NAME}
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8 python evaluation/main_nlu_prompt_batch.py tha ${MODEL_NAME} 4
python evaluation/main_local_prompt_batch.py center ${MODEL_NAME} 0 1
python evaluation/main_local_prompt_batch.py south ${MODEL_NAME} 0 1
python evaluation/main_local_prompt_batch.py north ${MODEL_NAME} 0 1
python evaluation/main_local_prompt_batch.py east ${MODEL_NAME} 0 1