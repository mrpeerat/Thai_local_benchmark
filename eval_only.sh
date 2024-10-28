#!/bin/bash
# MODEL_NAME='/home/peerat/LLaMA-Factory/gemma2-9b-wangchanx' # airesearch/LLaMa3-8b-WangchanX-sft-Demo, google/gemma-7b-it
# echo Eval on ${MODEL_NAME}
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8 python evaluation/main_nlu_prompt_batch.py tha ${MODEL_NAME} 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation/main_nlg_prompt_batch.py tha ${MODEL_NAME} 0 1

MODEL_NAME='google/gemma-2-2b-it' # airesearch/LLaMa3-8b-WangchanX-sft-Demo, google/gemma-7b-it
# echo Eval on ${MODEL_NAME}
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8 python evaluation/main_nlu_prompt_batch.py tha ${MODEL_NAME} 4
CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation/main_local_prompt_batch.py esan ${MODEL_NAME} 0 1

# MODEL_NAME='/home/peerat/LLaMA-Factory/gemma-7b-wangchanx' # airesearch/LLaMa3-8b-WangchanX-sft-Demo, google/gemma-7b-it
# echo Eval on ${MODEL_NAME}
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8 python evaluation/main_nlu_prompt_batch.py tha ${MODEL_NAME} 4
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python evaluation/main_nlg_prompt_batch.py tha ${MODEL_NAME} 0 1
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8 python evaluation/main_llm_judge_batch.py ${MODEL_NAME} --data ThaiLLM-Leaderboard/mt-bench-thai

# MODEL_NAME='google/gemma-7b-it' # airesearch/LLaMa3-8b-WangchanX-sft-Demo, google/gemma-7b-it
# echo Eval on ${MODEL_NAME}
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8 python evaluation/main_nlu_prompt_batch.py tha ${MODEL_NAME} 4
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python evaluation/main_nlg_prompt_batch.py tha ${MODEL_NAME} 0 1