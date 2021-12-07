#!/bin/bash

# In this example, show evaluation model on STS2017 mutilingual

# python multilingual_evaluation.py \
#     --model_name_or_path output/make-multilingual-original-T-output-BSL_tuning_output-unsup-simcse-mBert-2021-11-12_04-49-02-S-distilbert-base-multilingual-cased
    
    
# python multilingual_evaluation.py \
#     --model_name_or_path output/making-T-output-BSL_tuning_output-unsup-simcse-mBert-2021-11-12_04-49-02-S-microsoft-Multilingual-MiniLM-L12-H384
    
    
# python multilingual_evaluation.py \
#     --model_name_or_path output/skd-T-output-multilingual_NLI_simcse_distilbert_base_multilingual_cased-S-output-multilingual_NLI_simcse_distilbert_base_multilingual_cased
       
    
python multilingual_evaluation.py \
    --model_name_or_path output/bsl_multilingual_distill/BSL_skd-shuffle-T-sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2-S-microsoft-Multilingual-MiniLM-L12-H384
    
    
# python multilingual_evaluation.py \
#     --model_name_or_path output/making/making-T-sentence-transformers-bert-base-nli-stsb-mean-tokens-S-microsoft-Multilingual-MiniLM-L12-H384
    
# python multilingual_evaluation.py \
#     --model_name_or_path output/making/making-T-sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2-S-microsoft-Multilingual-MiniLM-L12-H384