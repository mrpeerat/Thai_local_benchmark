import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import senteval
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, models
from glob import glob

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = params['encoder'](batch)
    return embeddings

pool_mode = 'cls'
mlp_mode = True
model_list = glob(f'output/simcse_original/Simcse_original_*{pool_mode}*MLP{mlp_mode}_wikiFalse_nliTrue')
for model in model_list:
#     sim_cse = SentenceTransformer(model)
    print(f"Model:{model}")
    word_embedding_model = models.Transformer(model, max_seq_length=32)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_mode=pool_mode)
    sim_cse = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    params = {'task_path': 'SentEval/data/', 'usepytorch': True, 'kfold': 10}
    params['encoder'] = sim_cse.encode
    se = senteval.engine.SE(params, batcher, prepare)

    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    results = se.eval(transfer_tasks)

    
    spearman_val = 0
    for key in results.keys():
        print(key)
    #     print(results)
        if key not in ['STSBenchmark','SICKRelatedness']:
            result_temp = results[key]['all']['spearman']['all']
            spearman_val+=result_temp
            print(f"Spearman:{result_temp*100:.2f}")
        else:
            result_temp = results[key]['test']['spearman'].correlation
            spearman_val+=result_temp
            print(f"Spearman:{result_temp*100:.2f}")
    print(f"Avg:{(spearman_val/len(results.keys()))*100:.2f}")
    print(f"*"*50)
    print()