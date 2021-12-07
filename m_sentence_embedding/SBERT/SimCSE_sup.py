"""
This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

SimCSE will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

Usage:
python train_simcse_from_file.py path/to/sentences.txt

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
import logging
from datetime import datetime
import gzip
import sys
import tqdm
import csv
import zipfile
import io
from torch import nn
import pandas as pd

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# model_alls = ['nreimers/TinyBERT_L-4_H-312_v2', 'nreimers/MiniLM-L3-H384-uncased', 'distilbert-base-cased', 'microsoft/MiniLM-L12-H384-uncased', 'nreimers/MiniLM-L6-H384-uncased', 'nreimers/TinyBERT_L-6_H-768_v2', 'nreimers/BERT-Tiny_L-2_H-128_A-2', 'nreimers/BERT-Small-L-4_H-512_A-8', 'google/mobilebert-uncased', 'nreimers/BERT-Mini_L-4_H-256_A-4', 'nreimers/albert-small-v2']

# model_alls = ['roberta-large']
model_alls = ['nreimers/albert-small-v2','roberta-base']

dev_inference = 32
max_seq_length = 32
num_epochs = 3
training_data = '../SimCSE/data/nli_for_simcse.csv'
sts_dataset_path = '../BSL/training/data/stsbenchmark.tsv.gz'
nli_dataset_path = '../BSL/training/data/back_translated_nli.txt'

pooling_mode = 'cls' # cls,max,mean
mlp_mode = True # True, False

for model_now in model_alls:
    if model_now == 'roberta-large':
        train_batch_size = 256
        learning_rate = 1e-5
    else:
        train_batch_size = 512
        learning_rate = 5e-5
    model_name = model_now
    
    
    model_save_path = f"output/simcse_original/Simcse_supervised_supervised_{model_name.replace('-','_').replace('/','_')}_{pooling_mode}_MLP{mlp_mode}"
    print(f"Path:{model_save_path}")

    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_mode=pooling_mode)
    if mlp_mode:
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=pooling_model.get_sentence_embedding_dimension(), activation_function=nn.Tanh())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    #Read STSbenchmark dataset and use it as development set
    logging.info("Read STSbenchmark dev dataset")
    dev_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'dev':
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=dev_inference, name='sts-dev')

    train_data = []
    train_sentences = pd.read_csv(training_data).values.tolist()
    train_data += [InputExample(texts=[s[0], s[1], s[2]]) for s in train_sentences]

    try:
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size, drop_last=True)


        logging.info("Train sentences: {}".format(len(train_data)))


        # We train our model using the MultipleNegativesRankingLoss
        train_loss = losses.MultipleNegativesRankingLoss(model)
        evaluation_steps = 125 #Evaluate every 10% of the data
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                evaluator=dev_evaluator,
                evaluation_steps=evaluation_steps,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': learning_rate},
                show_progress_bar=True,
                output_path=model_save_path,
                use_amp=True  # Set to True, if your GPU supports FP16 cores
                )
    except:
        train_batch_size /= 2
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size, drop_last=True)
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                evaluator=dev_evaluator,
                evaluation_steps=evaluation_steps,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': learning_rate},
                show_progress_bar=True,
                output_path=model_save_path,
                use_amp=True  # Set to True, if your GPU supports FP16 cores
                )
