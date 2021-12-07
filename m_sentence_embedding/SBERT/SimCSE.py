"""
This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

SimCSE will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

Usage:
python train_simcse_from_file.py path/to/sentences.txt

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# model_alls = ['nreimers/TinyBERT_L-4_H-312_v2', 'nreimers/MiniLM-L3-H384-uncased', 'bert-base-uncased', 'distilbert-base-cased', 'microsoft/MiniLM-L12-H384-uncased', 'nreimers/MiniLM-L6-H384-uncased', 'nreimers/TinyBERT_L-6_H-768_v2', 'nreimers/BERT-Tiny_L-2_H-128_A-2', 'nreimers/BERT-Small-L-4_H-512_A-8', 'google/mobilebert-uncased', 'nreimers/BERT-Mini_L-4_H-256_A-4', 'nreimers/albert-small-v2']

model_alls = ['roberta-large']

max_seq_length = 32
num_epochs = 1
training_data = '../SimCSE/data/wiki1m_for_simcse.txt'
sts_dataset_path = '../BSL/training/data/stsbenchmark.tsv.gz'
nli_dataset_path = '../BSL/training/data/back_translated_nli.txt'

pooling_mode = 'cls' # cls,max,mean
mlp_mode = True # True, False
wiki_dataset = False
nli_dataset = True

for model_now in model_alls:
    if 'roberta' in model_now:
        if model_now == 'roberta-base':
            train_batch_size = 512
            learning_rate = 1e-5
        else:
            train_batch_size = 256
            learning_rate = 3e-5
    else:
        train_batch_size = 64
        learning_rate = 3e-5
    model_name = model_now
    
    
    model_save_path = f"output/simcse_original/Simcse_original_{model_name.replace('-','_').replace('/','_')}_{pooling_mode}_MLP{mlp_mode}_wiki{wiki_dataset}_nli{nli_dataset}"
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

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

    train_data = []
    if wiki_dataset:
        train_sentences = open(training_data).readlines()
        train_data += [InputExample(texts=[s, s]) for s in train_sentences]

    nli_data = []
    if nli_dataset:
        train_sentences = open(nli_dataset_path).readlines()
        train_sentences_split = [s.strip().split('\t')[0] for s in train_sentences]
        train_data += [InputExample(texts=[s, s]) for s in train_sentences_split]
    train_dataloader = DataLoader(train_data+nli_data, shuffle=True, batch_size=train_batch_size, drop_last=True)


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
