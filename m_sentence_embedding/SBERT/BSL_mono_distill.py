import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction
from sentence_transformers.datasets import ParallelSentencesDataset
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import numpy as np
import zipfile
import io
import tqdm
from glob import glob
from torch import nn

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
										datefmt='%Y-%m-%d %H:%M:%S',
										level=logging.INFO,
										handlers=[LoggingHandler()])
#### /print debug information to stdout


sts_dataset_path = '../BSL/training/data/stsbenchmark.tsv.gz'
nli_dataset_path = '../BSL/training/data/back_translated_nli.txt'

source_languages = ['en']                     
target_languages = ['en', 'de', 'es', 'fr', 'ar', 'tr']    

# ../../SBERT/output/multilingual_NLI_simcse_bert_base_multilingual_cased
teacher_model_name = 'princeton-nlp/unsup-simcse-roberta-large'   #output/BSL_tuning_output-unsup-simcse-mBert-2021-11-12_04-49-02
student_model_name = 'roberta-base'       

# nreimers/BERT-Tiny_L-2_H-128_A-2, nreimers/BERT-Mini_L-4_H-256_A-4, nreimers/albert-small-v2
# nreimers/TinyBERT_L-4_H-312_v2, nreimers/MiniLM-L3-H384-uncased, nreimers/MiniLM-L6-H384-uncased

# nreimers/BERT-Small-L-4_H-512_A-8, microsoft/MiniLM-L12-H384-uncased, distilbert-base-cased

# nreimers/TinyBERT_L-6_H-768_v2, bert-base-uncased, roberta-base, roberta-large



train_batch_size = 64
num_epochs = 20
max_seq_length = 128
moving_average_decay = 0.999
inference_batch_size = 32
unsupervised_check = True

model_save_path = f"output/bsl_mono_distill/BSL_making-T-{teacher_model_name.replace('/','-')}-S-{student_model_name.replace('/','-').replace('simcse_original','').replace('output','').replace('--','-')}"

print(f'Save path: {model_save_path}')
teacher_emb_model = models.Transformer(teacher_model_name, max_seq_length=32)
teacher_dimension = teacher_emb_model.get_word_embedding_dimension()
del teacher_emb_model
teacher_model = SentenceTransformer(teacher_model_name)

logging.info(f"Create student model from scratch:{student_model_name}")
student_word_embedding_model = models.Transformer(student_model_name, max_seq_length=max_seq_length)
student_dimension = student_word_embedding_model.get_word_embedding_dimension()
student_pooling_model = models.Pooling(student_dimension)

if teacher_dimension != student_dimension:
    dense_model = models.Dense(in_features=student_dimension, out_features=teacher_dimension, activation_function=nn.Tanh())
    student_model = SentenceTransformer(modules=[student_word_embedding_model, student_pooling_model,dense_model])
else:
    student_model = SentenceTransformer(modules=[student_word_embedding_model, student_pooling_model])


train_samples = []
train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=True)
train_data.load_data(nli_dataset_path)
for index,sent_para in enumerate(train_data.datasets[-1]):
	en = sent_para[0]
	non_en = ''
	for item in sent_para[1]:
		if item != en:
			non_en = item
			break
	if non_en != '':
		train_samples.append(InputExample(texts=[en, non_en]))
	else:
		None
		# print(f"Cant find pair of sent:{index}\n Example:{sent_para}")
	

# logging.info("Read Multilingual NLI train dataset")

train_dataloader_distill = DataLoader(train_data, shuffle=False, batch_size=train_batch_size)
train_loss_distill = losses.MSELoss(model=student_model)

train_dataset = SentencesDataset(train_samples, model=student_model)
train_dataloader_bsl = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size)
train_loss_bsl = losses.BYOLoss(model=student_model, sentence_embedding_dimension=student_model.get_sentence_embedding_dimension(), moving_average_decay=moving_average_decay)

# if len(train_samples)*2 != train_data.num_sentences:
# 	raise Exception(f"Data is not equal please check para:{train_data.num_sentences} bsl:{len(train_samples)*2}")


#Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
	reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
	for row in reader:
		if row['split'] == 'dev':
			score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
			dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev',main_similarity=SimilarityFunction.COSINE)


# Configure the training
warmup_steps = math.ceil(len(train_dataloader_bsl) * num_epochs * 0.1)  # 10% of train data for warm-up
evaluation_steps = int(len(train_dataloader_bsl) * 0.1) #Evaluate every 10% of the data
# evaluation_steps = 5
logging.info("Training sentences: {}".format(len(train_samples)))
logging.info("Warmup-steps: {}".format(warmup_steps))
logging.info("Performance before training")
dev_evaluator(student_model)


# Train the model
student_model.fit(train_objectives=[(train_dataloader_bsl, train_loss_bsl), (train_dataloader_distill, train_loss_distill)],
		  evaluator=dev_evaluator,
		  epochs=num_epochs,
		  evaluation_steps=evaluation_steps,
		  warmup_steps=warmup_steps,
		  output_path=model_save_path,
		  optimizer_params={'lr': 5e-5},
		  use_amp=True        #Set to True, if your GPU supports FP16 cores
		  )
