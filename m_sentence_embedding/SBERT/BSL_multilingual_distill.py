import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator
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


sts_corpus = "datasets/STS2017-extended.zip"     # Extended STS2017 dataset for more languages
sts_dataset_path = '../BSL/training/data/stsbenchmark.tsv.gz'
train_folder = 'multilingual_nli'
nli_dataset_path = '../BSL/training/data/multilingual_NLI'

source_languages = ['en']                     
target_languages = ['en', 'de', 'es', 'fr', 'ar', 'tr']    

# ../../SBERT/output/multilingual_NLI_simcse_bert_base_multilingual_cased
teacher_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'   #output/BSL_tuning_output-unsup-simcse-mBert-2021-11-12_04-49-02
student_model_name = 'microsoft/Multilingual-MiniLM-L12-H384'       # distilbert-base-multilingual-cased, microsoft/Multilingual-MiniLM-L12-H384
train_batch_size = 64
num_epochs = 20
max_seq_length = 128
moving_average_decay = 0.999
inference_batch_size = 64
unsupervised_check = True

model_save_path = f"output/bsl_multilingual_distill/BSL_skd-shuffle-T-{teacher_model_name.replace('/','-')}-S-{student_model_name.replace('/','-')}"

teacher_emb_model = models.Transformer(teacher_model_name, max_seq_length=32)
teacher_dimension = teacher_emb_model.get_word_embedding_dimension()
del teacher_emb_model
teacher_model = SentenceTransformer(teacher_model_name)

logging.info("Create student model from scratch")
student_word_embedding_model = models.Transformer(student_model_name, max_seq_length=max_seq_length)
student_dimension = student_word_embedding_model.get_word_embedding_dimension()
student_pooling_model = models.Pooling(student_dimension)

if teacher_dimension != student_dimension:
    dense_model = models.Dense(in_features=student_dimension, out_features=teacher_dimension, activation_function=nn.Tanh())
    student_model = SentenceTransformer(modules=[student_word_embedding_model, student_pooling_model,dense_model])
else:
    student_model = SentenceTransformer(modules=[student_word_embedding_model, student_pooling_model])


train_files = []
for f in glob(f"{train_folder}/en-*.train"):
    train_files.append(f)

train_samples = []
train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=True)
for train_file in train_files:
	train_data.load_data(train_file)
	for index,sent_para in enumerate(train_data.datasets[-1]):
		en = sent_para[0]
		non_en = ''
		for item in sent_para[1]:
			if item != en:
				non_en = item
				break
		if non_en == '':
			print(f"Cant find pair of sent:{index}\n Example:{sent_para}")
		train_samples.append(InputExample(texts=[en, non_en]))

# logging.info("Read Multilingual NLI train dataset")

train_dataloader_distill = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss_distill = losses.MSELoss(model=student_model)

train_dataset = SentencesDataset(train_samples, model=student_model)
train_dataloader_bsl = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss_bsl = losses.BYOLoss(model=student_model, sentence_embedding_dimension=student_model.get_sentence_embedding_dimension(), moving_average_decay=moving_average_decay)

if len(train_samples)*2 != train_data.num_sentences:
	raise Exception(f"Data is not equal please check para:{train_data.num_sentences} bsl:{len(train_samples)*2}")


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


# Configure the training
warmup_steps = math.ceil(len(train_dataloader_bsl) * num_epochs * 0.1)  # 10% of train data for warm-up
evaluation_steps = int(len(train_dataloader_bsl) * 0.1) #Evaluate every 10% of the data
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

##############################################################################
#
# Load the stored model and evaluate its performance on STS  dataset
#
##############################################################################


logging.info("Read STS test dataset")
if not os.path.exists(sts_corpus):
    util.http_get('https://sbert.net/datasets/STS2017-extended.zip', sts_corpus)
    

##### Read cross-lingual Semantic Textual Similarity (STS) data ####
all_languages = list(set(list(source_languages)+list(target_languages)))
sts_data = {}
evaluators = [] 
#Open the ZIP File of STS2017-extended.zip and check for which language combinations we have STS data
with zipfile.ZipFile(sts_corpus) as zip:
		filelist = zip.namelist()
		for i in range(len(all_languages)):
				for j in range(i, len(all_languages)):
						lang1 = all_languages[i]
						lang2 = all_languages[j]
						filepath = 'STS2017-extended/STS.{}-{}.txt'.format(lang1, lang2)
						if filepath not in filelist:
								lang1, lang2 = lang2, lang1
								filepath = 'STS2017-extended/STS.{}-{}.txt'.format(lang1, lang2)

						if filepath in filelist:
								filename = os.path.basename(filepath)
								sts_data[filename] = {'sentences1': [], 'sentences2': [], 'scores': []}

								fIn = zip.open(filepath)
								for line in io.TextIOWrapper(fIn, 'utf8'):
										sent1, sent2, score = line.strip().split("\t")
										score = float(score)
										sts_data[filename]['sentences1'].append(sent1)
										sts_data[filename]['sentences2'].append(sent2)
										sts_data[filename]['scores'].append(score)

model = SentenceTransformer(model_save_path)
for filename, data in sts_data.items():
		test_evaluator = EmbeddingSimilarityEvaluator(data['sentences1'], data['sentences2'], data['scores'], batch_size=16, name=filename, show_progress_bar=False)
		test_evaluator(model, output_path=model_save_path)




