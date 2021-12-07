import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from datetime import datetime

import os
import logging
import sentence_transformers.util
import csv
import gzip
from tqdm.autonotebook import tqdm
import numpy as np
import zipfile
import io
from glob import glob 
from torch import nn

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


teacher_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'   #output/BSL_tuning_output-unsup-simcse-mBert-2021-11-12_04-49-02
student_model_name = 'microsoft/Multilingual-MiniLM-L12-H384'       # distilbert-base-multilingual-cased, microsoft/Multilingual-MiniLM-L12-H384

max_seq_length = 128                #Student model max. lengths for inputs (number of word pieces)
train_batch_size = 64               #Batch size for training
inference_batch_size = 64           #Batch size at inference

num_epochs = 20                       #Train for x epochs


# Define the language codes you would like to extend the model to
source_languages = set(['en'])                      # Our teacher model accepts English (en) sentences
target_languages = set(['de', 'es', 'fr', 'ar', 'tr'])    # We want to extend the model to these new languages. For language codes, see the header of the train file


output_path = f"output/skd/skd-T-{teacher_model_name.replace('/','-')}-S-{student_model_name.replace('/','-')}"


# Here we define train train and dev corpora
sts_corpus = "datasets/STS2017-extended.zip"     # Extended STS2017 dataset for more languages
sts_dataset_path = '../BSL/training/data/stsbenchmark.tsv.gz'
train_folder = 'multilingual_nli'
nli_dataset_path = '../BSL/training/data/multilingual_NLI'

######## Start the extension of the teacher model to multiple languages ########
logger.info("Load teacher model")
teacher_emb_model = models.Transformer(teacher_model_name, max_seq_length=32)
teacher_dimension = teacher_emb_model.get_word_embedding_dimension()
del teacher_emb_model
teacher_model = SentenceTransformer(teacher_model_name)

logger.info("Create student model from scratch")
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


###### Read Parallel Sentences Dataset ######
train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=True)
for train_file in train_files:
    train_data.load_data(train_file)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=student_model)
 
###### Read Parallel Sentences Dataset ######
train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=student_model, batch_size=inference_batch_size, use_embedding_cache=True)
for train_file in train_files:
    train_data.load_data(train_file)

train_dataloader_2 = DataLoader(train_data, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss_self = losses.MSELoss(model=student_model)



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

warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
evaluation_steps = int(len(train_dataloader) * 0.1) #Evaluate every 10% of the data

# Train the model
student_model.fit(train_objectives=[(train_dataloader, train_loss),(train_dataloader_2, train_loss_self)], # train_dataloader_2 , L2_loss
# student_model.fit(train_objectives=[(train_dataloader, train_loss)], # train_dataloader_2 , L2_loss
          evaluator=dev_evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          evaluation_steps=evaluation_steps,
          output_path=output_path,
          save_best_model=True,
          optimizer_params= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
          )


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

model = SentenceTransformer(output_path)
for filename, data in sts_data.items():
		test_evaluator = EmbeddingSimilarityEvaluator(data['sentences1'], data['sentences2'], data['scores'], batch_size=16, name=filename, main_similarity=SimilarityFunction.COSINE, show_progress_bar=False)
		test_evaluator(model, output_path=output_path)




