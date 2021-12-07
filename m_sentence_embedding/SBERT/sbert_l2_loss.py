import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
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

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


teacher_model_name = 'output/BSL_tuning_output-unsup-simcse-mBert-2021-11-12_04-49-02'   #Our monolingual teacher model, we want to convert to multiple languages
student_model_name = 'distilbert-base-multilingual-cased'       # distilbert-base-multilingual-cased, microsoft/Multilingual-MiniLM-L12-H384

max_seq_length = 128                #Student model max. lengths for inputs (number of word pieces)
train_batch_size = 64               #Batch size for training
inference_batch_size = 64           #Batch size at inference

num_epochs = 20                       #Train for x epochs


# Define the language codes you would like to extend the model to
source_languages = set(['en'])                      # Our teacher model accepts English (en) sentences
target_languages = set(['de', 'es', 'fr', 'ar', 'tr'])    # We want to extend the model to these new languages. For language codes, see the header of the train file


output_path = f"output/make-multilingual-original-T-{teacher_model_name.replace('/','-')}-S-{student_model_name.replace('/','-')}"


# Here we define train train and dev corpora
sts_corpus = "datasets/STS2017-extended.zip"     # Extended STS2017 dataset for more languages
sts_dataset_path = '../BSL/training/data/stsbenchmark.tsv.gz'
train_folder = 'multilingual_nli'
nli_dataset_path = '../BSL/training/data/multilingual_NLI'


logger.info("Create student model from scratch")
word_embedding_model = models.Transformer(student_model_name, max_seq_length=max_seq_length)
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])



# ################# Read the train corpus  #################
label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = []

s_e1 = gzip.open(os.path.join(nli_dataset_path, 's_e1.train.gz'),
			   mode="rt", encoding="utf-8").readlines()
s_e2 = gzip.open(os.path.join(nli_dataset_path, 's_e2.train.gz'),
			   mode="rt", encoding="utf-8").readlines()
s_en = gzip.open(os.path.join(nli_dataset_path, 's_en.train.gz'),
			   mode="rt", encoding="utf-8").readlines()
s_de = gzip.open(os.path.join(nli_dataset_path, 's_de.train.gz'),
			   mode="rt", encoding="utf-8").readlines()
s_es = gzip.open(os.path.join(nli_dataset_path, 's_es.train.gz'),
			   mode="rt", encoding="utf-8").readlines()
s_fr = gzip.open(os.path.join(nli_dataset_path, 's_fr.train.gz'),
			   mode="rt", encoding="utf-8").readlines()
s_ar = gzip.open(os.path.join(nli_dataset_path, 's_ar.train.gz'),
			   mode="rt", encoding="utf-8").readlines()
s_tr = gzip.open(os.path.join(nli_dataset_path, 's_tr.train.gz'),
			   mode="rt", encoding="utf-8").readlines()
labels = gzip.open(os.path.join(nli_dataset_path, 'labels.train.gz'),
				   mode="rt", encoding="utf-8").readlines()

for e1, e2, en, de, es, fr, ar, tr, label in zip(s_e1, s_e2, s_en, s_de, s_es, s_fr, s_ar, s_tr, labels):
	e1 = e1.strip().split('\t')[0]
	e2 = e2.strip().split('\t')[0]
	en = en.strip().split('\t')[0]
	de = de.strip().split('\t')[0]
	es = es.strip().split('\t')[0]
	fr = fr.strip().split('\t')[0]
	ar = ar.strip().split('\t')[0]
	tr = tr.strip().split('\t')[0]
	label = label.strip().split('\t')[0]
	label_id = label2int[label]
	if e1 != e2:
	  train_samples.append(InputExample(texts=[e1, e2]))
	train_samples.append(InputExample(texts=[en, de]))
	train_samples.append(InputExample(texts=[en, es]))
	train_samples.append(InputExample(texts=[en, fr]))
	train_samples.append(InputExample(texts=[en, ar]))
	train_samples.append(InputExample(texts=[en, tr]))

# We train our model using the MultipleNegativesRankingLoss
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MSELoss(model=student_model)


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
student_model.fit(train_objectives=[(train_dataloader, train_loss)], # train_dataloader_2 , L2_loss
          evaluator=dev_evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          evaluation_steps=evaluation_steps,
        #   output_path=output_path,
        #   save_best_model=True,
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




