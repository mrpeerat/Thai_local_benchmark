"""
This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

SimCSE will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

Usage:
python train_simcse_from_file.py path/to/sentences.txt

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Training parameters
model_name = 'distilbert-base-multilingual-cased' # bert-base-multilingual-cased, xlm-roberta-base, microsoft/Multilingual-MiniLM-L12-H384, distilbert-base-multilingual-cased
train_batch_size = 128
max_seq_length = 32
num_epochs = 1
nli_dataset_path = '../BSL/training/data/multilingual_NLI'
sts_dataset_path = '../BSL/training/data/stsbenchmark.tsv.gz'
sts_corpus = "../BSL/training/data/STS2017-extended.zip"
source_languages = ['en']                     
target_languages = ['en', 'de', 'es', 'fr', 'ar', 'tr']
pooling_mode = 'mean' # cls,max,mean

# Save path to store our model
output_name = ''
if len(sys.argv) >= 3:
    output_name = "-"+sys.argv[2].replace(" ", "_").replace("/", "_").replace("\\", "_")


model_save_path = f"output/{nli_dataset_path.split('/')[-1]}_simcse_{model_name.replace('-','_')}_{pooling_mode}"
print(f"Path:{model_save_path}")

#


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_mode=pooling_mode)
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



################# Read the train corpus  #################
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

   

logging.info("Train sentences: {}".format(len(train_samples)))

# We train our model using the MultipleNegativesRankingLoss
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MultipleNegativesRankingLoss(model)
evaluation_steps = int(len(train_dataloader) * 0.1) #Evaluate every 10% of the data
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          evaluator=dev_evaluator,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          optimizer_params={'lr': 5e-5},
          show_progress_bar=True,
          output_path=model_save_path,
          use_amp=True  # Set to True, if your GPU supports FP16 cores
          )


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
		test_evaluator = EmbeddingSimilarityEvaluator(data['sentences1'], data['sentences2'], data['scores'], batch_size=16, name=filename, main_similarity=SimilarityFunction.COSINE, show_progress_bar=False)
		test_evaluator(model, output_path=model_save_path)




