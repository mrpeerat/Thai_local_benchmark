#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
import gzip
import zipfile
import io
import os
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, 
        help="Transformers' model name or path")
print()
args = parser.parse_args()
print(f"Model:{args.model_name_or_path}")
sts_corpus = "../BSL/training/data/STS2017-extended.zip"
source_languages = ['en']                     
target_languages = ['en', 'de', 'es', 'fr', 'ar', 'tr']

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

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

model = SentenceTransformer(args.model_name_or_path)
for filename, data in sts_data.items():
    test_evaluator = EmbeddingSimilarityEvaluator(data['sentences1'], data['sentences2'], data['scores'], main_similarity=SimilarityFunction.COSINE, batch_size=16, name=filename, show_progress_bar=False)
    test_evaluator(model)

