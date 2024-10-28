from seacrowd import SEACrowdConfigHelper
from seacrowd.utils.constants import Tasks
import pandas as pd
import datasets
from enum import Enum
import datasets
import nltk
nltk.download('punkt_tab')

def patch_resolve_trust_remote_code():
    def resolve_trust_remote_code(trust_remote_code: bool | None, repo_id: str):
        print('Patch `trust_remote_code` to enable fully auto-run. Beware of the risk of code injection in the dataset.')
        return True
    datasets.load.resolve_trust_remote_code = resolve_trust_remote_code

patch_resolve_trust_remote_code()


NLU_TASK_LIST = {
    "wisesight_thai_sentiment_seacrowd_text",
    "m3exam_tha_seacrowd_qa",
    "xcopa_tha_seacrowd_qa",
    "belebele_tha_thai_seacrowd_qa",
    "xnli.tha_seacrowd_pairs",
    'thaiexam_qa'
}


NLG_TASK_LIST = [
    "xl_sum_tha_seacrowd_t2t",
    "flores200_eng_Latn_tha_Thai_seacrowd_t2t",
    "flores200_tha_Thai_eng_Latn_seacrowd_t2t",
    "iapp_squad_seacrowd_qa",
]

LOCAL_TASK_LIST = [
    "mrp/sum_esan",
    "mrp/sum_north",
    "mrp/sum_south",
    "mrp/sum_center",
    "mrp/translation_esan_eng",
    "mrp/translation_north_eng",
    "mrp/translation_south_eng",
    "mrp/translation_center_eng",
    "mrp/translation_eng_esan",
    "mrp/translation_eng_north",
    "mrp/translation_eng_south",
    "mrp/translation_eng_center",
    "mrp/qa_esan",
    "mrp/qa_north",
    "mrp/qa_south",
    "mrp/qa_center",
    "mrp/food_esan",
    "mrp/food_north",
    "mrp/food_south",
    "mrp/food_center",
    "mrp/converation_esan",
    "mrp/converation_north",
    "mrp/converation_south",
    "mrp/converation_center",
]



def load_nlu_datasets():
    nc_conhelp = SEACrowdConfigHelper()
    cfg_name_to_dset_map = {}

    for config_name in NLU_TASK_LIST:
        if config_name == 'thaiexam_qa':
            ds = datasets.load_dataset('kunato/thai-exam-seacrowd', revision='59198720623a81239dbbde1e77a98a183f002c41')
            cfg_name_to_dset_map[config_name] = (ds, Tasks.QUESTION_ANSWERING)
        else:
            schema = config_name.split('_')[-1]
            con = nc_conhelp.for_config_name(config_name)
            cfg_name_to_dset_map[config_name] = (con.load_dataset(), list(con.tasks)[0])
    return cfg_name_to_dset_map

def load_local_datasets():
    data = {}
    for config_name in LOCAL_TASK_LIST:
        data[config_name.split('/')[-1]] = datasets.load_dataset(config_name)
    return data

def load_nlg_datasets():
    nc_conhelp = SEACrowdConfigHelper()
    cfg_name_to_dset_map = {}

    for config_name in NLG_TASK_LIST:
        schema = config_name.split('_')[-1]
        con = nc_conhelp.for_config_name(config_name)
        cfg_name_to_dset_map[config_name] = (con.load_dataset(), list(con.tasks)[0])
    return cfg_name_to_dset_map