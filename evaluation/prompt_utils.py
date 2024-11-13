TASK_TO_PROMPT = {
    'center': {
        'conversation': [
            'จงสร้างบทสนทนาของคนภาคกลาง โดยพูดคุยเกี่ยวกับ [INPUT] บทสนทนาไม่ควรเกิน 5 turns',
        ],
        'food': [
            'จงเขียนวิธีทำ [INPUT]',
        ],

        'translation': [
            'แปลข้อความต่อไปนี้จาก [SOURCE] เป็น [TARGET] ให้การแปลของคุณโดยตรงโดยไม่ต้องมีข้อมูลเพิ่มเติมใดๆ\nข้อความ: [INPUT]\nคำแปล:',
        ],
        
        # Tasks.SUMMARIZATION
        'summarization': [
            'จงสรุปข้อความด้านล่าง\nข้อความ: [INPUT]\nสรุป:',
        ],

        # Task.QA_EXTRACTIVE_ABSTRACTIVE
        'qa':[
            'โปรดอ้างอิงถึงข้อความด้านล่างนี้และตอบคำถามต่อไปนี้ โดยตอบโดยใช้แค่ข้อความที่อยู่ในบทความ:\nข้อความ: [CONTEXT]\nคำถาม: [QUESTION]\nคำตอบ:',
        ],
    },
    'north': {
        'conversation': [
            'จงสร้างบทสนหื้อเขียนกำอู้ของคนเหนือ โดยอู้เกี่ยวกับ [INPUT] บทสนทนาบะควรเกิน 5 turns',
        ],
        'food': [
            'หื้อเขียนวิธียะ [INPUT] หื้อเป๋นภาษาเหนือ',
        ],

        'translation': [
            'แปลข้อความต่อไปนี้จาก [SOURCE] เป๋น [TARGET] หื้อตั๋วแปลโดยตรงโดยตี้บะต้องมีข้อมูลเพิ่มเติมใดๆ\nข้อความ: [INPUT]\nคำแปล:',
        ],
        
        # Tasks.SUMMARIZATION
        'summarization': [
            'หื้อสรุปข้อความตางล่างหือเป็นภาษาเหนือ\nข้อความ: [INPUT]\nสรุป:',
        ],

        # Task.QA_EXTRACTIVE_ABSTRACTIVE
        'qa':[
            'โปรดอ้างอิงถึงข้อความตางล่างนี้และตอบคำถามต่อไปนี้หื้อเป็นภาษาเหนือ หื้อตอบโดยใช้ก่าข้อความตี้อยู่ในบทความ:\nข้อความ: [CONTEXT]\nคำถาม: [QUESTION]\nคำตอบ:',
        ],
    },
    'east': {
        'conversation': [
            'จงเขียนบทเว่าของคนภาคอีสาน โดยเว่าเกี่ยวกับ [INPUT] บทสนทนาบ่ควรเกิน 5 turns',
        ],
        'food': [
            'จงเขียนวิธีเฮ็ด [INPUT] เป็นภาษาอีสาน',
        ],

        'translation': [
            'แปลข้อความต่อไปนี้จาก [SOURCE] เป็น [TARGET] ให้เจ้าแปลโดยตรงโดยบ่ต้องมีข้อมูลเพิ่มเติมใดๆ\nข้อความ: [INPUT]\nคำแปล:',
        ],
        
        # Tasks.SUMMARIZATION
        'summarization': [
            'จงสรุปข้อความทางลุ่มเป็นภาษาอีสาน\nข้อความ: [INPUT]\nสรุป:',
        ],

        # Task.QA_EXTRACTIVE_ABSTRACTIVE
        'qa':[
            'โปรดอ้างอิงถึงข้อความทางลุมพี้และตอบคำถามต่อไปนี้เป็นภาษาอีสาน ให้ตอบโดยใช้แค่ข้อความที่อยู่ในบทความ:\nข้อความ:[CONTEXT]:\nคำถาม: [QUESTION]\nคำตอบ:',
        ],
    },
    'south': {
        'conversation': [
            'ให้เขียนบทพูดของคนใต้ โดยแหลงเกี่ยวกับ [INPUT] บทพูดไม่ควรเกิน 5 ตา',
        ],
        'food': [
            'ให้เขียนวิธีทำ [INPUT] เป็นภาษาใต้',
        ],

        'translation': [
            'แปลข้อความต่อจากนี้ [SOURCE] เป็น [TARGET] ให้การแปลของเติ้ลแปลตรงตัวเลย ไม่ต้องไสข้อมูลเพิ่มเติมไหร\nข้อความ: [INPUT]\nคำแปล:',
        ],
        
        # Tasks.SUMMARIZATION
        'summarization': [
            'ให้สรุปข้อความข้างล่างเป็นภาษาใต้\nข้อความ:[INPUT]\nสรุปว่า:',
        ],

        # Task.QA_EXTRACTIVE_ABSTRACTIVE
        'qa':[
            'ช่วยใช้ข้อความข้างล่างนี้ตอบคำถามเป็นภาษาใต้โดยตอบแค่คำตอบที่มีอยู๋ในบทความ:\nข้อความ: [CONTEXT]\nคำถาม: [QUESTION]\nคำตอบ:',
        ],
    },
}

LABEL_LANG_MAP ={
    # Tasks.SENTIMENT_ANALYSIS
    'lazada_review_filipino_seacrowd_text': {
        'eng': {'1': 'very negative', '2': 'negative', '3': 'neutral', '4': 'positive', '5': 'very positive'},
        'ind': {'1': 'sangat negatif', '2': 'negatif', '3': 'netral', '4': 'positif', '5': 'sangat positif'},
    },
    'gklmip_sentiment_seacrowd_text': {
        'eng': {'1': 'very negative', '2': 'negative', '3': 'neutral', '4': 'positive', '5': 'very positive'},
        'ind': {'1': 'sangat negatif', '2': 'negatif', '3': 'netral', '4': 'positif', '5': 'sangat positif'},
    },
    'indolem_sentiment_seacrowd_text': {
        'eng': {'negative': 'negative', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'positive': 'positif'},
    },
    'id_sentiment_analysis_seacrowd_text': {
        'eng': {'-1': 'negative', '0': 'neutral', '1': 'positive'},
        'ind': {'-1': 'negatif', '0': 'netral', '1': 'positif'},
    },
    'karonese_sentiment_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'wisesight_thai_sentiment_seacrowd_text': {
        'eng': {'pos': 'positive', 'neu': 'neutral', 'neg': 'negative', 'q': 'not applicable (question)'},
        'ind': {'pos': 'positif', 'neu': 'netral', 'neg': 'negatif', 'q': 'tidak berlaku (pertanyaan)'},
    },
    'wongnai_reviews_seacrowd_text': {
        'eng': {'1': 'very negative', '2': 'negative', '3': 'neutral', '4': 'positive', '5': 'very positive'},
        'ind': {'1': 'sangat negatif', '2': 'negatif', '3': 'netral', '4': 'positif', '5': 'sangat positif'},
    },
    'vlsp2016_sa_seacrowd_text': {
        'eng': {'POS': 'positive', 'NEU': 'neutral', 'NEG': 'negative'},
        'ind': {'POS': 'positif', 'NEU': 'netral', 'NEG': 'negatif'},
    },
    'typhoon_yolanda_tweets_seacrowd_text': {
        'eng': {'-1': 'negative', '0': 'neutral', '1': 'positive'},
        'ind': {'-1': 'negatif', '0': 'netral', '1': 'positif'},
    },
    # 'total_defense_meme_sentiment_seacrowd_text' - will be added later
    'smsa_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'prdect_id_sentiment_seacrowd_text': {
        'eng': {'Negative': 'negative', 'Positive': 'positive'},
        'ind': {'Negative': 'negatif', 'Positive': 'positif'},
    },
    'id_sent_emo_mobile_apps_sentiment_seacrowd_text': {
        'eng': {'Negative': 'negative', 'Neutral': 'neutral', 'Positive': 'positive'},
        'ind': {'Negative': 'negatif', 'Neutral': 'netral', 'Positive': 'positif'},
    },
    'shopee_reviews_tagalog_seacrowd_text': {
        'eng': {'0': 'very negative', '1': 'negative', '2': 'neutral', '3': 'positive', '4': 'very positive'},
        'ind': {'0': 'sangat negatif', '1': 'negatif', '2': 'netral', '3': 'positif', '4': 'sangat positif'},
    },
    'nusatranslation_senti_ind_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_abs_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_btk_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_bew_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_bhp_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_jav_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_mad_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_mak_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_min_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_mui_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_rej_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_sun_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_ind_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_ace_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_jav_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_sun_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_min_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_bug_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_bbc_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_ban_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_nij_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_mad_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_bjn_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_eng_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'indonglish_seacrowd_text': {
        'eng': {'Negatif': 'negative', 'Netral': 'neutral', 'Positif': 'positive'},
        'ind': {'Negatif': 'negatif', 'Netral': 'netral', 'Positif': 'positif'},
    },
    # Tasks.TOPIC_MODELING
    'gklmip_newsclass_seacrowd_text': {
        'eng': {"culture": "culture", "economic": "economic", "education": "education", "environment": "environment", "health": "health", "politics": "politics", "right": "right", "science": "science"},
        'ind': {"culture": "kebudayaan", "economic": "ekonomi", "education": "edukasi", "environment": "lingkungan", "health": "kesehatan", "politics": "politik", "right": "hak", "science": "sains"},
    },
    'indonesian_news_dataset_seacrowd_text': {
        'eng': {"bola": "soccer", "news": "news", "bisnis": "business", "tekno": "technology", "otomotif": "vehicle"},
        'ind': {"bola": "sepak bola", "news": "berita", "bisnis": "bisnis", "tekno": "teknologi", "otomotif": "otomotif"},
    },
    'uit_vion_seacrowd_text': {
        'eng': {0: 'technology', 1: 'travel', 2: 'education', 3: 'entertainment', 4: 'science', 5: 'business', 6: 'law', 7: 'health', 8: 'world', 9: 'sports', 10: 'news', 11: 'vehicle', 12: 'life'},
        'ind': {0: 'teknologi', 1: 'jalan-jalan', 2: 'edukasi', 3: 'hiburan', 4: 'sains', 5: 'bisnis', 6: 'hukum', 7: 'kesehatan', 8: 'dunia', 9: 'olahraga', 10: 'berita', 11: 'otomotif', 12: 'kehidupan sehari-hari'},
    },
    # total_defense_meme_topic_seacrowd_text - will be added later
    'sib_200_ace_Arab_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_ace_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_ban_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_bjn_Arab_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_bjn_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_bug_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_ceb_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_ilo_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_ind_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_jav_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_kac_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_khm_Khmr_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_lao_Laoo_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_lus_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_min_Arab_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_min_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_mya_Mymr_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_pag_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_shn_Mymr_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_sun_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_tgl_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_tha_Thai_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_vie_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_war_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_zsm_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'nusaparagraph_topic_btk_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_bew_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_bug_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_jav_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_mad_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_mak_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_min_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_mui_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_rej_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_sun_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    # Tasks.MORALITY_CLASSIFICATION
    'emotes_3k_tgl_seacrowd_text': {
        'eng': {'Moral': 'moral', 'Immoral': 'immoral'},
        'ind': {'Moral': 'bermoral', 'Immoral': 'tidak bermoral'},
    },
    'emotes_3k_eng_seacrowd_text': {
        'eng': {'Moral': 'moral', 'Immoral': 'immoral'},
        'ind': {'Moral': 'bermoral', 'Immoral': 'tidak bermoral'},
    },
    # # Tasks.QUESTION_ANSWERING & Tasks.COMMONSENSE_REASONING - no need
    "indo_story_cloze_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "xstorycloze_id_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "xstorycloze_my_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "indommlu_ind_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_ban_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_mad_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_mak_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_sun_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_jav_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_bjn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_abl_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_nij_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "seaeval_cross_mmlu_ind_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_mmlu_vie_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_mmlu_zlm_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_mmlu_fil_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_logiqa_ind_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_logiqa_vie_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_logiqa_zlm_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_logiqa_fil_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "m3exam_jav_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "m3exam_tha_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "thaiexam_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "m3exam_vie_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "copal_colloquial_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "xcopa_tha_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "xcopa_vie_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "xcopa_ind_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "seaeval_sg_eval_eng_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_ph_eval_eng_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "mabl_ind_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "mabl_jav_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "mabl_sun_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    # "facqa_seacrowd_qa",
    # "iapp_squad_seacrowd_qa",
    # "idk_mrc_seacrowd_qa",
    # "vihealthqa_seacrowd_qa",
    # "uit_vicov19qa_seacrowd_qa",
    # "qasina_seacrowd_qa",
    "belebele_ceb_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_ilo_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_ind_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_jav_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_kac_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_khm_khmr_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_lao_laoo_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_mya_mymr_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_shn_mymr_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_sun_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_tgl_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_tha_thai_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_vie_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_war_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_zsm_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    # "mkqa_khm_seacrowd_qa",
    # "mkqa_zsm_seacrowd_qa",
    # "mkqa_tha_seacrowd_qa",
    # "mkqa_vie_seacrowd_qa",
    # "xquad.th_seacrowd_qa",
    # "xquad.vi_seacrowd_qa",
    "okapi_m_arc_ind_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    # "okapi_m_mmlu_ind_seacrowd_qa",
    "okapi_m_arc_vie_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    # "okapi_m_mmlu_vie_seacrowd_qa",
    # # Tasks.TEXTUAL_ENTAILMENT
    'indonli_seacrowd_pairs': {
        'eng': {'c': 'contradiction', 'e': 'entailment', 'n': 'irrelevant'},
        'ind': {'c': 'saling berlawanan', 'e': 'saling mendukung', 'n': 'tidak berhubungan'},
    },
    'wrete_seacrowd_pairs': {
        'eng': {'NotEntail': 'contradiction', 'Entail_or_Paraphrase': 'entailment'},
        'ind': {'NotEntail': 'saling berlawanan', 'Entail_or_Paraphrase': 'saling mendukung'},
    },
    'snli_indo_seacrowd_pairs': {
        'eng': {"kontradiksi": "contradiction", "keterlibatan": "entailment", "netral": "irrelevant"},
        'ind': {"kontradiksi": "saling berlawanan", "keterlibatan": "saling mendukung", "netral": "tidak berhubungan"},
    },
    'myxnli_seacrowd_pairs': {
        'eng': {"contradiction": "contradiction", "entailment": "neutral", "neutral": "irrelevant"},
        'ind': {"contradiction": "saling berlawanan", "entailment": "saling mendukung", "neutral": "tidak berhubungan"},
    },
    'xnli.tha_seacrowd_pairs': {
        'eng': {"contradiction": "contradiction", "entailment": "neutral", "neutral": "irrelevant"},
        'ind': {"contradiction": "saling berlawanan", "entailment": "saling mendukung", "neutral": "tidak berhubungan"},
    },
    'xnli.vie_seacrowd_pairs': {
        'eng': {"contradiction": "contradiction", "entailment": "neutral", "neutral": "irrelevant"},
        'ind': {"contradiction": "saling berlawanan", "entailment": "saling mendukung", "neutral": "tidak berhubungan"},
    },
}

LANG_MAP = {
    'eng': {
        'ind': 'Indonesian',
        'xdy': 'Dayak',
        'bug': 'Buginese',
        'mad': 'Madurese',
        'bjn': 'Banjarese',
        'tiociu': 'Tiociu',
        'jav': 'Javanese',
        'sun': 'Sundanese',
        'ace': 'Acehnese',
        'ban': 'Balinese',
        'min': 'Minangkabau',
        'eng': 'English',
        'vie': 'Vietnamese',
        'mya': 'Burmese',
        'gor': 'Gorontalo',
        'ceb': 'Cebuano',
        'ilo': 'Iloko',
        'hil': 'Hiligaynon',
        'khm': 'Khmer',
        'lao': 'Lao',
        'zlm': 'Malay',
        'nia': 'Nias',
        'tgl': 'Tagalog',
        'tha': 'Thai',
        'end': 'Ende',
        'nxe': 'Nage',
        'ssq': 'ssq',
        'ljl': 'Lio',
        'kac': 'Kachin',
        'eng-US': 'English',
        'lus': 'Lushai',
        'min': 'Minangkabau',
        'pag': 'Pangasinan',
        'shn': 'Shan',
        'war': 'Waray',
        'zsm': 'Standard Malay',
        'hmv': 'Hmong Do',
        'bbc': 'Batak Toba',
        'nij': 'Ngaju',
        'fil': 'Filipino',
        'tl': 'Tagalog',
        'km': 'Khmer',
        'vi': 'Vietnamese',
        'th': 'Thai',
        'ms': 'Malay',
        'id': 'Indonesian',
        'lo': 'Lao',
        'my': 'Mayan',
        'en': 'English',
        'tam': 'Tamil'
    },
    'tha': {
        'tha': 'Thai',
        'eng': 'English',
        'eng-US': 'English',
    },
    'ind': {
        'ind': 'Indonesia',
        'xdy': 'Dayak',
        'bug': 'Bugis',
        'mad': 'Madura',
        'bjn': 'Banjar',
        'tiociu': 'Tiociu',
        'jav': 'Jawa',
        'sun': 'Sunda',
        'ace': 'Aceh',
        'ban': 'Bali',
        'min': 'Minangkabau'
    }
}

def get_label_mapping(dset_subset, prompt_lang):
    try:
        return LABEL_LANG_MAP[dset_subset][prompt_lang]
    except KeyError:
        return LABEL_LANG_MAP[dset_subset]['eng']

def get_lang_name(prompt_lang, lang_code):
    return LANG_MAP[prompt_lang][lang_code]

def get_prompt(prompt_lang, return_only_one=False):
    prompt_templates = {}
    for config, prompts in TASK_TO_PROMPT[prompt_lang].items():
        if return_only_one:
            prompt_templates[config] = [prompts[0]]
        else:
            prompt_templates[config] = prompts
    return prompt_templates