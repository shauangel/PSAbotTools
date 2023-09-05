#!/usr/bin/env python3
import numpy as np
import config
import logging
# nltk
import nltk
# spacy
import spacy
from collections import Counter
from heapq import nlargest
# LDA model
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel, EnsembleLda, CoherenceModel
from gensim.models.callbacks import PerplexityMetric, CoherenceMetric
from itertools import chain
import json
import os


# TextAnalyze Module: pre-processing word content, ex
class TextAnalyze:
    STOPWORDS = []  # 停用詞: 可忽略的詞，沒有賦予上下文句意義的詞
    POS_TAG = ['NOUN', 'PROPN']  # 欲留下的詞類noun
    WHITE_LIST = ['pandas']

    def __init__(self):
        try:
            self.STOPWORDS = nltk.corpus.stopwords.words('english')
        except Exception:
            nltk.download('stopwords')
            self.STOPWORDS = nltk.corpus.stopwords.words('english')
        self.STOPWORDS += ['use', 'python']
        return

    # 文本前置處理
    def content_pre_process(self, raw_text):
        nlp = spacy.load('en_core_web_sm')
        # Step 1. lowercase & tokenize
        doc = nlp(raw_text.lower())

        # Step 2. remove punctuation
        pure_word = [token.text for token in doc if not token.is_punct and token.text != '\n']

        # Step 3. remove stopwords
        filtered_token = [word for word in pure_word if word not in self.STOPWORDS]

        # Step 4. pos_tag filter & lemmatization
        doc = nlp(" ".join(filtered_token))
        lemma = [token.text if token.lemma_ == "-PRON-" or token.text in self.WHITE_LIST
                 else token.lemma_ for token in doc if token.pos_ in self.POS_TAG]

        """
        for token in pure_word:
            if token.pos_ in self.POS_TAG:
                if token.lemma_  == "-PRON-" or token.text in self.WHITE_LIST:
                    lemma.append(token.text)
                else:
                    lemma.append(token.lemma_)
        """
        return lemma, doc

    # 取得文章摘要 - extractive summarization
    def text_summarization(self, raw_text):
        # Step 1.過濾必要token
        keyword, doc = self.content_pre_process(raw_text)  # 保留詞
        freq_word = Counter(keyword)  # 計算關鍵詞的出現次數
        # Step 2.正規化
        max_freq_word = Counter(keyword).most_common(1)[0][1]  # 取得最常出現單詞次數
        for word in freq_word.keys():
            freq_word[word] = freq_word[word] / max_freq_word  # 正規化處理
        # Step 3.sentence加權
        sentence_w = {}
        for sen in doc.sents:
            for word in sen:
                if word.text in freq_word.keys():
                    if sen in sentence_w.keys():
                        sentence_w[sen] += freq_word[word.text]
                    else:
                        sentence_w[sen] = freq_word[word.text]
        # Step 4.nlargest(句子數量, 可迭代之資料(句子&權重), 分別須滿足的條件)
        summarized_sen = nlargest(3, sentence_w, key=sentence_w.get)

        return summarized_sen

    # 利用LDA topic modeling取出關鍵字
    # !-- 需要重新修改方法 --!
    def keyword_extraction(self, data_list):
        comp_preproc_list = [self.content_pre_process(data)[0] for data in data_list]
        keywords = []
        lda_model, dictionary = self.lda_topic_modeling(comp_preproc_list, 5)
        for i in range(0, 5):
            keywords.append([w[0] for w in lda_model.show_topics(formatted=False, num_words=3)[i][1]])
        keywords = list(chain.from_iterable(keywords))
        keywords = list(dict.fromkeys(keywords))
        return keywords

    # LDA topic modeling
    # data -> 2維陣列[[keywords], [keywords], [keywords], ...[]]
    # topic_num = 欲分割成多少數量
    # keyword_num = 取前n關鍵字
    @staticmethod
    def train_lda_model(data, topic_num):

        dictionary = Dictionary(data)
        corpus = [dictionary.doc2bow(t) for t in data]


        # setup logger
        perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
        u_mass_logger = CoherenceMetric(corpus=corpus, dictionary=dictionary,
                                        coherence='u_mass', topn=10, logger='shell')

        # LDA model settings
        lda_model = LdaModel(corpus=corpus, id2word=dictionary,
                             num_topics=topic_num, chunksize=config.CHUNKSIZE, update_every=1,
                             alpha=config.ALPHA, eta=config.ETA, iterations=config.ITERATION,
                             per_word_topics=True, eval_every=1, passes=config.PASSES,
                             callbacks=[perplexity_logger, u_mass_logger])

        return lda_model

    # 關聯度評分(舊方法)
    # input(question keywords, pure word of posts' question)
    def old_similarity_ranking(self, question_key, compare_list):
        nlp = spacy.load('en_core_web_lg')
        # pre-process text
        comp_preproc_list = [self.content_pre_process(content)[0] for content in compare_list]
        # LDA topic modeling
        dictionary = Dictionary(comp_preproc_list)
        corpus = [dictionary.doc2bow(t) for t in comp_preproc_list]

        lda_model = LdaModel(corpus=corpus, id2word=dictionary,
                             num_topics=5, chunksize=config.CHUNKSIZE, update_every=1,
                             alpha=config.ALPHA, eta=config.ETA, iterations=config.ITERATION,
                             per_word_topics=True, eval_every=1, passes=config.PASSES)

        # topic prediction
        q_bow = dictionary.doc2bow(question_key)
        q_topics = sorted(lda_model.get_document_topics(q_bow), key=lambda x: x[1], reverse=True)

        # choose top 3 prediction
        top3_topic_pred = [q_topics[i][0] for i in range(3)]  # top3 topic
        # print(top3_topic_pred)
        top3_prob = [q_topics[i][1] for i in range(3)]  # top3 topic prediction probability
        print(top3_prob)
        top3_topic_keywords = [" ".join([w[0] for w in lda_model.show_topics(formatted=False, num_words=5)[pred_t][1]])
                               for pred_t in top3_topic_pred]
        print(top3_topic_keywords)
        q_vec_list = [nlp(keywords) for keywords in top3_topic_keywords]
        top3pred_sim = [[q_vec.similarity(nlp(" ".join(comp))) for comp in comp_preproc_list] for q_vec in q_vec_list]
        top3pred_sim = np.array(top3pred_sim)
        print(np.array([top3pred_sim[i] * top3_prob[i] for i in range(3)]))
        score_result = np.sum(np.array([top3pred_sim[i] * top3_prob[i] for i in range(3)]), axis=0)
        return score_result

    # eLDA
    @staticmethod
    def train_elda_model(data, topic_num, model_num):
        dictionary = Dictionary(data)
        corpus = [dictionary.doc2bow(t) for t in data]

        # LDA model settings
        elda = EnsembleLda(topic_model_class='lda', corpus=corpus, id2word=dictionary,
                           num_topics=topic_num, num_models=model_num,
                           chunksize=config.CHUNKSIZE,
                           alpha=config.ALPHA, eta=config.ETA, iterations=config.ITERATION,
                           per_word_topics=True, eval_every=1, passes=config.PASSES)

        # measure u_mass coherence
        cm_u_mass_model = CoherenceModel(model=elda, topn=config.TOPIC_TERM_NUM,
                                         corpus=corpus, dictionary=dictionary, coherence="u_mass")
        try:
            print("measure u_mass")
            logging.info("measuring u_mass...")
            u_mass = "{:5.4f}".format(cm_u_mass_model.get_coherence())
            u_mass_per_t = cm_u_mass_model.get_coherence_per_topic()
            logging.info("Coherence u_mass: " + str(u_mass))
            logging.info("Coherence u_mass per-topic: " + str(u_mass_per_t))
        except Exception:
            print("mathematical err...")

        return elda

    @staticmethod
    def evaluate_coherence(model, raw_text, dictionary, corpus):
        # Coherence Measure
        c_v = ""
        u_mass = ""
        c_uci = ""
        c_npmi = ""
        c_v_per_t = []
        u_mass_per_t = []
        c_uci_per_t = []
        c_npmi_per_t = []

        # c_v measure
        cm_c_v_model = CoherenceModel(model=model, topn=8, texts=raw_text, corpus=corpus,
                                      dictionary=dictionary, coherence="c_v")
        try:
            print("measure c_v")
            logging.info("measuring c_v...")
            c_v = "{:5.4f}".format(cm_c_v_model.get_coherence())
            c_v_per_t = cm_c_v_model.get_coherence_per_topic()
            logging.info("Coherence c_v: " + str(c_v))
            logging.info("Coherence c_v per-topic: " + str(c_v_per_t))

        except Exception:
            print("mathematical err...")

        # u_mass measure
        cm_u_mass_model = CoherenceModel(model=model, topn=8, corpus=corpus, dictionary=dictionary, coherence="u_mass")
        try:
            print("measure u_mass")
            logging.info("measuring u_mass...")
            u_mass = "{:5.4f}".format(cm_u_mass_model.get_coherence())
            u_mass_per_t = cm_u_mass_model.get_coherence_per_topic()
            logging.info("Coherence u_mass: " + str(u_mass))
            logging.info("Coherence u_mass per-topic: " + str(u_mass_per_t))
        except Exception:
            print("mathematical err...")

        # c_uci measure
        cm_c_uci_model = CoherenceModel(model=model, topn=8, texts=raw_text, corpus=corpus,
                                        dictionary=dictionary, coherence="c_uci")
        try:
            print("measure c_uci")
            logging.info("measuring c_uci...")
            c_uci = "{:5.4f}".format(cm_c_uci_model.get_coherence())
            c_uci_per_t = cm_c_uci_model.get_coherence_per_topic()
            logging.info("Coherence c_uci: " + str(c_uci))
            logging.info("Coherence c_uci per-topic: " + str(c_uci_per_t))
        except Exception:
            print("mathematical err...")

        # c_npmi measure
        cm_c_npmi_model = CoherenceModel(model=model, topn=8, texts=raw_text, corpus=corpus,
                                         dictionary=dictionary, coherence="c_npmi")
        try:
            print("measure c_npmi")
            logging.info("measuring c_npmi...")
            c_npmi = "{:5.4f}".format(cm_c_npmi_model.get_coherence())
            c_npmi_per_t = cm_c_npmi_model.get_coherence_per_topic()
            logging.info("Coherence c_npmi: " + str(c_uci))
            logging.info("Coherence c_npmi per-topic: " + str(c_uci_per_t))
        except Exception:
            print("mathematical err...")
        # Display result
        return {"c_v": c_v, "u_mass": u_mass, "c_uci": c_uci, "c_npmi": c_npmi,
                "c_v_per_topic": c_v_per_t,
                "u_mass_per_topic": u_mass_per_t,
                "c_uci_per_topic": c_uci_per_t,
                "c_npmi_per_topic": c_npmi_per_t}


def block_ranking(stack_items, qkey):
    a = TextAnalyze()
    ans = [items['answers'] for items in stack_items]

    # data pre-process
    all_content = [[{"id": sing_ans["id"], "content": sing_ans['content']}
                    for sing_ans in q_ans_list] for q_ans_list in ans]
    all_content_flat = list(chain.from_iterable(all_content))
    raw = [t["content"] for t in all_content_flat]

    # similarity ranking
    temp_result = a.old_similarity_ranking(qkey, raw)
    for i in range(len(all_content_flat)):
        all_content_flat[i]["score"] = temp_result[i]
    rank = sorted(all_content_flat, key=lambda data: data["score"], reverse=True)
    return rank


def get_filename():
    with open("testQ.json", "r", encoding='utf-8') as f:
        q = json.load(f)
        q = q['CoreLanguage']
    filelist = []
    for i in os.listdir('stack_data'):
        if "CoreLanguage" in i:
            filelist.append(i)
    return q, sorted(filelist)


if __name__ == "__main__":

    questions, responses = get_filename()
    analyzer = TextAnalyze()
    for idx in range(len(questions)):
        # set loggers
        logging.basicConfig(level=config.LOG_MODE, force=True,
                            filename="logs/old_method/" + str(idx + 1) + "/result.log", filemode='w',
                            format=config.FORMAT, datefmt=config.DATE_FORMAT)
        # get parse posts
        with open("stack_data/" + responses[idx], "r", encoding="utf-8") as raw_file:
            data = json.load(raw_file)
            titles = [i['question']['title'] for i in data]
            raw_file.close()

        # pre-process user question
        key = analyzer.content_pre_process(questions[idx])[0]

        # start block ranking process
        r = block_ranking(data, key)
        for detail in r:
            print(detail)
            logging.info(detail)
