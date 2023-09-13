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


if __name__ == "__main__":
    print("Text Analysis V.1")
