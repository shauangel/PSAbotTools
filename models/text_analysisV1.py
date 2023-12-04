#!/usr/bin/env python3
# Created in 2023
# Block Analysis v.1

# basic tools
import math
import requests
import config
import logging
import nltk
import spacy
from itertools import chain
from collections import Counter
# LDA model
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel, EnsembleLda, CoherenceModel
from gensim.models.callbacks import PerplexityMetric, CoherenceMetric
# word embeddings
import tensorflow as tf
import tensorflow_hub as hub
import embeddings


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


def block_ranking(stack_items, question):
    # 1. data pre-process
    print("Step 1. Data pre-process")
    analyzer = TextAnalyze()
    # q_corpus = [analyzer.content_pre_process(i['question']['content'])[0]
    #            for i in stack_items]
    ans_corpus = [[analyzer.content_pre_process(ans['content'])[0] for ans in i['answers']]
                  for i in stack_items]
    print(ans_corpus)
    # list(chain.from_iterable([analyzer.content_pre_process(ans['content'])[0]

    # 2. generate training data
    print("Step 2. Generating train data")
    training_data = [list(chain.from_iterable(ans)) for ans in ans_corpus]
    print(training_data)
    dictionary = Dictionary(training_data)
    # corpus = [dictionary.doc2bow(text) for text in ans_corpus]

    # 3. train topic model
    print("Step 3. Train topic model")
    # model = analyzer.train_lda_model(training_data, config.TOPIC_NUM)  # LDA
    model = analyzer.train_elda_model(training_data, config.TOPIC_NUM, 5)  # eLDA
    topics = [t for t in model.print_topics(config.TOPIC_TERM_NUM)]
    print(topics)

    # 4. apply word embeddings
    print("Step 4. Word embeddings")
    embed = hub.KerasLayer("embeds/Wiki-words-250_2",
                           input_shape=[],
                           dtype=tf.string,
                           trainable=True,
                           name="Word_Embedding_Layer")
    # user_q_vector = embed([question])  # user question embed
    # q_title_vectors = embed([i['question']['title'] for i in stack_items])  # post questions embeds

    user_q_vector = embeddings.embeds()
    q_title_vectors = embed([i['question']['title'] for i in stack_items])  # post questions embeds


    # 5. calculate similarity between questions
    print("Step 5. Calculate similarity")
    question_sim = [tf.losses.CosineSimilarity()(user_q_vector[0], t_vec).numpy() for t_vec in q_title_vectors]
    abs_question_sim = [abs(1 + sim) for sim in question_sim]
    print(abs_question_sim)

    # 6. predict the topic distribution of user question & blocks (which topic it might belong to)
    print("Step 6. Predict topic distribution")
    user_q_topic_dist = model[dictionary.doc2bow(analyzer.content_pre_process(question)[0])][0]
    user_q_topic_dist = {t[0]: t[1] for t in user_q_topic_dist}
    print(user_q_topic_dist)
    ans_blocks_dist = []
    for ans in ans_corpus:
        a_corpus = [dictionary.doc2bow(terms) for terms in ans]
        ans_blocks_dist.append([model[block][0] for block in a_corpus])

    # 7. calculate topic similarity between user question & blocks
    print("Step 7. Rankings")
    block_prob = []
    block_terms = []
    block_count = 0
    for d in range(len(stack_items)):
        temp = []
        ans_score = [ans['score'] for ans in stack_items[d]['answers']]
        for a_idx in range(len(stack_items[d]['answers'])):
            block_count += len(stack_items[d]['answers'])
            prob = math.log(sum([user_q_topic_dist[t[0]] * t[1] for t in ans_blocks_dist[d][a_idx]]) / 3)
            if ans_score[a_idx] >= 0:
                block_prob.append({"id": stack_items[d]['answers'][a_idx]['id'],
                                   "prob": prob,
                                   "q_sim": str(abs_question_sim[d]),
                                   "b_score": str(prob*abs_question_sim[d]),
                                   "so_score": str(ans_score[a_idx])})
            # prepare terms for keyword extraction
            temp.append(Counter(ans_corpus[d][a_idx]))
        block_terms.append(temp)
    print("total block count: " + str(block_count))
    print("reduced block: " + str(len(block_prob)))

    # 8. Rank blocks
    rank = sorted(block_prob, key=lambda x: x['b_score'])
    return rank[:10]


if __name__ == "__main__":
    print("Text Analysis V.1")