import logging
from os import path

from gensim.corpora import Dictionary, MmCorpus

from topic_extraction.advisor import Advisor


class TopicModelEssential:
    topic_model_essential = None

    file_types = [["dictionary", "dict"], ["corpus", "mm"]]

    @classmethod
    def init(cls,
             lang: str,
             data_version: int,
             dictionary_version: float,
             language_processed_data: list,
             no_below: int,
             no_above: float,
             n_most_frequent: int):
        if cls.topic_model_essential is None:
            cls.topic_model_essential = TopicModelEssential(lang, data_version, dictionary_version,
                                                            language_processed_data,
                                                            no_below, no_above, n_most_frequent)
        return cls.topic_model_essential

    def __init__(self,
                 lang: str,
                 data_version: int,
                 dictionary_version: float,
                 language_processed_data: list,
                 no_below: int,
                 no_above: float,
                 n_most_frequent: int):
        self.dictionary = None
        self.corpus = None
        self.get_dictionary(lang, data_version, dictionary_version, no_above, no_below, n_most_frequent,
                            language_processed_data)
        self.get_corpus(lang, data_version, dictionary_version, language_processed_data)

    def set_dictionary(self, language_processed_data: list, no_below: int, no_above: float, n_most_frequent: int,
                       dictionary_file_path):
        logging.info("---- Creating dictionary from processed data")
        dic = Dictionary(language_processed_data)
        dic.filter_n_most_frequent(n_most_frequent)
        dic.filter_extremes(no_below=no_below, no_above=no_above)
        dic.save(dictionary_file_path)
        self.dictionary = dic
        logging.info("---- Dictionary is created")
        return

    def get_dictionary(self, lang, data_version, dictionary_version, no_above, no_below, n_most_frequent,
                       language_processed_data: list):
        logging.info("--- Getting dictionary")
        if self.dictionary is None:
            dictionary_file_path = Advisor.get_dictionary_version_folder_file_path(lang, data_version,
                                                                                   dictionary_version,
                                                                                   self.file_types[0][0],
                                                                                   self.file_types[0][1])
            if path.exists(dictionary_file_path):
                logging.info("---- Dictionary was created before")
                self.dictionary = Dictionary.load(dictionary_file_path)
            else:
                self.set_dictionary(language_processed_data, no_below, no_above, n_most_frequent, dictionary_file_path)
        logging.info("--- Dictionary captured")
        return

    def get_corpus(self, lang, data_version, dictionary_version, language_processed_data: list = None):
        logging.info("--- Getting corpus")
        if self.corpus is None:
            corpus_file_path = Advisor.get_dictionary_version_folder_file_path(lang, data_version, dictionary_version,
                                                                               self.file_types[1][0],
                                                                               self.file_types[1][1])
            if path.exists(corpus_file_path):
                logging.info("---- Corpus was created before")
                self.corpus = list(MmCorpus(corpus_file_path))
            else:
                self.set_corpus(language_processed_data, corpus_file_path)
        logging.info("--- Corpus captured")
        return

    def set_corpus(self, language_processed_data: list, corpus_file_path: str):
        logging.info("---- Creating corpus from processed data")
        corpus = [self.dictionary.doc2bow(list_of_words_of_doc)
                  for list_of_words_of_doc in language_processed_data]
        MmCorpus.serialize(corpus_file_path, corpus)
        self.corpus = corpus
        logging.info("---- Corpus is created")
        return
