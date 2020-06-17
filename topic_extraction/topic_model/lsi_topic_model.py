import logging
from os import path

from gensim.models import TfidfModel, LsiModel, CoherenceModel

from topic_extraction.advisor import Advisor
from topic_extraction.visualization.model_view import ModelView
from topic_extraction.topic_model.topic_model import TopicModel


class LsiTopicModel(TopicModel):
    def __init__(self,
                 lang,
                 number_of_topics: int,
                 data_version: int, dictionary_version: float, model_version: str,
                 param_name: str, param_version: int,
                 language_processed_data: list,
                 no_below: int,
                 no_above: float,
                 n_most_frequent: int,
                 model_view: bool):
        """
        TopicModel's parameters:
        :param number_of_topics: For its parent
        :param train_version: For its parent
        :param language_processed_data: For its parent
        :param no_below: For its parent
        :param no_above: For its parent
        """
        super().__init__(lang,
                         number_of_topics,
                         data_version, dictionary_version,
                         language_processed_data,
                         no_below,
                         no_above,
                         n_most_frequent)
        self.set_model_type()
        self.visualization = ModelView(lang, data_version, dictionary_version, model_version, param_name, param_version)
        self.get_model(lang, data_version, dictionary_version, model_version,param_name, param_version,
                       language_processed_data, model_view)

    @classmethod
    def set_model_type(cls):
        cls.model_type = "lsi"

    def set_model(self, lang: str, data_version: int, dictionary_version: float, model_version: str,
                  param_name: str, param_version: int,
                  model_file_path: str, language_processed_data: list):
        logging.info("---- Create LSI model ")
        tf_idf = TfidfModel(self.essentials.corpus)
        tf_idf_corpus = tf_idf[self.essentials.corpus]
        model = LsiModel(tf_idf_corpus,
                         id2word=self.essentials.dictionary,
                         num_topics=self.number_of_topics)
        model.save(model_file_path)
        self.model = model
        logging.info("---- LSI model is created")
        metrics = self.get_model_evaluation_metrics(language_processed_data)
        parameters = self.get_model_parameters()
        self.write_model_evaluation_metrics(lang, data_version, dictionary_version, model_version,
                                            param_name, param_version, metrics, parameters)
        return

    def get_model(self, lang: str, data_version: int, dictionary_version: float, model_version: str,
                  param_name: str, param_version: int,
                  language_processed_data: list, model_view: bool):
        logging.info("--- Getting LSI model")
        if self.model is None:
            model_file_path = Advisor.get_model_type_folders_file_path(lang, data_version,
                                                                       dictionary_version, model_version,
                                                                       param_name, param_version,
                                                                       self.model_type, "LSI-model")
            if path.exists(model_file_path):
                logging.info("---- LSI model was created before")
                self.model = LsiModel.load(model_file_path)
            else:
                self.set_model(lang, data_version, dictionary_version, model_version,
                               param_name, param_version,
                               model_file_path,
                               language_processed_data)
        logging.info("--- LSI model captured")
        if model_view:
            self.visualization.get_model_visualizations(self.model_type, self.model, self.essentials.corpus,
                                                        language_processed_data)
        return self.model

    def get_documents_topic(self, document: list):
        document = self.essentials.dictionary.doc2bow(document)
        return self.model[document]

    def get_model_evaluation_metrics(self, language_processed_data: list) -> dict:
        topics = self.model.show_topics(formatted=False, num_topics=-1, num_words=30)
        lsi_topics = [[word for word, prob in topic] for topic_id, topic in topics]
        coherence = CoherenceModel(topics=lsi_topics, corpus=self.essentials.corpus,
                                   dictionary=self.essentials.dictionary,
                                   coherence='c_v', texts=language_processed_data)
        c_v = coherence.get_coherence()
        coherence = CoherenceModel(topics=lsi_topics, corpus=self.essentials.corpus,
                                   dictionary=self.essentials.dictionary,
                                   coherence='u_mass', texts=language_processed_data)
        u_mass = coherence.get_coherence()
        return {"c_v": c_v,
                "u_mass": u_mass}

    def get_model_parameters(self) -> dict:
        return {"common": self.get_common_parameters(),
                "lsi": {"tfidf": True}}
