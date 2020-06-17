import logging
from os import path

from gensim.models import HdpModel, CoherenceModel

from topic_extraction.advisor import Advisor
from topic_extraction.visualization.model_view import ModelView
from topic_extraction.topic_model.topic_model import TopicModel


class HdpTopicModel(TopicModel):
    def __init__(self,
                 lang: str,
                 data_version: int, dictionary_version: float, model_version: str,
                 param_name: str, param_version: int,
                 language_processed_data: list,
                 no_below: int,
                 no_above: float,
                 chunk_size: int,
                 alpha: float,
                 beta: float,
                 n_most_frequent: int,
                 model_view: bool):
        """
        TopicModel's Parameters:
        :param data_folder_train_version_related_path: For its parent
        :param train_version: For its parent
        :param language_processed_data: For its parent
        :param no_below: For its parent
        :param no_above: For its parent


        LdaTopicModel's Parameters:
        :param chunk_size:
        :param alpha:
        :param beta:
        """
        super().__init__(lang,
                         0,
                         data_version, dictionary_version,
                         language_processed_data,
                         no_below,
                         no_above,
                         n_most_frequent)
        self.beta = beta
        self.alpha = alpha
        self.chunk_size = chunk_size

        self.set_model_type()
        self.visualization = ModelView(lang, data_version, dictionary_version, model_version, param_name, param_version)
        self.get_model(lang, data_version, dictionary_version, model_version, param_name, param_version, language_processed_data, model_view)

    @classmethod
    def set_model_type(cls):
        cls.model_type = "hdp"
        return

    def set_model(self, lang: str, data_version: int, dictionary_version: float, model_version: str, param_name: str, param_version: int,
                  model_file_path: str, language_processed_data: list):
        # Make a index to word dictionary.
        logging.info("---- Creating HDP model")
        temp = self.essentials.dictionary[0]
        model = HdpModel(corpus=self.essentials.corpus, id2word=self.essentials.dictionary.id2token)
                         # , alpha="symmetric",
                         # eta=self.beta, chunksize=self.chunk_size)
        model.save(model_file_path)
        self.model = model
        logging.info("---- HDP model is created")

        metrics = self.get_model_evaluation_metrics(language_processed_data)
        parameters = self.get_model_parameters()
        self.write_model_evaluation_metrics(lang, data_version, dictionary_version, model_version, param_name, param_version, metrics, parameters)
        return

    def get_model(self, lang: str, data_version: int, dictionary_version: float, model_version: str,param_name: str, param_version: int,
                  language_processed_data: list, model_view: bool):
        if self.model is None:
            logging.info("--- Getting HDP model")
            model_file_path = Advisor.get_model_type_folders_file_path(lang, data_version,
                                                                       dictionary_version, model_version, param_name, param_version,
                                                                       self.model_type, "HDP-model")
            if path.exists(model_file_path):

                self.model = HdpModel.load(model_file_path)
            else:
                logging.info("---- HDP model was crated before")
                self.set_model(lang, data_version, dictionary_version, model_version,param_name, param_version, model_file_path,
                               language_processed_data)
        logging.info("--- HDP model captured")
        if model_view:
            self.visualization.get_model_visualizations(self.model_type, self.model, self.essentials.corpus,
                                                        language_processed_data)
        return self.model

    def get_documents_topic(self, document: list):
        return self.model[document]

    def hdp_to_lda(self):
        return self.model.hdp_to_lda()

    # def get_model_evaluation_metrics(self) -> dict:
    #     pass

    def get_model_evaluation_metrics(self, language_processed_data: list) -> dict:
        topics = self.model.show_topics(formatted=False)
        hdp_topics = [[word for word, prob in topic] for topic_id, topic in topics]
        coherence = CoherenceModel(topics=hdp_topics, corpus=self.essentials.corpus,
                                   dictionary=self.essentials.dictionary,
                                   coherence='c_v', texts=language_processed_data, model=self.model)
        c_v = coherence.get_coherence()
        coherence = CoherenceModel(topics=hdp_topics, corpus=self.essentials.corpus,
                                   dictionary=self.essentials.dictionary,
                                   coherence='u_mass', texts=language_processed_data)
        u_mass = coherence.get_coherence()

        return {"c_v": c_v,
                "u_mass": u_mass}

    def get_model_parameters(self) -> dict:
        return {"common": self.get_common_parameters(),
                "hdp": {"chunk_size": self.chunk_size,
                        "alpha": self.alpha,
                        "beta": self.beta}}
