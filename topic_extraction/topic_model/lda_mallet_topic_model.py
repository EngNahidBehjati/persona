import logging
import os.path

from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet

from topic_extraction.advisor import Advisor
from topic_extraction.visualization.model_view import ModelView
from topic_extraction.topic_model.topic_model import TopicModel


class LdaMalletTopicModel(TopicModel):
    def __init__(self,
                 lang: str,
                 number_of_topics: int,
                 data_version: int, dictionary_version: float, model_version: str,
                 param_name: str, param_version: int,
                 language_processed_data: list,
                 no_below: int,
                 no_above: float,
                 n_most_frequent: int,
                 chunk_size: int = 2000,
                 alpha: float = 'auto',
                 beta: float = 'auto',
                 iterations: int = 400,
                 passes: int = 20,
                 eval_every: int = 10,
                 model_view: bool = False):
        super().__init__(lang,
                         number_of_topics,
                         data_version, dictionary_version,
                         language_processed_data,
                         no_below,
                         no_above,
                         n_most_frequent)
        self.beta = beta
        self.alpha = alpha
        self.passes = passes
        self.iterations = iterations
        self.chunk_size = chunk_size
        self.eval_every = eval_every

        self.set_model_type()
        self.visualization = ModelView(lang, data_version, dictionary_version, model_version, param_name, param_version)
        self.get_model(lang, data_version, dictionary_version, model_version, param_name, param_version, language_processed_data, model_view)

    @classmethod
    def set_model_type(cls):
        cls.model_type = "mallet"

    def set_model(self, lang: str, data_version: int, dictionary_version: float, model_version: str, param_name: str, param_version: int,
                  model_file_path: str, language_processed_data: list):
        my_path = os.path.abspath(os.path.dirname(__file__))
        logging.info("---- Creating LDA Mallet model")
        logging.info("------ Getting LDA Mallet model file")
        mallet_path = os.path.join(my_path, "../../statics/mallet-2.0.8/bin/mallet")
        temp = self.essentials.dictionary[0]
        model = LdaMallet(mallet_path,
                          corpus=self.essentials.corpus, num_topics=self.number_of_topics,
                          id2word=self.essentials.dictionary.id2token)
        model.save(model_file_path)
        self.model = model
        logging.info("---- LDA Mallet model is created")

        metrics = self.get_model_evaluation_metrics(language_processed_data)
        parameters = self.get_model_parameters()
        self.write_model_evaluation_metrics(lang, data_version, dictionary_version, model_version,param_name, param_version, metrics, parameters)
        return

    def get_model(self, lang: str, data_version: int, dictionary_version: float, model_version: str,param_name: str, param_version: int,
                  language_processed_data: list, model_view: bool):
        logging.info("--- Getting LDA Mallet model")
        if self.model is None:
            model_file_path = Advisor.get_model_type_folders_file_path(lang, data_version,
                                                                       dictionary_version, model_version, param_name, param_version,
                                                                       self.model_type, "MLDA-model")
            if os.path.exists(model_file_path):
                self.model = LdaMallet.load(model_file_path)
            else:
                logging.info("---- LDA Mallet model was crated before")
                self.set_model(lang, data_version, dictionary_version, model_version, param_name, param_version, model_file_path,
                               language_processed_data)
        logging.info("--- LDA Mallet model captured")
        if model_view:
            self.visualization.get_model_visualizations(self.model_type, self.model, self.essentials.corpus,
                                                        language_processed_data)
        return self.model

    def get_documents_topic(self, document: list):
        document = self.essentials.dictionary.doc2bow(document)
        return self.model[document]

    def get_model_evaluation_metrics(self, language_processed_data: list) -> dict:
        topics = self.model.show_topics(formatted=False, num_topics=-1, num_words=30)
        lda_m_topics = [[word for word, prob in topic] for topic_id, topic in topics]

        coherence = CoherenceModel(topics=lda_m_topics, corpus=self.essentials.corpus,
                                   dictionary=self.essentials.dictionary,
                                   coherence='c_v', texts=language_processed_data)
        c_v = coherence.get_coherence()
        coherence = CoherenceModel(topics=lda_m_topics, corpus=self.essentials.corpus,
                                   dictionary=self.essentials.dictionary,
                                   coherence='u_mass', texts=language_processed_data)
        u_mass = coherence.get_coherence()

        return {"c_v": c_v,
                "u_mass": u_mass}

    def get_model_parameters(self) -> dict:
        return {"common": self.get_common_parameters(),
                "lda_Mallet": {"chunk_size": self.chunk_size,
                               "alpha": self.alpha,
                               "beta": self.beta,
                               "iteration": self.iterations,
                               "passes": self.passes,
                               "eval_every": self.eval_every}}


"topic = (self.dictionary.id2token[_id] for _id in topic)"
