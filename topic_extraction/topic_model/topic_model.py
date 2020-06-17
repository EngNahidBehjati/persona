import json
from abc import abstractmethod, ABC
from topic_extraction.advisor import Advisor
from topic_extraction.topic_model.topic_model_essential import TopicModelEssential

"""
we have a list of list -> collection of words of different pages for specific language
Approaches	to	finding	Latent	Topics:
    • Non-Negative	Matrix	factorization.	
    • Latent	Dirichelt Allocation	(LDA)
    • Probabilistic	Latent	Semantic	Indexing	(pLSI)
    • Correlated	Topic	Model	(CTM)
"""


class TopicModel(ABC):
    """
    This is an abstract class that any topic_extraction_model class in this project should inherit from it
    """
    model_type = str()

    def __init__(self,
                 lang: str,
                 number_of_topics: int,
                 data_version: int,
                 dictionary_version: float,
                 language_processed_data: list,
                 no_below: int,
                 no_above: float,
                 n_most_frequent: int):
        """
        :param lang: the language that model is for
        :param number_of_topics: number of all documents's topics
        :param data_version: the state of topic_model's parameters (data and it's model's parameters)
        :param language_processed_data: preprocessed documents(the output of text preprocess step) is
        a list of processed documents that each processed document is a list of processed words
        :param no_below: this parameter is used for creating dictionary. says words that are
        in less than #no_below documents should not be in the dictionary
        :param no_above: this parameter is used for creating dictionary. says words that are
        in more than #no_above documents should not be in the dictionary
        """
        self.no_below = no_below
        self.no_above = no_above
        self.n_most_frequent = n_most_frequent
        self.number_of_topics = number_of_topics

        self.essentials = TopicModelEssential.init(lang, data_version, dictionary_version, language_processed_data,
                                                   no_below, no_above, n_most_frequent)

        self.model = None

    def write_model_evaluation_metrics(self, lang: str, data_version, dictionary_version, model_version,
                                       param_name: str, param_version: int,
                                       metrics: dict, model_parameters: dict):
        model_evaluation_file_path = Advisor.get_model_type_folders_file_path(lang, data_version,
                                                                              dictionary_version, model_version,
                                                                              param_name, param_version,
                                                                              self.model_type, "evaluation")
        model_version_file_path = Advisor.get_model_type_folders_file_path(lang, data_version,
                                                                           dictionary_version, model_version,
                                                                           param_name, param_version,
                                                                           self.model_type, "meta.json")
        with open(model_evaluation_file_path, "w") as json_file:
            json.dump(metrics, json_file, indent=4)
            json_file.close()

        with open(model_version_file_path, "w") as json_file:
            json.dump(model_parameters, json_file, indent=4)
            json_file.close()
        return

    def get_common_parameters(self) -> dict:
        return {"topic_number" : self.number_of_topics,
                "no_below": self.no_below,
                "no_above": self.no_above,
                "n_most_frequent": self.n_most_frequent}

    @classmethod
    @abstractmethod
    def set_model_type(cls):
        """
        this method must run in child class's init function before any thing else(ofcourse after parent's init )
        it sets child's model name
        :return: None
        """
        pass

    @abstractmethod
    def set_model(self, lang: str, data_version: int, dictionary_version: float, model_version: str,
                  param_name: str, param_version: int,
                  model_file_path: str, language_processed_data: list):
        pass

    @abstractmethod
    def get_model(self, lang: str, data_version: int, dictionary_version: float, model_version: str,
                  param_name: str, param_version: int,
                  language_processed_data: list, model_view: bool):
        pass

    @abstractmethod
    def get_documents_topic(self, document: list):
        pass

    @abstractmethod
    def get_model_evaluation_metrics(self, language_processed_data: list) -> dict:
        pass

    @abstractmethod
    def get_model_parameters(self) -> dict:
        pass
