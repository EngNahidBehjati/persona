import logging
import sys

from topic_extraction.advisor import Advisor
from topic_extraction.data.text_preprocessor import TextPreprocessor
from topic_extraction.manager.execution import Execution
from topic_extraction.manager.interfaces.data_params import DataParams
from topic_extraction.manager.interfaces.dictionary_params import DictionaryParams
from topic_extraction.manager.interfaces.exe_params import ExeParams
from topic_extraction.manager.interfaces.model_params import ModelParams
from topic_extraction.manager.interfaces.request_params import RequestParams
from topic_extraction.manager.interfaces.versionify_params import VersionifyParams
from topic_extraction.topic_model.hdp_topic_model import HdpTopicModel
from topic_extraction.topic_model.lda_mallet_topic_model import LdaMalletTopicModel
from topic_extraction.topic_model.lda_topic_model import LdaTopicModel
from topic_extraction.topic_model.lsi_topic_model import LsiTopicModel
from topic_extraction.visualization.comparative_view import ComparativeView

logging.basicConfig(format='%(asctime)s --%(levelname)s %(message)s', level=logging.INFO)


class TopicExtractionHandler:

    def __init__(self):
        self.comparative_view = ComparativeView()

    def get_topic_extraction(self, passed_args):
        passed_args, data_args, dictionary_args, model_args, version_args, request_args = ExeParams.get_parameters(passed_args)
        Advisor.set_data_folder_path(data_args[DataParams.data_folder_path])
        processed_data = self._get_processed_data(data_args, version_args)

        all_models_args_values = ModelParams.get_possible_model_params_values()

        for requested_model_type in request_args[RequestParams.requested_models]:
            for lang in processed_data:
                if lang in request_args[RequestParams.requested_langs]:
                    # ModelParams.set_model_params_value(model_args)
                    logging.info("- Get Topic extraction for '%s' language" % lang)
                    for test_param in all_models_args_values:
                        metrics = self._get_requested_models_metrics(requested_model_type, lang,
                                                                     processed_data[lang],
                                                                     dictionary_args, all_models_args_values[test_param],
                                                                     version_args, request_args, test_param)
                        metrics = self._make_param_metrics_ready_to_plot(metrics)
                        self.comparative_view.plot_metrics(metrics, test_param, lang, requested_model_type,
                                                           version_args[VersionifyParams.data_version],
                                                           version_args[VersionifyParams.dictionary_version],
                                                           version_args[VersionifyParams.model_version])
        return

    @classmethod
    def _get_processed_data(cls, data_args: dict, version_args: dict):
        return TextPreprocessor.get_processed_data(data_args[DataParams.data_file_name],
                                                   data_args[DataParams.data_file_extension],
                                                   version_args[VersionifyParams.data_version],
                                                   data_args[DataParams.data_file_type],
                                                   data_args[DataParams.include_tags],
                                                   data_args[DataParams.exclude_tags])

    @classmethod
    def _make_param_metrics_ready_to_plot(cls, param_metrics):
        metrics = dict()
        if param_metrics:
            for param_test_value in param_metrics:
                for metric in param_metrics[param_test_value]:
                    if metric not in metrics:
                        metrics[metric] = list()
                    metrics[metric].append([param_test_value, param_metrics[param_test_value][metric]])
        return metrics

    def _get_requested_models_metrics(self, model_name: str, lang: str, language_processed_data: list,
                                      dictionary_args, model_args_possible_values: list, version_args, request_args,
                                      test_param_name):
        if model_name == "lda":
            return self._get_lda_model_metrics(lang, language_processed_data, dictionary_args, model_args_possible_values,
                                               version_args, request_args, test_param_name)
        if model_name == "mallet":
            return self._get_mallet_model_metrics(lang, language_processed_data, dictionary_args, model_args_possible_values,
                                                  version_args, request_args, test_param_name)
        if model_name == "lsi":
            return self._get_lsi_model_metrics(lang, language_processed_data, dictionary_args, model_args_possible_values,
                                               version_args, request_args, test_param_name)
        if model_name == "hdp":
            return self._get_hdp_model_metrics(lang, language_processed_data, dictionary_args, model_args_possible_values,
                                               version_args, request_args, test_param_name)

    @classmethod
    def _get_lda_model_metrics(cls, lang: str, language_processed_data: list, dictionary_args,
                               model_args_possible_values: list, version_args, request_args, test_param_name):
        logging.info("--- LDA : Evaluating topic number")
        test_param_metrics = dict()
        for index, model_args in enumerate(model_args_possible_values):
            logging.info("-- Creating LDA topic model by topic number '%d'" % model_args[ModelParams.number_of_topics.name])
            model = LdaTopicModel(lang, model_args[ModelParams.number_of_topics.name],
                                  version_args[VersionifyParams.data_version],
                                  version_args[VersionifyParams.dictionary_version],
                                  version_args[VersionifyParams.model_version],
                                  test_param_name, model_args[test_param_name],
                                  language_processed_data, dictionary_args[DictionaryParams.no_below],
                                  dictionary_args[DictionaryParams.no_above],
                                  dictionary_args[DictionaryParams.n_most_frequent],
                                  model_args[ModelParams.chunk_size.name], model_args[ModelParams.alpha.name],
                                  model_args[ModelParams.beta.name],
                                  model_args[ModelParams.iterations.name], model_args[ModelParams.passes.name],
                                  model_args[ModelParams.eval_every.name],
                                  request_args[RequestParams.model_view])
            test_param_metrics[model_args[test_param_name]] = model.get_model_evaluation_metrics(language_processed_data)
        return test_param_metrics

    @classmethod
    def _get_mallet_model_metrics(cls, lang: str, language_processed_data: list, dictionary_args,
                                  model_args_possible_values: list, version_args, request_args, test_param_name):
        logging.info("--- MALLET : Evaluating topic number")
        metrics = dict()
        for index, model_args in enumerate(model_args_possible_values):
            logging.info("-- Creating LDA topic model by topic number '%d'" % model_args[ModelParams.number_of_topics.name])
            model = LdaMalletTopicModel(lang, model_args[ModelParams.number_of_topics.name],
                                        version_args[VersionifyParams.data_version],
                                        version_args[VersionifyParams.dictionary_version],
                                        version_args[VersionifyParams.model_version],
                                        test_param_name, model_args[test_param_name],
                                        language_processed_data, dictionary_args[DictionaryParams.no_below],
                                        dictionary_args[DictionaryParams.no_above],
                                        dictionary_args[DictionaryParams.n_most_frequent],
                                        model_args[ModelParams.chunk_size.name], model_args[ModelParams.alpha.name],
                                        model_args[ModelParams.beta.name],
                                        model_args[ModelParams.iterations.name], model_args[ModelParams.passes.name],
                                        model_args[ModelParams.eval_every.name],
                                        request_args[RequestParams.model_view])
            metrics[model_args[test_param_name]] = model.get_model_evaluation_metrics(language_processed_data)
        return metrics

    @classmethod
    def _get_lsi_model_metrics(cls, lang: str, language_processed_data: list, dictionary_args,
                               model_args_possible_values: list, version_args, request_args, test_param_name):
        logging.info("--- LSI : Evaluating topic number")
        metrics = dict()
        for index, model_args in enumerate(model_args_possible_values):
            # model_version = "%s-%s-%d" % (
            #     version_args[VersionifyParams.model_version], test_param_name, model_args[test_param_name])
            logging.info("-- Creating LDA topic model by topic number '%d'" % model_args[ModelParams.number_of_topics.name])
            model = LsiTopicModel(lang, model_args[ModelParams.number_of_topics.name],
                                  version_args[VersionifyParams.data_version],
                                  version_args[VersionifyParams.dictionary_version],
                                  version_args[VersionifyParams.model_version],
                                  test_param_name, model_args[test_param_name],
                                  language_processed_data, dictionary_args[DictionaryParams.no_below],
                                  dictionary_args[DictionaryParams.no_above],
                                  dictionary_args[DictionaryParams.n_most_frequent],
                                  request_args[RequestParams.model_view])
            metrics[model_args[test_param_name]] = model.get_model_evaluation_metrics(language_processed_data)
        return metrics

    @classmethod
    def _get_hdp_model_metrics(cls, lang: str, language_processed_data: list, dictionary_args,
                               model_args_possible_values: list, version_args, request_args, test_param_name):
        logging.info("--- HDP : Evaluating topic number")
        metrics = dict()
        for index, model_args in enumerate(model_args_possible_values):
            logging.info("-- Creating LDA topic model by topic number '%d'" % model_args[ModelParams.number_of_topics.name])
            model = HdpTopicModel(lang,
                                  version_args[VersionifyParams.data_version],
                                  version_args[VersionifyParams.dictionary_version],
                                  version_args[VersionifyParams.model_version],
                                  test_param_name, model_args[test_param_name],
                                  language_processed_data, dictionary_args[DictionaryParams.no_below],
                                  dictionary_args[DictionaryParams.no_above],
                                  model_args[ModelParams.chunk_size.name], model_args[ModelParams.alpha.name],
                                  model_args[ModelParams.beta.name],dictionary_args[DictionaryParams.n_most_frequent],
                                  request_args[RequestParams.model_view])
            metrics[model_args[test_param_name]] = model.get_model_evaluation_metrics(language_processed_data)
        return


if __name__ == '__main__':
    exe = Execution()
    args = exe.parse_topic_extraction_command_line(sys.argv[1:])
    TopicExtractionHandler().get_topic_extraction(args)
    logging.info("---------- Done ----------")
