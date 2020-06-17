from topic_extraction.manager.interfaces.data_params import DataParams
from topic_extraction.manager.interfaces.model_params import ModelParams
from topic_extraction.manager.interfaces.dictionary_params import DictionaryParams
from topic_extraction.manager.interfaces.request_params import RequestParams
from topic_extraction.manager.interfaces.versionify_params import VersionifyParams


class ExeParams(DataParams, DictionaryParams, ModelParams, VersionifyParams, RequestParams):

    @classmethod
    def extract_exe_parameters_as_dict(cls, args):
        passed_args = dict()
        args = args._get_kwargs()
        for param_name, param_value in args:
            passed_args[param_name] = param_value
        return passed_args

    @classmethod
    def get_data_parameters(cls, passed_args: dict):
        data_params = dict()
        for data_param in cls.get_data_params_as_list():
            if data_param in passed_args:
                data_params[data_param] = passed_args[data_param]
            else:
                data_params[data_param] = None
        return data_params

    @classmethod
    def get_dictionary_parameters(cls, passed_args: dict):
        dictionary_params = dict()
        for dict_param in cls.get_dictionary_params_as_list():
            if dict_param in passed_args:
                dictionary_params[dict_param] = passed_args[dict_param]
            else:
                dictionary_params[dict_param] = None
        return dictionary_params

    @classmethod
    def get_model_parameters(cls, passed_args: dict):
        model_params = dict()
        for model_param in cls.get_model_params_as_list():
            if model_param in passed_args:
                model_params[model_param] = passed_args[model_param]
            else:
                model_params[model_param] = None
        ModelParams.set_model_params_value(model_params)
        return model_params

    @classmethod
    def get_versionify_parameters(cls, passed_args: dict):
        versionify_args = dict()
        for version in cls.get_versionify_params_as_list():
            if version in passed_args:
                versionify_args[version] = passed_args[version]
            else:
                versionify_args[version] = None
        return versionify_args

    @classmethod
    def get_request_parameters(cls, passed_args: dict):
        request_args = dict()
        for request in cls.get_request_params_as_list():
            if request in passed_args:
                request_args[request] = passed_args[request]
            else:
                request_args[request] = None
        return request_args

    @classmethod
    def get_parameters(cls, args):
        args_as_dict = cls.extract_exe_parameters_as_dict(args)
        data_args = cls.get_data_parameters(args_as_dict)
        dictionary_args = cls.get_dictionary_parameters(args_as_dict)
        model_args = cls.get_model_parameters(args_as_dict)
        versionify_args = cls.get_versionify_parameters(args_as_dict)
        request_args = cls.get_request_parameters(args_as_dict)
        return args_as_dict, data_args, dictionary_args, model_args, versionify_args, request_args


