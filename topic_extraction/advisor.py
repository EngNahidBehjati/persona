from os import path, makedirs


class Advisor:
    data_folder_is_set = False

    data_folder = "{}/"
    lang_folder_name = "lang-{}/"
    data_version_folder_name = "data-version-{}/"
    dictionary_version_folder_name = "dict-version-{}/"
    model_version_folder_name = "model-version-{}/"
    pram_version_folder_name = "{}-{}/"
    model_type_folder_name = "{}-model/"
    visualization_folder_name = "visualization/"
    topic_number_folder_name = "Topic-{}/"

    file_with_extension = "{}.{}"
    file_without_extension = "{}"

    @classmethod
    def set_data_folder_path(cls, data_folder: str):
        cls.data_folder = cls.data_folder.format(data_folder)
        cls.data_folder_is_set = True

    @classmethod
    def get_data_folder_path(cls):
        if cls.data_folder_is_set:
            return cls.data_folder
        raise FileNotFoundError("Data file name is not set yet")

    @classmethod
    def get_data_folders_file_path(cls, file_name: str, file_extension: str):
        """
        This is for getting files that are in data folder
        """
        return cls.data_folder + cls.file_with_extension.format(file_name, file_extension)

    @classmethod
    def get_data_file_meta_path(cls):
        return cls.get_data_folders_file_path("meta-of-data-file", "json")

    @classmethod
    def get_language_folders_path(cls, lang: str):
        folder = cls.get_data_folder_path() + cls.lang_folder_name.format(lang)
        return folder

    @classmethod
    def get_data_version_folder_path(cls, lang: str, data_version: int):
        folder = cls.get_language_folders_path(lang)
        folder += cls.data_version_folder_name.format(data_version)
        return folder

    @classmethod
    def get_data_version_folders_file_path(cls, lang: str, data_version: int, file_name: str, file_extension: str):
        folder = cls.get_data_version_folder_path(lang, data_version)
        if path.exists(folder) is False:
            makedirs(folder)
        return folder + cls.file_with_extension.format(file_name, file_extension)

    @classmethod
    def get_dictionary_version_folder_path(cls, lang: str, data_version: int, dictionary_version: float):
        folder = cls.get_data_version_folder_path(lang, data_version)
        folder += cls.dictionary_version_folder_name.format(dictionary_version)
        return folder

    @classmethod
    def get_dictionary_version_folder_file_path(cls, lang: str, data_version: int, dictionary_version: float,
                                                file_name: str, file_extension: str):
        folder = cls.get_dictionary_version_folder_path(lang, data_version, dictionary_version)
        if path.exists(folder) is False:
            makedirs(folder)
        return folder + cls.file_with_extension.format(file_name, file_extension)

    @classmethod
    def get_model_version_folder_path(cls, lang: str, data_version: int,
                                      dictionary_version: float, model_version: str):
        folder = cls.get_dictionary_version_folder_path(lang, data_version, dictionary_version)
        return folder + cls.model_version_folder_name.format(model_version)

    @classmethod
    def get_model_version_folders_file_path(cls, lang: str, data_version: int,
                                            dictionary_version: float, model_version: str,
                                            file_name: str, file_extension: str):
        folder = cls.get_model_version_folder_path(lang, data_version, dictionary_version, model_version)
        if path.exists(folder) is False:
            makedirs(folder)
        return folder + cls.file_with_extension.format(file_name, file_extension)

    @classmethod
    def get_param_version_folder_path(cls, lang: str, data_version: int,
                                      dictionary_version: float, model_version: str, param_name: str, param_version: int):
        folder = cls.get_model_version_folder_path(lang, data_version, dictionary_version, model_version)
        return folder + cls.pram_version_folder_name.format(param_name, param_version)

    @classmethod
    def get_param_version_folders_file_path(cls, lang: str, data_version: int,
                                            dictionary_version: float, model_version: str, param_name: str, param_version: int,
                                            file_name: str, file_extension: str):
        folder = cls.get_param_version_folder_path(lang, data_version, dictionary_version, model_version,
                                                   param_name, param_version)
        if path.exists(folder) is False:
            makedirs(folder)
        return folder + cls.file_with_extension.format(file_name, file_extension)

    @classmethod
    def get_model_type_folder_path(cls, lang: str, data_version: int,
                                   dictionary_version: float, model_version: str, param_name: str, param_version: int,
                                   model_type: str):
        folder = cls.get_param_version_folder_path(lang, data_version, dictionary_version, model_version, param_name, param_version)
        return folder + cls.model_type_folder_name.format(model_type)

    @classmethod
    def get_model_type_folders_file_path(cls, lang: str,
                                         data_version: int, dictionary_version: float, model_version: str,
                                         param_name: str, param_version: int,
                                         model_type: str, file_name: str):
        folder = cls.get_model_type_folder_path(lang, data_version, dictionary_version, model_version,
                                                param_name, param_version, model_type)
        if path.exists(folder) is False:
            makedirs(folder)
        return folder + cls.file_without_extension.format(file_name)

    @classmethod
    def get_visualization_folder_path(cls, lang: str, data_version: int,
                                      dictionary_version: float, model_version: str,
                                      param_name: str, param_version: int,
                                      model_type: str):
        folder = cls.get_model_type_folder_path(lang, data_version, dictionary_version, model_version,
                                                param_name, param_version,
                                                model_type)
        return folder + cls.visualization_folder_name

    @classmethod
    def get_visualization_folder_path_from_topic_version(cls, topic_version_folder_path: str, model_type: str):
        return (topic_version_folder_path + cls.model_type_folder_name.format(model_type) +
                cls.visualization_folder_name)

    @classmethod
    def get_visualization_file_path_from_topic_version(cls, topic_version_folder_path: str,
                                                       model_type: str, file_name: str, file_extension: str):
        folder = cls.get_visualization_folder_path_from_topic_version(topic_version_folder_path, model_type)
        if path.exists(folder) is False:
            makedirs(folder)
        return folder + cls.file_with_extension.format(file_name, file_extension)

    @classmethod
    def get_topic_folder_path_from_topic_version(cls, topic_version_folder_path: str,
                                                 model_type: str, topic_id: int):
        return (cls.get_visualization_folder_path_from_topic_version(topic_version_folder_path, model_type) +
                cls.topic_number_folder_name.format(topic_id))

    @classmethod
    def get_topic_folders_file_path_from_topic_version(cls, topic_version_folder_path: str,
                                                       model_type: str, topic_id: int,
                                                       file_name: str, file_extension: str):
        folder = cls.get_topic_folder_path_from_topic_version(topic_version_folder_path, model_type, topic_id)
        if path.exists(folder) is False:
            makedirs(folder)
        return folder + cls.file_with_extension.format(file_name, file_extension)
