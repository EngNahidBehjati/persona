import json
import logging
from os import path

import spacy
from spacy.tokens.token import Token

from topic_extraction.advisor import Advisor
from topic_extraction.data.data_provider import DataProvider

spacy.prefer_gpu()


class TextPreprocessor:
    lang_models = {
        "en": "en_core_web_lg",  # English
        "de": "de_core_news_md",  # German
        "fr": "fr_core_news_md",  # French
        "es": "es_core_news_md",  # Spanish
        "el": "el_core_news_md",  # Greek
        "pt": "pt_core_news_sm",  # Portuguese
        "it": "it_core_news_sm",  # Italian
        "nl": "nl_core_news_sm",  # Dutch
        "nb": "nb_core_news_sm",  # Norwegian Bokmål
        "lt": "lt_core_news_sm"  # Lithuanian
    }

    objects_of_TextPreprocessor_for_each_lang = dict()

    @staticmethod
    def __write_processed_data(data_file_name: str, processed_data: dict, version: int):
        """
        :param data_file_name: processed data's file name
        :param processed_data: the data that we want to write it down
        :param version: the version of data
        """
        for lang in processed_data:
            processed_data_file_path = Advisor.get_data_version_folders_file_path(lang, version, data_file_name, "json")

            with open(processed_data_file_path, 'w') as json_file:
                json.dump(processed_data[lang], json_file)
                json_file.close()
        return

    @staticmethod
    def __write_meta_data(language_list: list, tags: dict, data_version: int, lang_length: dict):
        data_file_meta_path = Advisor.get_data_file_meta_path()
        data_file_meta_content = {"languages": language_list,
                                  "lang_length": lang_length}

        data_process_version_meta_content = {
                                             "tags": tags, "token_ validation": {"must": ["alpha"],
                                                                                 "must Not":
                                                                                     ["stop_word", "space", "bracket",
                                                                                      "currency",
                                                                                      "url", "email", "number",
                                                                                      "verb"]}}
        with open(data_file_meta_path, "w") as json_file:
            json.dump(data_file_meta_content, json_file, indent=4)
            json_file.close()
        for lang in language_list:
            data_version_meta_path = Advisor.get_data_version_folders_file_path(lang, data_version,
                                                                                "data-process-mata", "json")
            with open(data_version_meta_path, "w") as json_file:
                json.dump(data_process_version_meta_content, json_file, indent=4)
                json_file.close()
        return

    @staticmethod
    def __read_data_meta_file():
        meta_file_path = Advisor.get_data_file_meta_path()
        if path.exists(meta_file_path):
            with open(meta_file_path, "r") as json_file:
                meta = json.load(json_file)
                json_file.close()
            return meta
        return None

    @staticmethod
    def __read_processed_data(data_file_name: str, version: int):
        """
        :param data_file_name: processed data's file name
        :param version: the data that we want to read it
        :return: (dict format) the processed data that we wrote it down before
        """
        meta = TextPreprocessor.__read_data_meta_file()
        if meta is None:
            return None
        processed_data = dict()
        for lang in meta["languages"]:
            processed_data_file_name = Advisor.get_data_version_folders_file_path(lang, version, data_file_name, "json")
            if path.exists(processed_data_file_name) is False:
                return None
            with open(processed_data_file_name) as json_file:
                processed_data[lang] = json.load(json_file)
                json_file.close()
        return processed_data

    def __init__(self, lang: str):
        """
        :raise This should not be called, instead "get_text_process_related_lang_object" must be called
        :param lang: the language of text
        """
        self.lang = lang
        self.model = self.__load_suitable_model(lang)
        self.model.max_length = 20000000
        return

    @classmethod
    def __is_verb(cls, token: Token) -> bool:
        if token.pos_ == "VERB":
            return True
        return False

    @classmethod
    def __is_adj(cls, token: Token):
        if token.pos_ == "ADJ":
            return True
        return False

    @classmethod
    def __is_adv(cls, token: Token):
        if token.pos_ == "ADV":
            return True
        return False

    @classmethod
    def __is_name_entity(cls, token: Token):
        if len(token.ent_type_) > 0:
            return True
        return False

    @classmethod
    def __is_token_valid(cls, token: Token) -> bool:
        if token.is_stop is True or \
                token.is_space is True or \
                token.is_currency is True or \
                token.is_bracket is True or \
                token.is_alpha is False or \
                token.like_url is True or \
                token.like_email is True or \
                token.like_num is True or \
                cls.__is_verb(token):
            # cls.__is_adj(token) or \
            # cls.__is_adv(token)

            return False
        return True

    @classmethod
    def __load_suitable_model(cls, lang: str):
        """
        :description: Load spacy's model
        :param lang: the model's language
        :return: a model that support the language
        """
        if lang in cls.lang_models:
            return spacy.load(cls.lang_models[lang])
        else:
            raise Exception("Language {} is not supported yet!!".format(lang))

    @classmethod
    def __process_raw_data_language_classified(cls, raw_data: dict) -> dict:
        """
        :param raw_data: the output of data provider step
        :return: (dict format) processed text based on its language
        """

        processed_data = dict()
        for lang in raw_data:
            try:
                text_processor = cls.init_replacement(lang)
                language_raw_data = raw_data[lang]
                processed_data[lang] = text_processor.__process_list_of_docs(language_raw_data)
            except NotImplementedError as err:
                logging.error(err)
        cls.release_objects_of_TextPreprocessor_for_each_lang()
        return processed_data

    @classmethod
    def release_objects_of_TextPreprocessor_for_each_lang(cls):
        """
        For training step there is no need for keeping TextPreprocessor objects in memory so we release them
        """
        cls.objects_of_TextPreprocessor_for_each_lang = dict()
        return

    @classmethod
    def init_replacement(cls, lang: str):
        """
        We must use this instead of constructor.
        Responsible for getting related model to given language.
        For the given language we have only one text_processor object that is saved in 'objects_of_models'
        class variable. When there is a request for specific language first we see, if we created before, return that,
        else we crate one, and save it, then return the new created one
        :param lang: detected language of text ;the TextProcessor object is based on this language
        :return: A TextProcessor object
        """
        logging.info("--- Getting TextProcessor object for language '%s'" % lang)
        if lang in cls.lang_models:
            if lang not in cls.objects_of_TextPreprocessor_for_each_lang:
                cls.objects_of_TextPreprocessor_for_each_lang[lang] = TextPreprocessor(lang)
            logging.info("-- Got TextProcessor object for language %s" % lang)
            return cls.objects_of_TextPreprocessor_for_each_lang[lang]
        raise NotImplementedError("--- Language '%s' is not supported" % lang)

    @classmethod
    def get_processed_data(cls, data_file_name, data_file_extension, data_version,
                           data_file_type, include_tags, exclude_tags) -> dict:
        """
        First it checks if data is read and processed before by looking for saved processed data file
        If it found the file read it and return the content
        Else read the train data by DataProvider and processed it then first save it for future and then return it

        :param data_file_name: processed or train data file name ( they have same name)
        :param data_file_extension: train data file extension
        :param data_version: the combination of version of train data and the process that we apply on it
        :param data_file_type: (DataProvider attribute)
        :param include_tags: (DataProvider attribute)
        :param exclude_tags: (DataProvider attribute)
        :return: processed data
        """
        languages_processed_data = cls.__read_processed_data(data_file_name, data_version)
        if languages_processed_data is None:
            logging.info("-- Data file was not read completely before")
            logging.info("-- Reading data from file")
            data_provider = DataProvider(include_tags, exclude_tags)
            raw_data = data_provider.get_train_data_ready_to_work(data_file_name=data_file_name,
                                                                  data_file_extension=data_file_extension,
                                                                  data_file_type=data_file_type)
            logging.info("-- Start processing data")
            languages_processed_data = TextPreprocessor.__process_raw_data_language_classified(raw_data)
            logging.info("-- Processing data is Done")
            languages = list(languages_processed_data.keys())
            cls.__write_meta_data(languages, {"include": data_provider.include_tags,
                                              "exclude": data_provider.exclude_tags,
                                              "important": data_provider.important_tags,
                                              "skip": data_provider.skip_tags}, data_version,
                                  {lang: len(languages_processed_data[lang]) for lang in languages})
            cls.__write_processed_data(version=data_version, data_file_name=data_file_name,
                                       processed_data=languages_processed_data)

        return languages_processed_data

    def __process_list_of_docs(self, list_of_docs: list) -> list:
        """

        :param list_of_docs: list of documents to preprocess that each of them is a string
        :return: list of documents that each of them is a list of preprocessed words that were is its document's text
        """
        logging.info("--- Processing texts extracted from pages")
        for index, doc in enumerate(list_of_docs):
            list_of_docs[index] = self.document_pre_process(doc)

        logging.info("--- Extracted texts are processed")
        return list_of_docs

    def document_pre_process(self, document: str):
        """
        :description: based on document's text's language there is a model in spacy, by that we preprocess the text
        :param document: is one string (concatenation of text of requested tags done in data_providing step)
        :return: a list of tokens of document's text
        """
        nlp = self.model
        post_process_data = list()
        if nlp is not None:
            data = nlp(document)
            if self.lang == "en":
                for token in data:
                    if self.__is_token_valid(token):
                        post_process_data.append(token.lemma_)
            else:
                for token in data:
                    if self.__is_token_valid(token):
                        post_process_data.append(token.lemma_)
        # return list(data.ents)
        return post_process_data
