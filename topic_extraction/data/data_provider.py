import json
import logging

from bs4 import BeautifulSoup

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from topic_extraction.advisor import Advisor


class DataProvider:
    data_file_type = ["CommonCrawl", "Json", "SemiJson"]
    important_tags = ["h1", "p"]
    skip_tags = ["head", "style", "script", "noscript", "title", "link", "meta"]

    def __init__(self,
                 include_tags: list = None,
                 exclude_tags: list = None
                 ):
        """

        :param data_folder_path: os path that data file is there (there is no '/' at the end of the path)
        :param file_extension: data file extension
        :param data_file_type:

        :param include_tags: html tags that their text are wanted (pass this param make exclude_tags params ignored)
        :param exclude_tags: html tags that their text are unwanted
        """
        self.include_tags = include_tags
        self.exclude_tags = exclude_tags

        return

    @staticmethod
    def _read_json_file(data_file_path: str) -> list:
        """
        :param data_file_path: os path to data file
        :return: list of web pages in html
        """
        file_data = list()
        with open(data_file_path) as json_file:
            for line in json_file:
                data_file = json.loads(line)
                file_data.append(data_file)
        return file_data

    @staticmethod
    def _read_semi_json_file(data_file_path: str) -> list:
        """
        :param data_file_path: os path to data file
        :return: list of web pages in html
        """
        file = open(data_file_path, "r")
        count = 0
        dataset = list()
        while True:
            line = file.readline()
            if not line:
                break
            count += 1
            splits = [l for l in line.split('|') if l]
            if len(splits) == 3:
                try:
                    # check the storage version protocol
                    if splits[0] == '1':
                        data = json.loads(json.loads(splits[2])['fd_data'])
                        dataset.append(data)
                    else:
                        raise Warning(f"Undefined storage version `{splits[0]}`")
                except:
                    logging.warning(" ---- There is a problem in reading data file line {} ----".format(count))
        file.close()
        return dataset

    @staticmethod
    def _read_common_crawl_file(data_file_path: str):
        """
        :param data_file_path: os path to data file
        :return: list of web pages in html
        """
        logging.info("---- Reading data file %s " % data_file_path)
        # line_number = 1
        error_line_count = 0
        file = open(data_file_path, "r")
        data_set = list()
        page = ""
        while True:
            try:
                line = file.readline()
                if not line:
                    data_set.append(page)
                    break
                line = line.strip()
                if line.startswith("Content-Type: text/html"):
                    data_set.append(page)
                    page = ""
                page += line
            except Exception as err:
                error_line_count += 1
        file.close()
        logging.info("------ Total error lines are %d" % error_line_count)
        logging.info("---- Reading data file %s is Finished " % data_file_path)
        return data_set

    @staticmethod
    def _convert_list_of_string_to_one_string(list_of_str: list):
        text = ""
        for element in list_of_str:
            text += element + " "
        return text

    @classmethod
    def _get_raw_data_from_path_file(cls, data_file_type: str, data_file_path: str):
        """
        :param data_file_type: one of the ["CommonCrawl", "Json", "SemiJson"]; base on this,file reading is happening
        :param data_file_path: the exact data file path
                        (path-to-data-folder/(data-file-name)-(train/test).data-extension)
        :return: list of web pages ( html code and content)
        """
        if data_file_type == cls.data_file_type[0]:
            return DataProvider._read_common_crawl_file(data_file_path)
        if data_file_type == cls.data_file_type[1]:
            return DataProvider._read_json_file(data_file_path)
        if data_file_type == cls.data_file_type[2]:
            return DataProvider._read_semi_json_file(data_file_path)

    def get_train_data_ready_to_work(self, data_file_name: str,
                                     data_file_extension: str, data_file_type: str, ) -> dict:
        """
        :param data_file_name: data file name -> {data_file_name}-train.{data_file_extension}
        :param data_file_extension: data file extension
        :param data_file_type: only three types are supported["CommonCrawl", "Json", "SemiJson"]
                        (path-to-data-folder/(data-file-name)-(train/test).data-extension)
        :return: dic of languages as keys, and their values are list of that language's documents;
        and each document is one string that holds all web page's text
        """
        data_file_path = Advisor.get_data_folders_file_path(data_file_name, data_file_extension)
        ready_to_train_data = dict()
        logging.info("--- Get data file content")

        text_data = self._get_raw_data_from_path_file(data_file_type,
                                                      data_file_path)
        logging.info("--- Getting pages's code and language")
        for index, page in enumerate(text_data):
            page = BeautifulSoup(page, features="html.parser")
            text, lang = self._get_text_data_from_page(page)
            if lang not in ready_to_train_data:
                ready_to_train_data[lang] = list()
            ready_to_train_data[lang].append(text)
        logging.info("--- Got pages's text and language")
        logging.info("-- Data is ready for further processes")
        return ready_to_train_data

    def _get_text_data_from_page(self, page: BeautifulSoup):
        """
        Based on requested tags it will extract the data from page
        :param page: page in BeautifulSoup type
        :return: the extracted text and its language;
        """
        lang = "None"
        if self.include_tags is not None:
            text = self._get_included_tag_data(page)
        elif self.exclude_tags is not None:
            text = self._get_not_excluded_tag_data(page)
        else:
            text = self._get_important_tags_data(page)
        try:
            lang = detect(text)
        except LangDetectException as err:
            pass
        return text, lang

    def _get_included_tag_data(self, web_page: BeautifulSoup) -> str:
        """
        :param web_page: the html code
        :return: list of texts in included tags in the page

        there is a miss in here if we have a text in a child tag in html and both child and parent tags are exploring
        the same text is there for each of them
        e.g. <p> <a> there is a example </a> </p>
        there is two text in return variable : "there is a example" twice, one for <p> and another for <a>
        """
        final_data = ""
        for tag in self.include_tags:
            group_tag = web_page.find_all(tag)
            for tag_data in group_tag:
                final_data += tag_data.text + " "
        return final_data

    def _get_not_excluded_tag_data(self, web_page: BeautifulSoup) -> str:
        """
        :param web_page: the html code
        :return: list of texts that are not in excluded tags in the page
        there is a miss in here if we have a text in a child tag in html and both child and parent tags are exploring
        the same text is there for each of them
        e.g. <p> <a> there is a example </a> </p>
        there is two text in return variable : "there is a example" twice, one for <p> and another for <a>
        """
        final_data = ""
        for tag in web_page.find_all():
            if tag.name not in self.exclude_tags and tag.name not in self.skip_tags:
                final_data += tag.text + " "
        return final_data

    def _get_important_tags_data(self, web_page: BeautifulSoup) -> str:
        """
        :param web_page: the html code in BeautifulSoup object
        :return: list of texts that are in important tags in the page
        if there is include_tags and exclude_tags are none we use important_tags
        """
        final_data = ""
        for tag in self.important_tags:
            group_tag = web_page.find_all(tag)
            for tag_data in group_tag:
                final_data += tag_data.text + " "
        return final_data
