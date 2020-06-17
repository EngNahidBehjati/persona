import logging
from random import random

import numpy

from matplotlib import pyplot
from wordcloud import WordCloud
from collections import Counter
from pandas import DataFrame, options, Series, concat
from numpy import mean, median, std, quantile

from topic_extraction.advisor import Advisor


class ModelView:
    number_of_decimal_digits: int = 5

    def __init__(self,
                 lang: str,
                 data_version: int, dictionary_version: float, model_version: str, param_name: str, param_version: int,
                 number_of_decimal_digits: int = 5,
                 max_colwidth: int = 100):

        self.topic_version_path = Advisor.get_param_version_folder_path(lang, data_version, dictionary_version,
                                                                        model_version, param_name, param_version)
        self.number_of_decimal_digits = number_of_decimal_digits
        options.display.max_colwidth = max_colwidth
        return

    @classmethod
    def _get_nth_dominate_topic_of_docs(cls, nth_topic: int, model, corpus):
        # Init output as panda's dataFrame
        doc_dominate_topics_df = DataFrame()

        # Get main topic in each document
        for i, row in enumerate(model[corpus]):
            # To get the dominated topic sort topics in descend way and pick the first one
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Percentage  Contribution and Keywords for each document
            if nth_topic < len(row):
                topic_num, prop_topic = row[nth_topic]
            else:
                topic_num, prop_topic = (-1, 0.0)

            # Get words that representing the chosen topic
            words_of_the_topic = model.show_topic(topic_num)
            # Concat the words to get one string
            topic_keywords = ", ".join([word for word, prop in words_of_the_topic])
            # Make a row for output dataFrame of [topic_num, prop_topic, topic_keywords]
            doc_dominate_topics_df = doc_dominate_topics_df.append(
                Series(
                    [int(topic_num),
                     round(prop_topic, cls.number_of_decimal_digits),
                     topic_keywords]),
                ignore_index=True)
            # Create a row for column's names
        doc_dominate_topics_df = doc_dominate_topics_df.reset_index()
        doc_dominate_topics_df.columns = ['Document_No', 'Dominant_Topic',
                                          'Percentage_of_Contribution', 'Topic_Keywords']

        return doc_dominate_topics_df

    @classmethod
    def _add_docs_text_to_doc_nth_dominate_topic_df(cls, doc_nth_dominate_topics_df: DataFrame, processed_data: list):
        contents = Series(processed_data)
        doc_dominate_topics_p_text_df = concat([doc_nth_dominate_topics_df, contents], axis=1)
        doc_dominate_topics_p_text_df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib',
                                                 'Keywords', 'Text']
        return doc_dominate_topics_p_text_df

    @classmethod
    def _get_top_topics_of_docs(cls, model, corpus, threshold: float) -> DataFrame:
        n_top_topics_of_docs = DataFrame(columns=["Document_No", "Topic", "Topic_Perc_Contrib"])
        documents = model[corpus]
        for document_id, row in enumerate(documents):
            # To get the dominated topic sort topics in descend way and pick the first one
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            for topic, portion in row:
                if portion >= threshold:
                    n_top_topics_of_docs = n_top_topics_of_docs.append(DataFrame({"Document_No": [document_id],
                                                                                  "Topic": [topic],
                                                                                  "Topic_Perc_Contrib": [portion]}),
                                                                       ignore_index=True)
        return n_top_topics_of_docs

    @classmethod
    def _get_topic_words_weight_and_words_counter(cls, model, processed_data):
        data_flat = [w for w_list in processed_data for w in w_list]
        counter = Counter(data_flat)

        topics = model.show_topics(formatted=False, num_topics=-1, num_words=30)
        topic_word = []
        for topic_id, word_matrix in topics:
            for word, weight in word_matrix:
                topic_word.append([word, int(topic_id), weight, counter[word]])

        topic_word = DataFrame(topic_word, columns=['word', 'topic_id', 'importance', 'word_count'])
        return topic_word

    def _plot_number_of_docs_in_each_dominate_topic(self, number_of_docs_in_each_dominate_topic: list,
                                                    model_type, nth_topic):
        file_path = Advisor.get_visualization_file_path_from_topic_version(self.topic_version_path, model_type,
                                                                           "Number_of_docs_in_each_"
                                                                           "'%d'th_dominate_topic" %
                                                                           nth_topic, "png")
        divide_no = 10
        number_of_docs_in_each_dominate_topic = DataFrame(number_of_docs_in_each_dominate_topic,
                                                          columns=["Topic_id", "No_of_Docs"])
        number_of_docs_in_each_dominate_topic = number_of_docs_in_each_dominate_topic.sort_values("No_of_Docs")

        topic_no = len(number_of_docs_in_each_dominate_topic)
        plot_no = int(topic_no / divide_no) if topic_no % divide_no == 0 else int(topic_no / divide_no) + 1
        fig, axes = pyplot.subplots(plot_no, 1)
        fig.set_size_inches(divide_no * 2.0, plot_no * 8.0)
        if plot_no > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        start = 0
        for i in range(plot_no):
            ax = axes[i]
            plot_data = number_of_docs_in_each_dominate_topic[start:start + divide_no]
            ax.bar(x='Topic_id', height="No_of_Docs", data=plot_data,
                   width=0.5, alpha=0.3)
            dominate_topic_range = plot_data.Topic_id.to_list()
            ax.set_ylabel('Document Count Percentage')
            ax.set_xlabel('Topics')
            ax.set_xticks(dominate_topic_range)
            ax.set_xticklabels(['Topic %d' % topic for topic in dominate_topic_range], rotation=30,
                               horizontalalignment='right', fontsize=8)
            self._set_bar_plot_text(ax)
            start += divide_no
        fig.savefig(file_path)
        fig.suptitle('Number of Documents in Percentage for each Topic', fontsize=14)
        pyplot.close(fig)
        return

    def _plot_number_of_docs_in_each_top_topic(self, number_of_docs_in_each_top_topic: list, model_type):
        divide_no = 10
        file_path = Advisor.get_visualization_file_path_from_topic_version(self.topic_version_path, model_type,
                                                                           "Number_of_docs_in_each_top_topic", "png")
        number_of_docs_in_each_top_topic = DataFrame(number_of_docs_in_each_top_topic,
                                                     columns=["Topic_id", "No_of_Docs"])
        number_of_docs_in_each_top_topic = number_of_docs_in_each_top_topic.sort_values("No_of_Docs")
        topic_no = len(number_of_docs_in_each_top_topic)
        plot_no = int(topic_no / divide_no) if topic_no % divide_no == 0 else int(topic_no / divide_no) + 1
        fig, axes = pyplot.subplots(plot_no, 1)
        fig.set_size_inches(divide_no * 2.0, plot_no * 8.0)
        if plot_no > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        start = 0
        for i in range(plot_no):
            ax = axes[i]
            plot_data = number_of_docs_in_each_top_topic[start: start + divide_no]
            ax.bar(x='Topic_id', height="No_of_Docs", data=plot_data,
                   width=0.5, alpha=0.3, label='Document Count')
            top_topic_range = plot_data.Topic_id.to_list()
            ax.set_ylabel('Document Count Percentage')
            ax.set_xlabel('Topics')
            ax.set_xticks(top_topic_range)
            ax.set_xticklabels(['Topic %d' % topic for topic in top_topic_range], rotation=30,
                               horizontalalignment='right', fontsize=8)
            self._set_bar_plot_text(ax)
            start += divide_no
        fig.savefig(file_path)
        fig.suptitle('Number of Documents in Percentage for each Topic', fontsize=14)
        pyplot.close(fig)

    @classmethod
    def _get_random_color(cls):
        r = random()
        b = random()
        g = random()
        return r, g, b

    def _contribution_of_each_dominate_topic_in_docs(self, contribution_of_dominate_topic_in_docs: dict,
                                                     doc_no, model_type: str, nth_topic: int):
        document_id_range = range(0, doc_no + 1, int(doc_no / 10)) if doc_no > 10 else range(0, doc_no + 1, 1)
        for topic_id in contribution_of_dominate_topic_in_docs:
            file_path = Advisor.get_topic_folders_file_path_from_topic_version(self.topic_version_path, model_type,
                                                                               int(topic_id),
                                                                               "Contribution_of_"
                                                                               "'%d'th_dominate_topic_in_docs" %
                                                                               nth_topic, "png")
            fig, ax = pyplot.subplots()
            ax.set_title("Contribution Of Topic '%d' ")
            data = DataFrame(contribution_of_dominate_topic_in_docs[topic_id],
                             columns=["Document_No", "Percentage_of_Contribution"])
            ax.plot('Document_No', 'Percentage_of_Contribution', data=data, marker='o',
                    color=self._get_random_color(), dashes=[6, 2])
            ax.set_ylabel("Contribution")
            ax.set_xlabel("Documents")
            ax.set_title("Contribution of Topic %d Over Documents" % topic_id)
            ax.set_xticks(document_id_range)
            ax.set_xticklabels(['Document %d' % document for document in document_id_range], rotation=45,
                               horizontalalignment='right', fontsize=8)
            text = list(data.Percentage_of_Contribution)
            d_mean = round(mean(text))
            d_median = round(median(text))
            d_std = round(std(text))
            d_one_percent = round(quantile(text, q=0.01))
            d_ninety_nine_percent = round(quantile(text, q=0.99))
            d_text = "Mean : {}\nMedian : {}\nStdev: {}\n1%ile : {}\n99%ile : {}".format(d_mean, d_median, d_std,
                                                                                         d_one_percent,
                                                                                         d_ninety_nine_percent)
            ax.text(0.9, 0.98, d_text, transform=ax.transAxes, bbox=dict(fc="none"), color='purple')
            ax.grid(True)

            fig.savefig(file_path)
            pyplot.close(fig)
        return

    def _contribution_of_dominate_topics_in_docs(self, contribution_of_dominate_topic_in_docs: dict,
                                                 model_type: str, nth_topic: int, doc_no: int):
        file_path = Advisor.get_visualization_file_path_from_topic_version(self.topic_version_path, model_type,
                                                                           "Contribution_of_"
                                                                           "%d_dominate_topics_in_docs" %
                                                                           nth_topic, "png")
        plot_no = len(contribution_of_dominate_topic_in_docs)
        document_id_range = range(0, doc_no + 1, int(doc_no / 10)) if doc_no > 10 else range(0, doc_no + 1, 1)
        fig, axes = pyplot.subplots(plot_no, 1, figsize=(2 * 2.0, (plot_no + 1) * 5.0))
        if plot_no > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        ax_no = 0
        fig.suptitle("Contribution Of '%dth' Dominate Topics In Docs" % nth_topic)
        docs = set()
        for topic_id in contribution_of_dominate_topic_in_docs:
            ax = axes[ax_no]
            ax_no += 1
            label = "%d" % topic_id
            data = DataFrame(contribution_of_dominate_topic_in_docs[topic_id],
                             columns=["Document_No", "Percentage_of_Contribution"])
            ax.plot('Document_No', 'Percentage_of_Contribution', data=data, marker='o',
                    color=self._get_random_color(), label=label, dashes=[6, 2])
            docs.update(data["Document_No"].tolist())

            ax.legend()
            ax.set_ylabel("Contribution of Topic")
            ax.set_title("Topic %d" % topic_id)
            ax.set_xticks(document_id_range)
            ax.set_xticklabels(['Document %d' % document for document in document_id_range], rotation=30,
                               horizontalalignment='right', fontsize=8)
            ax.grid(True)

        fig.savefig(file_path)
        pyplot.close(fig)
        return

    def _contribution_of_top_topics_in_docs(self, contribution_of_top_topic_in_docs: dict, model_type: str,
                                            doc_no: int):
        file_path = Advisor.get_visualization_file_path_from_topic_version(self.topic_version_path, model_type,
                                                                           "Contribution_of_top_topics_in_docs", "png")
        plot_no = len(contribution_of_top_topic_in_docs)
        document_id_range = range(0, doc_no + 1, int(doc_no / 10)) if doc_no > 10 else range(0, doc_no + 1, 1)
        fig, axes = pyplot.subplots(plot_no, 1, figsize=(2 * 2.0, (plot_no + 1) * 5.0))
        if plot_no > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        ax_no = 0
        fig.suptitle("Contribution Of Top Topics In Docs")
        for topic_id in contribution_of_top_topic_in_docs:
            ax = axes[ax_no]
            ax_no += 1
            label = "%d" % topic_id
            data = DataFrame(contribution_of_top_topic_in_docs[topic_id],
                             columns=["Document_No", "Percentage_of_Contribution"])
            ax.plot('Document_No', 'Percentage_of_Contribution', data=data, marker='o',
                    color=self._get_random_color(), label=label, dashes=[6, 2])
            ax.legend()
            ax.set_xlabel("Documents")
            ax.set_title("Topic %d" % topic_id)
            ax.set_xticks(document_id_range)
            ax.set_xticklabels(['Document %d' % document for document in document_id_range], rotation=30,
                               horizontalalignment='right', fontsize=8)
            ax.grid(True)

        fig.savefig(file_path)
        pyplot.close(fig)

    @classmethod
    def _set_bar_plot_text(cls, ax):
        for nth_topic in ax.patches:
            # get_x pulls left or right; get_height pushes up or down
            ax.text(nth_topic.get_x() - .03, nth_topic.get_height() + 0.0001,
                    str(round(nth_topic.get_height(), cls.number_of_decimal_digits)), fontsize=10,
                    color='dimgrey')
        return

    @classmethod
    def _get_words_tfidf(cls, topics, processed_list):
        flatten_topic_words = [word for _, words in topics for word, _ in words]
        docs_length = dict()
        words_info = dict()
        total_number_of_docs = len(processed_list)
        for word in flatten_topic_words:
            for doc_id, doc in enumerate(processed_list):
                docs_length[doc_id] = len(doc)
                for doc_word in doc:
                    if (doc_word == word) is False:
                        continue
                    if word not in words_info:
                        words_info[word] = dict()
                    if doc_id not in words_info[word]:
                        words_info[word][doc_id] = 0

                    words_info[word][doc_id] += 1
        for word in words_info:
            for doc_id in words_info[word]:
                words_info[word][doc_id] = ((words_info[word][doc_id] / len(processed_list[doc_id])) *
                                            (numpy.log(total_number_of_docs / len(words_info[word]))))
        return words_info

    def topic_words_and_its_joint(self, model, model_type, processed_data):
        topics = model.show_topics(formatted=False, num_topics=-1, num_words=30)
        all_topics_words = self._get_topic_words_weight_and_words_counter(model, processed_data)
        for topic in topics:
            topic_and_its_words_in_whole_data = dict()
            for word, weight in topic[1]:
                group = all_topics_words.loc[all_topics_words.word == word].groupby("topic_id")
                for topic_id, out_line in group:
                    topic_id = int(topic_id)
                    if topic_and_its_words_in_whole_data.get(topic_id, None) is None:
                        topic_and_its_words_in_whole_data[topic_id] = list()
                    topic_and_its_words_in_whole_data[topic_id].append([word, out_line.importance.values[0],
                                                                        out_line.word_count.values[0]])
            sub_plot_number = len(topic_and_its_words_in_whole_data)

            fig, axes = pyplot.subplots(sub_plot_number + 1, 1, figsize=(2 * 8.0, (sub_plot_number + 1) * 5.0))
            if len(topic_and_its_words_in_whole_data) > 1:
                axes = axes.flatten()
            word_count_ax = axes[0]
            topic_word_count = DataFrame(topic_and_its_words_in_whole_data[int(topic[0])],
                                         columns=['word', 'word_importance', 'word_count'])
            word_count_ax.bar(x='word', height="word_count", data=topic_word_count,
                              width=0.5, alpha=0.3)
            self._set_bar_plot_text(word_count_ax)
            word_count_ax.set_ylabel('Word Count')
            word_count_ax.set_title('Topic: %d Word Count' % topic[0], fontsize=12)
            word_count_ax.set_xticklabels(topic_word_count.word, rotation=30, horizontalalignment='right', fontsize=8)

            ax_n = 1
            for i in topic_and_its_words_in_whole_data:
                tdf = DataFrame(topic_and_its_words_in_whole_data[i], columns=['word', 'word_importance', 'word_count'])
                topic_word_ax = axes[ax_n]
                topic_word_ax.bar(x='word', height="word_importance", data=tdf, width=0.2)
                topic_word_ax.set_title('Topic: %d Word Weight' % i, fontsize=12)
                topic_word_ax.set_xticklabels(tdf.word, rotation=30, horizontalalignment='right', fontsize=8)
                self._set_bar_plot_text(topic_word_ax)

                ax_n += 1
            # fig.tight_layout(w_pad=2)
            fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=18, y=1.05)

            file_name = "WordCountsOfTopicKeywords"
            fig_file_name = Advisor.get_topic_folders_file_path_from_topic_version(self.topic_version_path, model_type,
                                                                                   topic[0],
                                                                                   file_name, "png")
            fig.savefig(fig_file_name)
            pyplot.close(fig=fig)
        return

    def word_cloud_of_top_n_words_in_each_topic(self, top_n: int, lda_model, model_type):
        cloud = WordCloud(background_color='white',
                          max_words=top_n,
                          max_font_size=5 * top_n,
                          prefer_horizontal=1.0,
                          font_step=5)

        topics = lda_model.show_topics(formatted=False, num_topics=-1, num_words=30)
        for topic in topics:
            topic_words = dict(topic[1])
            try:
                data_file_path = Advisor.get_topic_folders_file_path_from_topic_version(self.topic_version_path,
                                                                                        model_type, topic[0],
                                                                                        "WordCloud", 'png')
                cloud.generate_from_frequencies(topic_words, max_font_size=10 * top_n)
                cloud.to_file(data_file_path.format(topic[0]))
            except OSError as err:
                logging.error(err)
        return

    def distribution_of_nth_dominate_topics_over_documents(self, deeps_of_dominate_topic_to_go,
                                                           model_type, model, corpus):
        documents_count = len(corpus)

        for nth_topic in range(0, deeps_of_dominate_topic_to_go):
            doc_dominate_nth_topic = self._get_nth_dominate_topic_of_docs(nth_topic, model, corpus)
            docs_grouped_by_dominate_topic = doc_dominate_nth_topic.groupby('Dominant_Topic')

            number_of_docs_in_each_dominate_topic = list()
            contribution_of_dominate_topic_in_docs = dict()
            for dominate_topic_id, dt_group in docs_grouped_by_dominate_topic:
                if contribution_of_dominate_topic_in_docs.get(dominate_topic_id, None) is None:
                    contribution_of_dominate_topic_in_docs[dominate_topic_id] = list()

                number_of_docs_in_each_dominate_topic.append([dominate_topic_id, len(dt_group) / documents_count])
                for doc in dt_group.iterrows():
                    doc = doc[1]
                    contribution_of_dominate_topic_in_docs[dominate_topic_id].append([doc.Document_No,
                                                                                      doc.Percentage_of_Contribution])
            self._plot_number_of_docs_in_each_dominate_topic(number_of_docs_in_each_dominate_topic,
                                                             model_type, nth_topic)

            self._contribution_of_dominate_topics_in_docs(contribution_of_dominate_topic_in_docs,
                                                          model_type, nth_topic, len(corpus))

            self._contribution_of_each_dominate_topic_in_docs(contribution_of_dominate_topic_in_docs, len(corpus),
                                                              model_type, nth_topic)
        return

    def distribution_of_top_topics_over_documents(self, model, corpus, model_type, threshold: float = 0.5):
        documents_count = len(corpus)

        documents_top_topics = self._get_top_topics_of_docs(model, corpus, threshold)
        documents_grouped_by_top_topic_id = documents_top_topics.groupby('Topic')

        number_of_docs_in_each_top_topic_id = list()
        contribution_of_top_topic_in_docs = dict()
        for top_topic_id, dt_group in documents_grouped_by_top_topic_id:
            if contribution_of_top_topic_in_docs.get(top_topic_id, None) is None:
                contribution_of_top_topic_in_docs[top_topic_id] = list()

            number_of_docs_in_each_top_topic_id.append([top_topic_id, len(dt_group) / documents_count])
            for doc in dt_group.iterrows():
                doc = doc[1]
                contribution_of_top_topic_in_docs[top_topic_id].append([doc.Document_No,
                                                                        doc.Topic_Perc_Contrib])
        self._plot_number_of_docs_in_each_top_topic(number_of_docs_in_each_top_topic_id, model_type)
        self._contribution_of_top_topics_in_docs(contribution_of_top_topic_in_docs, model_type, len(corpus))
        return

    def get_topics_words_and_their_tfidf_of_them_over_docs(self, model, processed_data, model_type):
        topics = model.show_topics(formatted=False, num_topics=-1, num_words=30)
        words_tfidf = self._get_words_tfidf(topics, processed_data)
        max_doc = len(processed_data)
        document_id_range = range(0, max_doc + 1, int(max_doc / 10)) if max_doc > 10 else range(0, max_doc + 1)
        for topic_id, topic in topics:
            file_path = Advisor.get_topic_folders_file_path_from_topic_version(self.topic_version_path, model_type,
                                                                               topic_id,
                                                                               "Words TFiDF", "png")
            words_no = len(topic)
            # fig, axes = pyplot.subplots(words_no, 1, figsize=(2 * 2.2, min([(len(processed_data) + 1) * 5.0, 2 ** 16])))
            fig, axes = pyplot.subplots(words_no, 1)
            fig.suptitle("TFiDF of Topic '%d' over Docs" % topic_id)
            if words_no > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            ax_no = 0
            for word, _ in topic:
                ax = axes[ax_no]
                ax_no += 1
                data = list()
                word_data = words_tfidf[word]
                for doc_id in word_data:
                    data.append([word, doc_id, word_data[doc_id]])
                data = DataFrame(data, columns=["Word", "Document_No", "TFiDF"])
                ax.plot('Document_No', 'TFiDF', data=data, marker='o',
                        color=self._get_random_color(), label='Word : %s' % word, dashes=[6, 2])
                ax.legend()
                ax.set_ylabel("TFiDF of Words")
                ax.set_xlabel("Documents")
                ax.set_title("Word %s" % word)
                ax.set_xticks(document_id_range)
                ax.set_xticklabels(['Document %d' % document for document in document_id_range], rotation=30,
                                   horizontalalignment='right', fontsize=8)
                ax.grid(True)
            fig.savefig(file_path)
            pyplot.close(fig)
        return

    def get_model_visualizations(self, model_type: str, model, corpus, processed_data, top_n=20):
        logging.info("<<%s>> Topics Words and Their Joint" % model_type)
        self.topic_words_and_its_joint(model, model_type, processed_data)
        logging.info("<<%s>> Word Cloud Of Top '%d' Words In Each Topic" % (model_type, top_n))
        self.word_cloud_of_top_n_words_in_each_topic(top_n, model, model_type)
        logging.info("<<%s>> topics_words_and_their_tfidf_of_them_over_docs" % model_type)
        self.get_topics_words_and_their_tfidf_of_them_over_docs(model, processed_data, model_type)
        logging.info("<<%s>> distribution_of_top_topics_over_documents" % model_type)
        self.distribution_of_top_topics_over_documents(model, corpus, model_type)
        logging.info("<<%s>> distribution_of_nth_dominate_topics_over_documents" % model_type)
        self.distribution_of_nth_dominate_topics_over_documents(3, model_type, model, corpus)
