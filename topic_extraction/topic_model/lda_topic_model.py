import logging
import operator
from os import path

from gensim.models import CoherenceModel, LdaModel

from topic_extraction.advisor import Advisor
from topic_extraction.visualization.model_view import ModelView
from topic_extraction.topic_model.topic_model import TopicModel


class LdaTopicModel(TopicModel):
    def __init__(self,
                 lang: str,
                 number_of_topics: int,
                 data_version: int, dictionary_version: float, model_version: str,
                 param_name: str, param_version: int,
                 language_processed_data: list,
                 no_below: int,
                 no_above: float,
                 n_most_frequent: int,
                 chunk_size: int,
                 alpha: float,
                 beta: float,
                 iterations: int,
                 passes: int,
                 eval_every: int,
                 model_view: bool):
        """
        TopicModel's Parameters:
        :param data_folder_train_version_related_path: For its parent
        :param number_of_topics: For its parent
        :param language_processed_data: For its parent
        :param no_below: For its parent
        :param no_above: For its parent


        LdaTopicModel's Parameters:

        :param chunk_size: Number of documents to be used in each training chunk.
        controls how many documents are processed at a time in the training algorithm.
        Increasing chunk_size will speed up training, at least as long as the chunk of documents easily fit into memory.

        :param alpha: Can be set to an 1D array of length equal to the number of expected topics that expresses
        our a-priori belief for the each topics’ probability. Alternatively default prior selecting strategies
        can be employed by supplying a string:

                    ’asymmetric’: Uses a fixed normalized asymmetric prior of 1.0 / topicno.
                    ’auto’: Learns an asymmetric prior from the corpus (not available if distributed==True).

        :param beta: Word-Topic Density,  with a high beta, topics are made up of most of the words in the corpus,
        and with a low beta they consist of few words.

        :param iterations: is somewhat technical, but essentially
         it controls how often we repeat a particular loop over each document.
         It is important to set the number of “passes” and “iterations” high enough.

        :param passes: Number of passes through the corpus during training.
        controls how often we train the model on the entire corpus.
        Another word for passes might be “epochs”.
         It is important to set the number of “passes” and “iterations” high enough.

        :param eval_every:

        other parameters:
        - minimum_probability (float, optional) – Topics with a probability lower than this threshold will be filtered out.
        """
        super().__init__(lang,
                         number_of_topics,
                         data_version, dictionary_version,
                         language_processed_data,
                         no_below, no_above, n_most_frequent)
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
        cls.model_type = "lda"

    def set_model(self, lang: str, data_version: int, dictionary_version: float, model_version: str, param_name: str, param_version: int,
                  model_file_path: str, language_processed_data: list):
        """
        'alpha'& 'eta' are hyperparameters that affect sparsity of the topics.
        According to the Gensim docs, both defaults to 1.0/num_topics prior.
        :return:
        """
        # Make a index to word dictionary.
        logging.info("---- Creating LDA model")
        temp = self.essentials.dictionary[0]
        "for multicore model optimal workers=3, one less than the number of cores"
        model = LdaModel(
            # workers=self.workers,
            corpus=self.essentials.corpus,
            id2word=self.essentials.dictionary.id2token,
            chunksize=self.chunk_size,
            alpha=self.alpha,
            eta=self.beta,
            iterations=self.iterations,
            num_topics=self.number_of_topics,
            passes=self.passes,
            eval_every=self.eval_every
        )
        model.save(model_file_path)
        self.model = model
        logging.info("---- LDA model is created")

        metrics = self.get_model_evaluation_metrics(language_processed_data)
        parameters = self.get_model_parameters()
        self.write_model_evaluation_metrics(lang, data_version, dictionary_version, model_version,param_name, param_version, metrics, parameters)
        return

    def get_model(self, lang, data_version: int, dictionary_version: float, model_version: str, param_name: str, param_version: int,
                  language_processed_data: list, model_view: bool):
        logging.info("--- Getting LDA model")
        if self.model is None:
            model_file_path = Advisor.get_model_type_folders_file_path(lang, data_version,
                                                                       dictionary_version, model_version, param_name, param_version,
                                                                       self.model_type, "LDA-model")
            if path.exists(model_file_path):
                logging.info("---- LDA model was crated before")
                self.model = LdaModel.load(model_file_path)
            else:
                self.set_model(lang, data_version, dictionary_version, model_version, param_name, param_version,
                               model_file_path, language_processed_data)
        logging.info("--- LDA model captured")
        if model_view:
            self.visualization.get_model_visualizations(self.model_type, self.model, self.essentials.corpus,
                                                        language_processed_data)
        return self.model

    def get_documents_topic(self, document: list):
        document = self.essentials.dictionary.doc2bow(document)
        return self.model.get_document_topics(document)

    def get_model_evaluation_metrics(self, language_processed_data: list) -> dict:
        """
        Evaluate LDA topics
            Automatic & Manual
                • Automatic
                    • Perplexity: lower is better
                     It captures how surprised a model is of new data it has not seen before,
                     and is measured as the normalized log-likelihood of a held-out test set.
                     you can think of the perplexity metric as measuring
                     how probable some new unseen data is given the model that was learned earlier.
                     That is to say, how well does the model represent or reproduce the statistics of the held-out data.
                     However, recent studies have shown that predictive likelihood (or equivalently, perplexity) and
                     human judgment are often not correlated, and even sometimes slightly anti-correlated.

                    • Coherence: higher is better
                     The concept of topic coherence combines a number of measures into a
                     framework to evaluate the coherence between topics inferred by a model.
                     Topic Coherence measures score a single topic by measuring the degree of semantic similarity
                     between high scoring words in the topic. These measurements help distinguish between topics that
                     are semantically interpretable topics and topics that are artifacts of statistical inference.

                     C_V typically 0 < x < 1 and uMass -14 < x < 14.
                • Manual:
                    • Word intrusion:
                        Pick a topic and the top n words in	tha	topic. Next, pick another word in the bottom list which
                        is a top word in another topic.Present these n+1 words to a person and ask them to pick the odd
                        one.
                    • Topic	intrusion:
                        Pick the top n most probable topics. Next, from the low probability topics, randomly pick one.
                        Show subjects a small snippet of the document and ask them to pick the intruder.
        :return:
        """
        topics = self.model.show_topics(formatted=False, num_topics=-1, num_words=30)
        u_mass_top_topics = self.model.top_topics(self.essentials.corpus, topn=self.number_of_topics, coherence="u_mass")
        c_v_top_topics = self.model.top_topics(corpus=self.essentials.corpus, texts=language_processed_data,
                                               dictionary=self.essentials.dictionary,
                                               topn=self.number_of_topics, coherence="c_v")
        avg_topic_coherence_u_mass = sum([t[1] for t in u_mass_top_topics]) / self.number_of_topics
        avg_topic_coherence_c_v = sum([t[1] for t in c_v_top_topics]) / self.number_of_topics

        lda_topics = [[word for word, prob in topic] for topic_id, topic in topics]
        # Compute Perplexity
        perplexity = self.model.log_perplexity(self.essentials.corpus)

        # Compute Coherence Scores
        coherence = CoherenceModel(topics=lda_topics, corpus=self.essentials.corpus,
                                   dictionary=self.essentials.dictionary,
                                   coherence='c_v', texts=language_processed_data)
        c_v = coherence.get_coherence()
        coherence = CoherenceModel(topics=lda_topics, corpus=self.essentials.corpus,
                                   dictionary=self.essentials.dictionary,
                                   coherence='u_mass', texts=language_processed_data)
        u_mass = coherence.get_coherence()

        return {"perplexity": perplexity,
                "c_v_coherence_lda": c_v,
                "u_mass_coherence_lda": u_mass,
                "avg_topic_coherence_u_mass": avg_topic_coherence_u_mass,
                "avg_topic_coherence_c_v": avg_topic_coherence_c_v}

    def save_this_to_use_properly(self, language_processed_data: str):
        """
            Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
            quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
            evaluating the topic model until this threshold is crossed.

            Returns:
            -------
            lm: Final evaluated topic model
            top_topics: ranked topics in decreasing order. List of tuples
            """
        top_topics = [(0, 0)]
        lm = None
        while top_topics[0][1] < 0.97:
            lm = LdaModel(corpus=self.essentials.corpus, id2word=self.essentials.dictionary.id2token)
            coherence_values = {}
            for n, topic in lm.show_topics(num_topics=-1, formatted=False):
                topic = [word for word, _ in topic]
                cm = CoherenceModel(topics=[topic], texts=language_processed_data,
                                    dictionary=self.essentials.dictionary, window_size=10)
                coherence_values[n] = cm.get_coherence()
            top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
        return lm, top_topics

    def get_model_parameters(self) -> dict:
        return {"common": self.get_common_parameters(),
                "lda": {"chunk_size": self.chunk_size,
                        "alpha": self.alpha,
                        "beta": self.beta,
                        "iteration": self.iterations,
                        "passes": self.passes,
                        "eval_every": self.eval_every}}
