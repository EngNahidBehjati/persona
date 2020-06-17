from argparse import ArgumentParser

from numpy import float_


class Execution:
    def __init__(self):
        self.parser = ArgumentParser()

    def set_topic_extraction_args(self) -> None:
        # data_folder_path
        self.parser.add_argument("-folder-path", dest="data_folder_path", type=str, required=True,
                                 help="path to the folder that is the container of data e.g. /home/virtux/data")
        # data_file_name
        self.parser.add_argument("-file-name", dest="data_file_name", type=str, required=True,
                                 help="the name of the data file")
        # data_file_extension
        self.parser.add_argument("-ex", dest="data_file_extension", type=str, required=True,
                                 help="the extension of train data file e.g. txt or json")
        # data_version
        self.parser.add_argument("-data-v", dest="data_version", type=int, required=True,
                                 help="the version of train data")
        # data_file_type
        self.parser.add_argument("-data-type", dest="data_file_type", type=str,
                                 choices=["CommonCrawl", "Json", "SemiJson"],
                                 default="CommonCrawl",
                                 help="""data file is either a full json file or a semi one. if it's a full json this 
                                 variable should be True (it is True by default) otherwise it has to be False""")
        # include_tags
        self.parser.add_argument("-in-tags", dest="include_tags", type=str, nargs='?', default=None,
                                 help="""html tags that their text are important, so they are the only text that 
                                 going to process (either this should be passed or -ex-tags)e.g. h1 h2 p strong a """)
        # exclude_tags
        self.parser.add_argument("-ex-tags", dest="exclude_tags", type=str, nargs='?', default=None,
                                 help="""html tags that their text are not important, so they are the only text that 
                                 not going to process (either this should be passed or -in-tags) e.g. button div form """)
        # dictionary_version
        self.parser.add_argument("-dict-v", dest="dictionary_version", type=int, required=True,
                                 help="the version of dictionary")
        # model_version
        self.parser.add_argument("-model-v", dest="model_version", type=str, required=True,
                                 help="the version of topic model")

        # topic_number
        self.parser.add_argument("-t-num", dest="number_of_topics", type=int, nargs="*", default=[10],
                                 help="""number of topics on data file; 
                                 It is either a list of one exact value for 'number_of_topics' or 
                                 it can be a list with 3 elements [start, stop, step]. 
                                 These represent a rang of values to test 'number_of_topics'""")
        # worker_number
        # self.parser.add_argument("-w-num", dest="number_of_workers", type=int, default=4,
        #                          help="number of topics on data file")
        # chunk_size
        self.parser.add_argument("-chunk-size", dest="chunk_size", type=int, nargs="*", default=[200],
                                 help="""chunk size value for lda topic modeling; 
                                 It is either a list of one exact value for 'chunk_size' or 
                                 it can be a list with 3 elements [start, stop, step]. 
                                 These represent a rang of values to test 'chunk_size'""")
        # iterations
        self.parser.add_argument("-i", dest="iterations", type=int, nargs="*", default=[400],
                                 help="""iterations value for lda topic modeling;
                                 It is either a list of one exact value for 'iterations' or 
                                 it can be a list with 3 elements [start, stop, step]. 
                                 These represent a rang of values to test 'iterations'""")
        # passes
        self.parser.add_argument("-p", dest="passes", type=int, nargs="*", default=[20],
                                 help="""passes value for lda topic modeling;
                                 It is either a list of one exact value for 'passes' or 
                                 it can be a list with 3 elements [start, stop, step]. 
                                 These represent a rang of values to test 'passes'""")
        # alpha
        self.parser.add_argument("-a", dest="alpha", type=str, default=["symmetric"],
                                 help="""alpha value for lda topic modeling, it should be string;""")
        # beta
        self.parser.add_argument("-b", dest="beta", type=float, nargs="*", default=[1.0],
                                 help="""eta value for lda topic modeling, it should be string;
                                 It is either a list of one exact value for 'beta' or 
                                 it can be a list with 3 elements [start, stop, step]. 
                                 These represent a rang of values to test 'beta'""")
        # no_below
        self.parser.add_argument("-n-b", dest="no_below", type=int, default=1,
                                 help="""remove tokens that happens in less than 'no_below' documents;
                                 It is either a list of one exact value for 'no_below' or 
                                 it can be a list with 3 elements [start, stop, step]. 
                                 These represent a rang of values to test 'no_below'""")
        # no_above
        self.parser.add_argument("-n-a", dest="no_above", type=float, default=0.6,
                                 help="""remove tokens that happens in more than 'no_above' documents;
                                 It is either a list of one exact value for 'no_above' or 
                                 it can be a list with 3 elements [start, stop, step]. 
                                 These represent a rang of values to test 'no_above'""")
        # eval_every
        self.parser.add_argument("-e-e", dest="eval_every", type=int, nargs="*", default=[10],
                                 help="""By turning on the eval_every flag we’re able to process the corpus in chunks;
                                 It is either a list of one exact value for 'eval_every' or 
                                 it can be a list with 3 elements [start, stop, step]. 
                                 These represent a rang of values to test 'eval_every'""")
        # n_most_frequent
        self.parser.add_argument("-most-frequent", dest="n_most_frequent", type=int, default=1,
                                 help="""the number of most frequent tokens to delete from dictionary;
                                 It is either a list of one exact value for 'n_most_frequent' or 
                                 it can be a list with 3 elements [start, stop, step]. 
                                 These represent a rang of values to test 'n_most_frequent'""")
        # list_of_requested_models
        self.parser.add_argument("-models", dest="requested_models", nargs="*",
                                 choices=["lda", "mallet", "lsi", "hdp"],
                                 default=["lda", "mallet", "lsi", "hdp"],
                                 help="""list of models to train on data""")
        # visualization_of_model
        self.parser.add_argument("-m-view", dest="model_view", type=bool,
                                 default=False,
                                 help="""If True, each model get its all visualization plots(model_view)""")
        # requested_lang
        self.parser.add_argument("-lang", dest="requested_lang", nargs="*",
                                 default=["en", "de", "es", "el", "pt", "it", "nl", "nb", "lt"],
                                 choices=["en", "de", "es", "el", "pt", "it", "nl", "nb", "lt"],
                                 help="""The languages that we want to train data for; there is a list of languages
                                      that we support and they shouldn't be anything else""")

        return

    def parse_topic_extraction_command_line(self, args: list):
        self.set_topic_extraction_args()
        return self.parser.parse_args(args)
