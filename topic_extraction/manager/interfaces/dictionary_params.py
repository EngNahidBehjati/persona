class DictionaryParams:
    no_below = "no_below"
    no_above = "no_above"
    n_most_frequent = "n_most_frequent"

    @classmethod
    def get_dictionary_params_as_list(cls):
        return [cls.no_below,
                cls.no_above,
                cls.n_most_frequent]
