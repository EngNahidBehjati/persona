class ModelParamType:
    def __init__(self, name: str):
        self.name = name
        self.value_index = 0
        self.value = None
        self.iterable = False
        self.best_value = None  # if the param value is a list this points to the first value of the list otherwise it points to the value

    def set_value(self, value: list):
        if len(value) > 1:
            self.iterable = True
            start = value[0]
            stop = value[1]
            step = value[2]
            value = [start]
            while (start + step) <= stop:
                value.append(start + step)
                start += step
        self.value = value
        self.best_value = value[0]
        return

    def update_best_value(self, value):
        self.best_value = value


class ModelParams:
    number_of_topics = ModelParamType("number_of_topics")
    eval_every = ModelParamType("eval_every")
    chunk_size = ModelParamType("chunk_size")
    iterations = ModelParamType("iterations")
    passes = ModelParamType("passes")
    alpha = ModelParamType("alpha")
    beta = ModelParamType("beta")

    _iterables = list()
    _model_params_as_list = None

    @classmethod
    def get_model_params_as_list(cls) -> list:
        return [cls.number_of_topics.name,
                cls.eval_every.name,
                cls.chunk_size.name,
                cls.iterations.name,
                cls.passes.name,
                cls.alpha.name,
                cls.beta.name]

    @classmethod
    def _get_model_params_as_list(cls) -> list:
        return [cls.number_of_topics,
                cls.eval_every,
                cls.chunk_size,
                cls.iterations,
                cls.passes,
                cls.alpha,
                cls.beta]

    @classmethod
    def set_model_params_value(cls, model_args: dict):
        if cls._model_params_as_list is None:
            cls._model_params_as_list = cls._get_model_params_as_list()
        for param in cls._model_params_as_list:
            param.set_value(model_args[param.name])
            if param.iterable:
                cls._iterables.append(param)
        return

    @classmethod
    def _get_params_values(cls, test_param: ModelParamType):
        list_of_params_values = list()
        base_params_value = dict()
        params = cls._get_model_params_as_list()
        for param in params:
            base_params_value[param.name] = param.best_value
        test_param_values = test_param.value
        for value in test_param_values:
            params_list = base_params_value.copy()
            params_list[test_param.name] = value
            list_of_params_values.append(params_list)
        return list_of_params_values

    @classmethod
    def get_possible_model_params_values(cls) -> dict:
        test_params_dict_and_their_possible_values = dict()

        for test_param in cls._iterables:
            test_params_dict_and_their_possible_values[test_param.name] = cls._get_params_values(test_param)
        return test_params_dict_and_their_possible_values
