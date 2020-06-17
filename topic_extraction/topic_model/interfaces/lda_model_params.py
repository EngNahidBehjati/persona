from topic_extraction.topic_model.interfaces.essential_params import EssentialParams


class LdaModelParams(EssentialParams):
    model_version = "model_version"

    number_of_topics = "number_of_topics"
    chunk_size = "chunk_size"
    alpha = "alpha"
    beta = "beta"
    iterations = "iterations"
    passes = "passes"
    eval_every = "eval_every"

    model_view = "model_view"
