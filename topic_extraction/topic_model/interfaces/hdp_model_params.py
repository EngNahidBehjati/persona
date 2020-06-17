from topic_extraction.topic_model.interfaces.essential_params import EssentialParams


class HdpModelParams(EssentialParams):
    model_version = "model_version"

    chunk_size = "chunk_size"
    alpha = "alpha"
    beta = "beta"

    model_view = "model_view"
