class VersionifyParams:
    data_version = "data_version"
    dictionary_version = "dictionary_version"
    model_version = "model_version"

    @classmethod
    def get_versionify_params_as_list(cls):
        return [cls.data_version,
                cls.dictionary_version,
                cls.model_version]
