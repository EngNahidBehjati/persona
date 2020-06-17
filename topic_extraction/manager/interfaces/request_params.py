class RequestParams:
    model_view = "model_view"
    requested_langs = "requested_lang"
    requested_models = "requested_models"

    @classmethod
    def get_request_params_as_list(cls):
        return [cls.model_view, cls.requested_langs, cls.requested_models]