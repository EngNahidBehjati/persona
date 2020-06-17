class DataParams:
    include_tags = "include_tags"
    exclude_tags = "exclude_tags"
    data_file_type = "data_file_type"
    data_file_name = "data_file_name"
    data_folder_path = "data_folder_path"
    data_file_extension = "data_file_extension"

    @classmethod
    def get_data_params_as_list(cls):
        return [cls.data_folder_path,
                cls.data_file_name,
                cls.data_file_extension,
                cls.data_file_type,
                cls.include_tags,
                cls.exclude_tags]

