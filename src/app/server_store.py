import os
from pathlib import Path
from typing import Dict

from fastapi.exceptions import FastAPIError

from app.server_config import server_config


class ServerStore:
    def __init__(self, server_config):
        self.server_config = server_config
        self.model_directory = str(server_config["MODEL_DIRECTORY"])
        self.model_extension = str(server_config["MODEL_EXTENSION"])
        self.batch_size = int(server_config["PREDICT_BATCH_SIZE"])
        self.cached_models = {}
        self.refresh_model_filepaths()

    def refresh_model_filepaths(self) -> Dict[str, Path]:
        """
        Returns dictionary {model_name: model_path} for all models in MODEL_DIRECTORY directory with MODEL_EXTENSION
        """

        if not os.path.isdir(self.model_directory):
            raise FastAPIError(
                "Path '{}' is not a directory".format(self.model_directory)
            )
        model_filepaths = {
            model_filepath.stem: model_filepath
            for model_filepath in Path(self.model_directory).rglob(
                "*." + self.model_extension
            )
        }
        if len(model_filepaths) == 0:
            raise FastAPIError(
                "Path '{}' should contain at least one model with the extension {}".format(
                    self.model_directory, self.model_extension
                )
            )
        self.model_filepaths = model_filepaths
        return model_filepaths


server_store = ServerStore(server_config)
