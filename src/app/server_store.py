import os
from pathlib import Path
from typing import Dict

from fastapi.exceptions import FastAPIError

from app.config import config


class ServerStore:
    def __init__(self, config):
        self.config = config
        self.refresh_model_filepaths()

    def refresh_model_filepaths(self) -> Dict[str, Path]:
        """
        Returns dictionary {model_name: model_path} for all models in MODEL_DIRECTORY directory with MODEL_EXTENSION
        """
        model_directory = str(config["MODEL_DIRECTORY"])
        model_extension = str(config["MODEL_EXTENSION"])
        if not os.path.isdir(model_directory):
            raise FastAPIError("Path '{}' is not a directory".format(model_directory))
        model_filepaths = {
            model_filepath.stem: model_filepath
            for model_filepath in Path(model_directory).rglob("*." + model_extension)
        }
        if len(model_filepaths) == 0:
            raise FastAPIError(
                "Path '{}' should contain at least one model with the extension {}".format(
                    model_directory, model_extension
                )
            )
        self.model_filepaths = model_filepaths
        return model_filepaths


server_store = ServerStore(config)
