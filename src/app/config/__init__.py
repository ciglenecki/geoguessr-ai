from pathlib import Path

from dotenv import dotenv_values


_exception_message = """
.env file that is located at src/app/.env should contain the following variables: {}
Example of env file:

MODEL_DIRECTORY = "models"  # relative to the lumen-geoguesser directory
MODEL_EXTENSION = "ckpt"
PREDICT_BATCH_SIZE = 4
HOST = "localhost"
PORT = 8090
HOT_RELOAD = 0  # please don't enable hot reloading as it's unstable
"""


class EnvFileException(Exception):
    pass


_config = dotenv_values()
_keys = ["MODEL_DIRECTORY", "MODEL_EXTENSION", "PREDICT_BATCH_SIZE", "HOST", "PORT", "HOT_RELOAD"]
if any([key not in _config for key in _keys]):
    raise EnvFileException(_exception_message.format(_keys))

config = _config
