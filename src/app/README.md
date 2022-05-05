### Before consuming the endpoints set the variable MODEL_DIRECTORY in .env to a directory that contains model checkpoints (.ckpt). Models outside of this directory can't be used for inference via endpoints

# How to run the server?

1. your current path should be the project itself
2. if you didn't follow the [Setup](../../README.md#⬇️-setup) in the initial [README.md](../../README.md)
2. set the variable MODEL_DIRECTORY in .env to a directory that contains model checkpoints (.ckpt)
4. run the server with the following command `python3 src/app/main.py`

(venv) username@pc:~/lumen-geoguesser$ python3 src/app/main.py
