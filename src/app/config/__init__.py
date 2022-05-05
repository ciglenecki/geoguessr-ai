from pathlib import Path

from dotenv import dotenv_values

config = dotenv_values()

print("Init", config)
