"""
Specified paths -- directory structure
"""
from os import getcwd
from pathlib import Path

PATH_CWD = Path(getcwd())
PATH_DATA = Path(PATH_CWD, "data")
PATH_DATA_EXTERNAL = Path(PATH_DATA, "external.ignoreme")
PATH_DATA_RAW = Path("/media/filip/DA2A5AE02A5AB8E9/Dokumenti/LUMEN/Filip_faks/lumen-datasci-2022-train/")
PATH_REPORT = Path(PATH_DATA, "reports")
PATH_FIGURE = Path(PATH_REPORT, "figures")
PATH_MODEL = Path(PATH_CWD, "models")

if __name__ == "__main__":
    pass
