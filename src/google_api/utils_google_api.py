import argparse
import asyncio
import base64
import hashlib
import hmac
import os
import sys
import urllib.parse as urlparse
from pathlib import Path
from time import sleep

import aiohttp
import pandas as pd
from asgiref import sync
from tqdm import tqdm


def chunker(seq, size):
    for pos in range(0, len(seq), size):
        start = pos
        end = pos + size
        yield seq[start:end]


def add_columns_if_they_dont_exist(df: pd.DataFrame, cols):
    for new_col in cols:
        if new_col not in df:
            df[new_col] = None
    return df


def get_rows_unresolved_status(df: pd.DataFrame):
    return df.loc[df["status"].isna(), :]


def flip_df(df: pd.DataFrame):
    return df.iloc[::-1]


def safely_save_df(df: pd.DataFrame, filepath: Path):
    print("Saving file...", filepath)
    path_tmp = Path(str(filepath) + ".tmp")
    path_bak = Path(str(filepath) + ".bak")
    df.to_csv(path_tmp, mode="w+", index=True, header=True)

    if os.path.isfile(filepath):
        os.rename(filepath, path_bak)
    os.rename(path_tmp, filepath)

    if os.path.isfile(path_bak):
        os.remove(path_bak)


def get_signature_param(url, payload, secret):
    """
    explained here: https://developers.google.com/maps/documentation/streetview/digital-signature#python
    """

    url_raw = url + "&".join(["{}={}".format(k, v) for k, v in payload.items()])
    url = urlparse.urlparse(url_raw)

    url_to_sign = url.path + "?" + url.query

    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())
    return encoded_signature.decode()
