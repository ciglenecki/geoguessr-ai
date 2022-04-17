import argparse
import asyncio
import os
import sys
from pathlib import Path
from time import sleep

import aiohttp
import pandas as pd
from asgiref import sync
from tqdm import tqdm

from utils_functions import get_timestamp
from utils_paths import PATH_DATA_SAMPLER
import hashlib
import hmac
import base64
import urllib.parse as urlparse
import uuid


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(Path(PATH_DATA_SAMPLER, "coords_sample__n_1000000.csv")),
        type=str,
        help="Path to dataframe you want to enrich",
        required=True,
    )

    parser.add_argument(
        "--override",
        action="store_true",
        help="Overrides the current csv file",
    )

    parser.add_argument(
        "--key",
        type=str,
        help="Google Cloud API Key",
        required=True,
    )

    parser.add_argument(
        "--radius",
        type=int,
        help="Search radius",
        default=1000,
    )

    parser.add_argument(
        "--start-from-end",
        help="Start from the end of the dataframe",
        action="store_true",
    )

    parser.add_argument(
        "--batch",
        type=int,
        help="Number of requests that will be sent in batches",
        default=20000,
    )
    args = parser.parse_args(args)
    return args


def get_json_batch(url, params):
    async def get_all(url, params):
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout, read_bufsize=2**25) as session:

            async def fetch(url, params):
                async with session.get(url, params=params) as response:
                    return await response.json()

            return await asyncio.gather(*[fetch(url, params) for params in params])

    print("Getting batch...")
    result = sync.async_to_sync(get_all)(url, params)
    print("Batch got...")
    return result


def get_sigature_param(url, payload):
    url_raw = url + "&".join(["{}={}".format(k, v) for k, v in payload.items()])
    url = urlparse.urlparse(url_raw)

    url_to_sign = url.path + "?" + url.query
    secret = "6bxHPeMyhlDnddGACzrmsqi2Lsk="
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())
    return encoded_signature.decode()


def get_params_json(api_key, lat, lng, radius, url):
    payload = {
        "location": "{},{}".format(lat, lng),
        "radius": radius,
        "return_error_code": "true",
        "source": "outdoor",
        "key": api_key,
    }
    signature = get_sigature_param(url, payload)
    payload["signature"] = signature
    return payload


def add_columns_if_they_dont_exist(df: pd.DataFrame, cols):
    for new_col in ["row_idx", "status", "radius", "true_lat", "true_lng", "panorama_id", "uuid"]:
        if new_col not in df:
            df[new_col] = None
    return df


def get_rows_unresolved_status(df: pd.DataFrame):
    return df.loc[df["status"].isna(), :]


def flip_df(df: pd.DataFrame):
    return df.iloc[::-1]


def extract_features_from_json(json):
    status = json["status"]
    if status == "OK":
        return {
            "status": status,
            "true_lat": json["location"]["lat"],
            "true_lng": json["location"]["lng"],
            "panorama_id": json["pano_id"],
        }
    if status != "OK" and status != "ZERO_RESULTS":
        print("Unexpected response", json)
    return {
        "status": status,
        "true_lat": None,
        "true_lng": None,
        "panorama_id": None,
    }


def chunker(seq, size):
    for pos in range(0, len(seq), size):
        start = pos
        end = pos + size
        yield seq[start:end]


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


def main(args):
    args = parse_args(args)
    url = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    timestamp = get_timestamp()
    batch_size, radius, override, csv_path = args.batch, args.radius, args.override, args.csv

    path_dir = Path(csv_path).parents[0]
    base_name = Path(csv_path).stem
    out = Path(path_dir, base_name + "_modified_" + timestamp + ".csv")
    if override:
        out = Path(csv_path)

    print("Saving to file:", str(out))

    df = pd.read_csv(csv_path, index_col=[0])
    df = add_columns_if_they_dont_exist(df, ["row_idx", "status", "radius", "true_lat", "true_lng", "panorama_id"])

    for row in df[df["uuid"].isna(), :]:
        row
        df.loc[df["Name"] == name, "uuid"] = uuid.uuid4()

    if args.start_from_end:
        df = flip_df(df)

    df_unresolved = get_rows_unresolved_status(df)
    itterator = tqdm(chunker(df_unresolved, batch_size), desc="Waiting for the first itteration to finish...", total=len(df) // batch_size)
    for rows in itterator:
        indices = rows.index
        start_idx, end_idx = indices[0], indices[-1]

        lats, lngs = rows["latitude"], rows["longitude"]
        params = [get_params_json(args.key, lat, lng, radius, url) for lat, lng in zip(lats, lngs)]
        response_json_batch = get_json_batch(url, params)

        rows_batch = [extract_features_from_json(json) for json in response_json_batch]
        df_batch = pd.DataFrame(rows_batch)
        df_batch["row_idx"] = indices

        df_batch.set_index(indices, inplace=True)
        df.update(df_batch)

        safely_save_df(df, out)

        num_ok_status = len(df.loc[df["status"] == "OK", :])
        num_zero_results_status = len(df.loc[df["status"] == "ZERO_RESULTS", :])
        itterator.set_description("Last saved index: {}-{}, OK: ({}), ZERO ({})".format(start_idx, end_idx, num_ok_status, num_zero_results_status))
        sleep(60)


if __name__ == "__main__":
    main(sys.argv[1:])
    pass
