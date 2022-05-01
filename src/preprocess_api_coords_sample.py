"""
Enriches the coords_sample__n_<int>.csv.csv file. This file originally contains uniformly sampled values (lat, lng). The problem is that an image might not exist on the sampled (lat, lng) location. This file goes through the rows of coords_sample__n_<int>.csv and populates additional columns which provide information if the street view image in fact exists on that location. If it does, additional metadata will be writen. 

E.g.
    Input (--csv)
        sample_longitude,sample_latitude
        ---------------------------------------
        0,16.725157101650694,45.80887959964089
        1,15.87582307662098,44.38751679151152

    Out (coords_sample__n_<int>_modified_<timestamp>):
        ,sample_longitude,sample_latitude,row_idx,status,radius,latitude,longitude,panorama_id,uuid
        ------------------------------------------------------------------------------------------------
        0,16.725157101650694,45.80887959964089,0.0,OK,,45.8102583442322,16.72203900718308,rC0RohGwjXkQX3u7rOi2XQ,a9615845-f0c3-4a46-a5fe-35e7ca32f6f0
        1,15.87582307662098,44.38751679151152,1.0,OK,,44.38660988435789,15.88616327245129,Akbp_5oGmjsSkVFosLfhSQ,c3062ee3-f172-4c8d-9c52-11fc274f2a6a
        2,17.50461126873125,45.72957536481877,2.0,ZERO_RESULTS,,,,,9a752d43-2e07-4803-bf0e-1d8940f19265

"""

import argparse
import asyncio
import base64
import hashlib
import hmac
import os
import sys
import urllib.parse as urlparse
import uuid
from pathlib import Path
from time import sleep
from typing import Any, Dict, List

import aiohttp
import pandas as pd
from asgiref import sync
from tqdm import tqdm

from utils_functions import get_timestamp
from utils_paths import PATH_DATA_SAMPLER


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
        help="Google Cloud API Key https://console.cloud.google.com/google/maps-apis/credentials?project=lumen-data-science",
        required=True,
    )

    parser.add_argument(
        "--signature",
        type=str,
        help="Signature secret for sigining requests https://console.cloud.google.com/google/maps-apis/credentials?project=lumen-data-science",
        required=True,
    )

    parser.add_argument(
        "--radius",
        type=int,
        help="Search radius for a given location",
        default=500,
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
        default=20_000,
    )
    args = parser.parse_args(args)
    return args


def get_json_batch(url, params_list: List[Dict[str, Any]]):
    """
    Args:
        url - base url that will be used for querying e.g. https://maps.googleapis.com/maps/api/streetview/metadata
        params - list of params. Params are parameters of the url request
    Returns:
        list of json responses

    For each set of params in params_list this function will call get request and gather all responses in a list
    """

    async def get_all(url, params_list):
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout, read_bufsize=2**25) as session:

            async def fetch(url, params):
                async with session.get(url, params=params) as response:
                    return await response.json()

            return await asyncio.gather(*[fetch(url, params) for params in params_list])

    print("Getting batch...")
    result = sync.async_to_sync(get_all)(url, params_list)
    print("Batch got...")
    return result


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


def get_params_json(api_key, signature, lat, lng, radius, url):
    """Constructs the json object (params) used for the get request"""

    payload = {
        "location": "{},{}".format(lat, lng),
        "radius": radius,
        "return_error_code": "true",
        "source": "outdoor",
        "key": api_key,
    }
    signature = get_signature_param(url, payload, signature)
    payload["signature"] = signature
    return payload


def add_columns_if_they_dont_exist(df: pd.DataFrame, cols):
    for new_col in cols:
        if new_col not in df:
            df[new_col] = None
    return df


def get_rows_with_unresolved_status(df: pd.DataFrame):
    return df.loc[df["status"].isna(), :]


def flip_df(df: pd.DataFrame):
    return df.iloc[::-1]


def upsert_uuids(df: pd.DataFrame):
    """Add uuid attribute to rows where uuid is NaN"""
    mask_no_uuid = df["uuid"].isna()
    df.loc[mask_no_uuid, "uuid"] = [uuid.uuid4() for _ in range(len(df.loc[mask_no_uuid, :].index))]
    return df


def extract_features_from_json(json):
    status = json["status"]
    if status == "OK":
        return {
            "status": status,
            "latitude": json["location"]["lat"],
            "longitude": json["location"]["lng"],
            "panorama_id": json["pano_id"],
        }
    if status != "OK" and status != "ZERO_RESULTS":
        print("Unexpected response", json)
    return {
        "status": status,
        "latitude": None,
        "longitude": None,
        "panorama_id": None,
    }


def chunker(df: pd.DataFrame, size: int):
    """Generator which itterates over the dataframe by taking chunks of rows of size `size`"""
    for pos in range(0, len(df), size):
        start = pos
        end = pos + size
        yield df[start:end]


def safely_save_df(df: pd.DataFrame, filepath: Path):
    """Safely save the dataframe by using and removing temporary files"""

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
    GOOGLE_RATE_LIMIT_SECONDS = 60  # 25000 requests in 60 seconds
    url = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    timestamp = get_timestamp()
    batch_size, radius, override, csv_path, signature, key = (
        args.batch,
        args.radius,
        args.override,
        args.csv,
        args.signature,
        args.key,
    )

    path_dir = Path(csv_path).parents[0]
    base_name = Path(csv_path).stem
    out = Path(path_dir, base_name + "_modified_" + timestamp + ".csv")
    if override:
        out = Path(csv_path)

    print("Saving to file:", str(out))

    df = pd.read_csv(
        csv_path, index_col=[0]
    )  # index_col - column which defines how rows are indexed (e.g. when `df.loc` is called)
    df = add_columns_if_they_dont_exist(
        df,
        ["row_idx", "status", "radius", "latitude", "longitude", "panorama_id", "uuid"],
    )
    df = upsert_uuids(df)
    df = flip_df(df) if args.start_from_end else df

    df_unresolved = get_rows_with_unresolved_status(df)

    base_itterator = chunker(df_unresolved, batch_size)
    pretty_itterator = tqdm(
        base_itterator,
        desc="Waiting for the first itteration to finish...",
        total=len(df) // batch_size,
    )

    for rows in pretty_itterator:
        indices = rows.index
        start_idx, end_idx = indices[0], indices[-1]

        lats, lngs = rows["sample_latitude"], rows["sample_longitude"]
        params = [get_params_json(key, signature, lat, lng, radius, url) for lat, lng in zip(lats, lngs)]
        response_json_batch = get_json_batch(url, params)

        rows_batch = [extract_features_from_json(json) for json in response_json_batch]

        """ Create temporary dataframe which will update the original dataframe"""
        df_batch = pd.DataFrame(rows_batch)
        df_batch["row_idx"] = indices
        df_batch.set_index(indices, inplace=True)
        df.update(df_batch)

        safely_save_df(df, out)

        num_ok_status = len(df.loc[df["status"] == "OK", :])
        num_zero_results_status = len(df.loc[df["status"] == "ZERO_RESULTS", :])
        pretty_itterator.set_description(
            "Last saved index range: [{}, {}], OK: {}, ZERO {}".format(
                start_idx, end_idx, num_ok_status, num_zero_results_status
            )
        )
        sleep(GOOGLE_RATE_LIMIT_SECONDS)


if __name__ == "__main__":
    main(sys.argv[1:])
    pass
