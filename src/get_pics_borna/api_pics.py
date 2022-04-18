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


def get_response_batch(url, params):
    async def get_all(url, params):
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout, read_bufsize=2**25) as session:

            async def fetch(url, params):
                async with session.get(url, params=params) as response:
                    # return await response.json()

                    # if response.status == 200:
                    #     directory = f"images/{params['uuid']}"
                    # if not aiofiles.os.path.exists(directory):
                    #     aiofiles.mkdir(directory)

                    # async with aiofiles.open(f"{directory}/{params['heading']}.jpg", 'wb') as file:
                    #     file.write(response.content)

                    return (await response.content.read(), response.ok, params)


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


def get_params_json(api_key, pano_id, unique_id, heading, url):
    payload = {
        "pano": pano_id,
        "heading": heading,
        "size": "640x640",
        "return_error_code": "true",
        "key": api_key,
        "uuid": unique_id,
    }
    signature = get_sigature_param(url, payload)
    payload["signature"] = signature
    return payload


def add_columns_if_they_dont_exist(df: pd.DataFrame, cols):
    for new_col in cols:
        if new_col not in df:
            df[new_col] = None
    return df


def get_rows_unresolved_status(df: pd.DataFrame):
    return df.loc[df["status"].isna(), :]


def flip_df(df: pd.DataFrame):
    return df.iloc[::-1]


# def extract_features_from_json(json):
#     status = json["status"]
#     if status == "OK":
#         return {
#             "status": status,
#             "true_lat": json["location"]["lat"],
#             "true_lng": json["location"]["lng"],
#             "panorama_id": json["pano_id"],
#         }
#     if status != "OK" and status != "ZERO_RESULTS":
#         print("Unexpected response", json)
#     return {
#         "status": status,
#         "true_lat": None,
#         "true_lng": None,
#         "panorama_id": None,
#     }

# def extract_features_from_response(resp, params):
#     if resp.ok:
#         return{
#             "image": resp.content,
#             "pano_id": params["pano"],
#             "heading": params["heading"],
#         }


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

import csv
def safely_save_csv(data, filepath: Path):
    # print("Saving file...", filepath)
    # path_tmp = Path(str(filepath) + ".tmp")
    # path_bak = Path(str(filepath) + ".bak")

    with open(filepath, 'w+', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)

    # if os.path.isfile(filepath):
    #     os.rename(filepath, path_bak)
    # os.rename(path_tmp, filepath)

    # if os.path.isfile(path_bak):
    #     os.remove(path_bak)



def main(args):
    args = parse_args(args)
    # url = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    url = "https://maps.googleapis.com/maps/api/streetview?"
    timestamp = get_timestamp()
    batch_size, radius, override, csv_path = args.batch, args.radius, args.override, args.csv

    path_dir = Path(csv_path).parents[0]
    base_name = Path(csv_path).stem
    out = Path(path_dir, base_name + "_modified_" + timestamp + ".csv")
    if override:
        out = Path(csv_path)

    print("Saving to file:", str(out))

    df = pd.read_csv(csv_path, index_col=[0])
    df = add_columns_if_they_dont_exist(df, ["row_idx", "status", "radius", "true_lat", "true_lng", "panorama_id", "uuid"])

    no_uuid_rows = df["uuid"].isna()
    df.loc[no_uuid_rows, "uuid"] = [uuid.uuid4() for _ in range(len(df.loc[no_uuid_rows, :].index))]

    if args.start_from_end:
        df = flip_df(df)

    # df_unresolved = get_rows_unresolved_status(df)
    itterator = tqdm(chunker(df, batch_size), desc="Waiting for the first itteration to finish...", total=len(df) // batch_size)
    for rows in itterator:
        indices = rows.index
        start_idx, end_idx = indices[0], indices[-1]

        # lats, lngs = rows["latitude"], rows["longitude"]
        pano_ids = rows["panorama_id"]
        unique_ids = rows["uuid"]

        params = [get_params_json(args.key, pano_id, unique_id, heading, url) for pano_id, unique_id in zip(pano_ids, unique_ids) for heading in (0, 90, 180, 270)]
        response_batch = get_response_batch(url, params)

        # rows_batch = [extract_features_from_response(resp, params) for resp, params in response_batch]

        output_batch = []

        for resp_content, resp_ok, params in response_batch:
            # params["pano"]
            # params["heading"]
            # unique_id = df.loc[df["panorama_id"] == params["pano"], "uuid"].to_string().split()[-1]

            output_batch.append([
                params["pano"],
                params["uuid"],
                params["heading"],
                resp_ok,
            ])

            if resp_ok:

                directory = f"images/{params['uuid']}"
                # # print(f"\n\n\nbeforeif dir: {directory}\n\n\n")
                # if not os.path.exists(directory):
                #     # print(f"\n\n\ndoesnotexist dir: {directory}\n\n\n")
                #     os.makedirs(directory)

                try:
                    os.makedirs(directory)
                except FileExistsError:
                    pass
                

                with open(f"{directory}/{params['heading']}.jpg", 'wb') as file:
                    file.write(resp_content)




        # df_batch = pd.DataFrame(output_batch)
        # df_batch["row_idx"] = indices

        # df_batch.set_index(indices, inplace=True)
        # df.update(df_batch)

        # safely_save_df(df_batch, out)
        safely_save_csv(output_batch, "pics_output.csv")

        # num_ok_status = len(df.loc[df["resp_ok"] == True, :])
        num_ok_status = len(list(filter(lambda x: x.get("resp_ok") == True, output_batch)))
        num_zero_results_status = len(output_batch) - num_ok_status
        itterator.set_description("Last saved index: {}-{}, OK: ({}), ZERO ({})".format(start_idx, end_idx, num_ok_status, num_zero_results_status))
        sleep(1)


if __name__ == "__main__":
    main(sys.argv[1:])
    pass
