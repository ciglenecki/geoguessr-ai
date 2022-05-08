import sys
from glob import glob
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve().parents[0]))


import argparse
import asyncio
import base64
import csv
import hashlib
import hmac
import os
import sys
import urllib.parse as urlparse
import uuid
from pathlib import Path
from time import sleep

import aiohttp
import pandas as pd
from asgiref import sync
from tqdm import tqdm
from utils_google_api import add_columns_if_they_dont_exist, chunker, flip_df, get_signature_param

from utils_functions import flatten, get_dirs_only, get_timestamp
from utils_paths import PATH_DATA_EXAMPLE_IMAGES, PATH_DATA_SAMPLER


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
        "--existing-image-directory",
        default=PATH_DATA_EXAMPLE_IMAGES,
        type=str,
        help="Directory with already downloaded images",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--key",
        type=str,
        help="Google Cloud API Key",
        required=True,
    )
    parser.add_argument(
        "--signature",
        type=str,
        help="Google Cloud API signature",
        required=True,
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
        default=1,
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        help="Location where images will be downloaded",
        required=True,
    )

    args = parser.parse_args(args)
    return args


def get_response_batch(url, params):
    async def get_all(url, params):
        print(params)
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout, read_bufsize=2**27) as session:

            async def fetch(url, params):
                print(url, params)
                async with session.get(url, params=params) as response:
                    print(url, params)
                    return (await response.content.read(), response.ok, params)

            return await asyncio.gather(*[fetch(url, params) for params in params])

    print("Getting batch...")
    result = sync.async_to_sync(get_all)(url, params)
    print("Batch got...")
    return result


def get_params_json(api_key, signature, pano_id, unique_id, heading, url):
    payload = {
        "pano": pano_id,
        "heading": heading,
        "size": "640x640",
        "return_error_code": "true",
        "key": api_key,
        "uuid": unique_id,
    }
    signature = get_signature_param(url, payload, signature)
    payload["signature"] = signature
    return payload


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


def safely_save_csv(data, filepath: Path):

    with open(filepath, "w+", newline="") as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)


def main(args):
    args = parse_args(args)
    url = "https://maps.googleapis.com/maps/api/streetview?"
    timestamp = get_timestamp()
    batch_size, csv_path, existing_image_directory, out_dir = (
        args.batch,
        args.csv,
        args.existing_image_directory,
        args.out_dir,
    )
    GOOGLE_RATE_LIMIT_SECONDS = 30

    os.makedirs(args.out_dir, exist_ok=True)

    print(existing_image_directory)
    image_paths = flatten(
        [glob(str(Path(images_dir, "**/*.jpg")), recursive=True) for images_dir in existing_image_directory]
    )

    existing_uuids = [Path(image_path).parent.stem for image_path in image_paths]
    existing_uuids = list(set(existing_uuids))

    base_name = Path(csv_path).stem
    out_csv = Path(out_dir, base_name + "_modified_downloaded_" + timestamp + ".csv")
    out_images = Path(out_dir, "images")

    print("Saving to file:", str(out_csv))

    df = pd.read_csv(csv_path, index_col=[0])
    print(df)

    mask_no_pano_id = df["panorama_id"].isna() == True
    mask_no_uuid = df["uuid"].isna() == True

    maks_images_that_exist = df["uuid"].isin(existing_uuids)
    df = df.loc[~(maks_images_that_exist | mask_no_pano_id | mask_no_uuid), :]

    if args.start_from_end:
        df = flip_df(df)

    itterator = tqdm(
        chunker(df, batch_size), desc="Waiting for the first itteration to finish...", total=len(df) // batch_size
    )

    for rows in itterator:
        indices = rows.index
        start_idx, end_idx = indices[0], indices[-1]

        pano_ids = rows["panorama_id"]
        unique_ids = rows["uuid"]

        params = [
            get_params_json(args.key, args.signature, pano_id, unique_id, heading, url)
            for pano_id, unique_id in zip(pano_ids, unique_ids)
            for heading in (0, 90, 180, 270)
        ]
        response_batch = get_response_batch(url, params)

        output_batch = []

        for resp_content, resp_ok, params in response_batch:
            pano_id, uuid_str, heading = params["pano"], params["uuid"], params["heading"]
            output_batch.append([pano_id, uuid_str, heading, resp_ok])
            if resp_ok:
                image_directory = Path(out_images, uuid_str)
                os.makedirs(image_directory, exist_ok=True)
                image_path = Path(image_directory, "{}.jpg".format(heading))
                with open(image_path, "wb") as file:
                    file.write(resp_content)

        safely_save_csv(output_batch, Path(args.out_dir, "api_pics_{}.csv".format(timestamp)))

        num_ok_status = len(list(filter(lambda x: x[-1] == True, output_batch)))
        num_zero_results_status = len(output_batch) - num_ok_status
        itterator.set_description(
            "Last saved index: {}-{}, OK: ({}), ZERO ({})".format(
                start_idx, end_idx, num_ok_status, num_zero_results_status
            )
        )
        sleep(GOOGLE_RATE_LIMIT_SECONDS)


if __name__ == "__main__":
    main(sys.argv[1:])
    pass
