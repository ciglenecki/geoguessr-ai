import argparse
import asyncio
import os
import sys
from pathlib import Path

import aiohttp
import pandas as pd
import requests
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

    args = parser.parse_args(args)
    return args


def get_json_batch(url, params):
    async def get_all(url, params):
        async with aiohttp.ClientSession() as session:

            async def fetch(url, params):
                async with session.get(url, params=params) as response:
                    return await response.json()

            return await asyncio.gather(*[fetch(url, params) for params in params])

    return sync.async_to_sync(get_all)(url, params)


def get_params_json(api_key, lat, lng, radius):
    return {
        "key": api_key,
        "location": "{}, {}".format(lat, lng),
        "radius": radius,
        "return_error_code": "true",
        "source": "outdoor",
    }


# def get_metadata(lat, lng, api_key, radius):
#     meta_base = "https://maps.googleapis.com/maps/api/streetview/metadata?"
#     meta_params = get_params_json(api_key, lat, lng, radius)
#     meta_response = requests.get(meta_base, params=meta_params)
#     return meta_response.json()


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


def main(args):
    args = parse_args(args)
    url = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    batch_size = 500

    timestamp = get_timestamp()
    radius = args.radius
    override = args.override

    path_dir = Path(args.csv).parents[0]
    base_name = Path(args.csv).stem
    out = Path(path_dir, base_name + "_modified_" + timestamp + ".csv")
    if override:
        out = Path(args.csv)
    print("Saving to file:", str(out))

    df = pd.read_csv(args.csv, index_col=[0])
    for new_col in ["row_idx", "status", "radius", "true_lat", "true_lng", "panorama_id"]:
        if new_col not in df:
            df[new_col] = None
    df = df.loc[df["status"].isna(), :]
    if args.start_from_end:
        df = df.iloc[::-1]

    itterator = tqdm(chunker(df, batch_size), desc="Waiting for the first itteration to finish...", total=len(df) // batch_size)

    for rows in itterator:
        indices = rows.index
        start_idx, end_idx = indices[0], indices[-1]

        lats, lngs = rows["latitude"], rows["longitude"]
        params = [get_params_json(args.key, lat, lng, radius) for lat, lng in zip(lats, lngs)]
        response_json_batch = get_json_batch(url, params)

        list_of_features = [extract_features_from_json(json) for json in response_json_batch]
        df_batch = pd.DataFrame(list_of_features, columns=["status", "true_lat", "true_lng", "panorama_id"])
        df_batch["row_idx"] = indices
        df_batch.set_index(indices, inplace=True)
        df.update(df_batch)

        path_tmp = Path(str(out) + ".tmp")
        path_bak = Path(str(out) + ".bak")
        df.to_csv(path_tmp, mode="w+", index=True, header=True)

        if os.path.isfile(out):
            os.rename(out, path_bak)
        os.rename(path_tmp, out)

        if os.path.isfile(path_bak):
            os.remove(path_bak)

        num_ok_status = len(df.loc[df["status"] == "OK", :])
        num_zero_results_status = len(df.loc[df["status"] == "ZERO_RESULTS", :])
        itterator.set_description("Last saved index: {}-{}, OK: ({}), ZERO ({})".format(start_idx, end_idx, num_ok_status, num_zero_results_status))


if __name__ == "__main__":
    main(sys.argv[1:])
    pass
