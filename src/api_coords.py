import sys
import csv
import argparse
from pathlib import Path
from urllib import response
from utils_functions import get_timestamp
from utils_paths import PATH_DATA_SAMPLER
import pandas as pd
import requests
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(Path(PATH_DATA_SAMPLER, "coords_sample__n_1000000.csv")), type=str, help="Path to dataframe you want to enrich", required=True)

    parser.add_argument(
        "--out",
        type=str,
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
        default=500,
    )

    parser.add_argument(
        "--start-from-end",
        help="Start from the end of the dataframe",
        action="store_true",
    )

    args = parser.parse_args(args)
    return args


def get_metadata(lat, lng, api_key, radius):
    meta_base = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    meta_params = {
        "key": api_key,
        "location": "{}, {}".format(lat, lng),
        "radius": radius,
        "return_error_code": "true",
        "source": "outdoor",
    }
    meta_response = requests.get(meta_base, params=meta_params)
    return meta_response.json()


def handle_json_response(json):
    status = json["status"]
    if status == "OK":
        true_lat = json["location"]["lat"]
        true_lng = json["location"]["lng"]
        panorama_id = json["pano_id"]
        return status, true_lat, true_lng, panorama_id
    if status != "OK" and status != "ZERO_RESULTS":
        print("Unexpected response", json)
    return status, None, None, None


def main(args):
    args = parse_args(args)

    batch_size = 10000
    timestamp = get_timestamp()
    radius = args.radius
    out = args.out

    if out is None:
        path_dir = Path(args.csv).parents[0]
        base_name = Path(args.csv).stem
        out = Path(path_dir, base_name + "_modified_" + timestamp + ".csv")
    print("Saving to file:", str(out))

    df = pd.read_csv(args.csv, index_col=[0])
    for new_col in ["row_idx", "status", "radius", "true_lat", "true_lng", "panorama_id"]:
        if new_col not in df:
            df[new_col] = None

    log_status_dict = {}
    df = df.loc[df["status"].isna(), :]
    if args.start_from_end:
        df = df.iloc[::-1]
    itterator = tqdm(enumerate(df.iterrows()), desc=str(log_status_dict), total=len(df))

    for i, (row_idx, row) in itterator:
        if (i + 1) % batch_size == 0:
            """Safely save csv to file"""
            path_tmp = Path(str(out) + ".tmp")
            path_bak = Path(str(out) + ".bak")
            df.to_csv(path_tmp, mode="w+", index=True, header=True)

            if os.path.isfile(out):
                os.rename(out, path_bak)
            os.rename(path_tmp, out)

            if os.path.isfile(path_bak):
                os.remove(path_bak)

            itterator.set_description("Last saved index: {}, Stats {}".format(row_idx, str(log_status_dict)))

        lat, lng = row["latitude"], row["longitude"]

        response_json = get_metadata(lat, lng, args.key, radius)
        status, true_lat, true_lng, panorama_id = handle_json_response(response_json)

        # df.at[row_idx, "row_idx"] = row_idx
        df.at[row_idx, "true_lat"] = true_lat
        df.at[row_idx, "true_lng"] = true_lng
        df.at[row_idx, "radius"] = radius
        df.at[row_idx, "status"] = status
        df.at[row_idx, "panorama_id"] = panorama_id

        if status not in log_status_dict:
            log_status_dict[status] = 0
        log_status_dict[status] += 1


if __name__ == "__main__":
    main(sys.argv[1:])
    pass
