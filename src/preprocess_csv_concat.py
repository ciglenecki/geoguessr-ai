"""Concaternates rows and columns of multiple csv dataframes"""
import argparse
import pandas as pd
import sys


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", metavar="file_a.csv file_b.csv")
    parser.add_argument("--out", metavar="csv", help="Path of the csv output")
    parser.add_argument("--no-out", action="store_true", help="Disable any dataframe or figure saving. Useful when calling inside other scripts")
    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)
    dfs = [pd.read_csv(csv_path) for csv_path in args.csv]
    df = pd.concat(dfs, ignore_index=True)
    df = df.loc[:, ["uuid", "latitude", "longitude"]]

    if args.no_out:
        return df

    print("Saving df ({}) to {}".format(len(df), args.out))
    df.to_csv(args.out)
    return df


if __name__ == "__main__":
    main(sys.argv[1:])
