from pathlib import Path
import pandas as pd
from utils_paths import PATH_DATA, PATH_DATA_RAW, PATH_DATA_SAMPLER

our_df = Path(PATH_DATA, "complete", "external.csv")

their_df = Path(PATH_DATA_RAW, "data.csv")

df_ours = pd.read_csv(our_df)
df_theirs = pd.read_csv(their_df)

df = pd.concat([df_theirs, df_ours])
print(df[df.duplicated(["latitude", "longitude"])].groupby(["latitude", "longitude"]).size())

df.to_csv(Path(PATH_DATA, "complete", "data.csv"))
print(len(df))
