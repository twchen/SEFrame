import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    choices=['gowalla', 'delicious', 'foursquare'],
    required=True,
    help='the dataset name',
)
parser.add_argument(
    '--input-dir',
    type=Path,
    default='datasets',
    help='the directory containing the raw data files',
)
parser.add_argument(
    '--output-dir',
    type=Path,
    default='datasets',
    help='the directory to store the preprocessed dataset',
)
parser.add_argument(
    '--train-split', type=float, default=0.6, help='the ratio of the training set'
)
parser.add_argument(
    '--max-len', type=int, default=50, help='the maximum session length'
)
args = parser.parse_args()

FILENAMES = {
    'gowalla': ['loc-gowalla_totalCheckins.txt', 'loc-gowalla_edges.txt'],
    'delicious': [
        'user_taggedbookmarks-timestamps.dat',
        'user_contacts-timestamps.dat',
        'raw_POIs.txt',
    ],
    'foursquare': [
        'dataset_WWW_Checkins_anonymized.txt',
        'dataset_WWW_friendship_new.txt',
    ],
}

filenames = FILENAMES[args.dataset]
for filename in filenames:
    if not (args.input_dir / filename).exists():
        print(f'File {filename} not found in {args.input_dir}', file=sys.stderr)
        sys.exit(1)
clicks = args.input_dir / filenames[0]
edges = args.input_dir / filenames[1]

import numpy as np
import pandas as pd
from srs.utils.data.preprocess import preprocess, update_id

print('reading dataset...')
if args.dataset == 'gowalla':
    args.interval = pd.Timedelta(days=1)
    args.max_items = 50000

    df = pd.read_csv(
        clicks,
        sep='\t',
        header=None,
        names=['userId', 'timestamp', 'latitude', 'longitude', 'itemId'],
        parse_dates=['timestamp'],
        infer_datetime_format=True,
    )
    df_clicks = df[['userId', 'timestamp', 'itemId']]
    df_loc = df.groupby('itemId').agg({
        'latitude': lambda col: col.iloc[0],
        'longitude': lambda col: col.iloc[0],
    }).reset_index()
    df_edges = pd.read_csv(edges, sep='\t', header=None, names=['follower', 'followee'])
elif args.dataset == 'delicious':

    df_clicks = pd.read_csv(
        clicks,
        sep='\t',
        skiprows=1,
        header=None,
        names=['userId', 'sessionId', 'itemId', 'timestamp'],
    )
    df_clicks['timestamp'] = pd.to_datetime(df_clicks.timestamp, unit='ms')
    df_loc = None
    df_edges = pd.read_csv(
        edges,
        sep='\t',
        skiprows=1,
        header=None,
        usecols=[0, 1],
        names=['follower', 'followee'],
    )
elif args.dataset == 'foursquare':
    args.interval = pd.Timedelta(days=1)
    args.max_users = 50000
    args.max_items = 50000

    df_loc = pd.read_csv(
        args.input_dir / 'raw_POIs.txt',
        sep='\t',
        header=None,
        usecols=[0, 1, 2],
        names=['itemId', 'latitude', 'longitude']
    )

    df_clicks = pd.read_csv(
        clicks,
        sep='\t',
        header=None,
        usecols=[0, 1, 2],
        names=['userId', 'itemId', 'timestamp'],
    )
    df_clicks['timestamp'] = pd.to_datetime(
        df_clicks.timestamp, format='%a %b %d %H:%M:%S %z %Y', errors='coerce'
    )

    df_edges = pd.read_csv(edges, sep='\t', header=None, names=['follower', 'followee'])
    df_edges_rev = pd.DataFrame({
        'followee': df_edges.follower,
        'follower': df_edges.followee
    })
    df_edges = df_edges.append(df_edges_rev, ignore_index=True)
else:
    print(f'Unsupported dataset {args.dataset}', file=sys.stderr)
    sys.exit(1)

df_edges = df_edges[df_edges.follower != df_edges.followee]
df_clicks = df_clicks.dropna()
print('converting IDs to integers...')
df_clicks, df_edges = update_id(
    df_clicks, df_edges, colname='userId', alias=['followee', 'follower']
)
if df_loc is None:
    df_clicks = update_id(df_clicks, colname='itemId')
else:
    df_clicks, df_loc = update_id(df_clicks, df_loc, colname='itemId')
df_clicks = df_clicks.sort_values(['userId', 'timestamp'])
np.random.seed(123456)
preprocess(df_clicks, df_edges, df_loc, args)
