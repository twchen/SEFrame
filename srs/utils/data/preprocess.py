import numpy as np


def group_sessions(df, interval):
    df_prev = df.shift()
    is_new_session = (df.userId !=
                      df_prev.userId) | (df.timestamp - df_prev.timestamp > interval)
    sessionId = is_new_session.cumsum() - 1
    df = df.assign(sessionId=sessionId)
    return df


def filter_short_sessions(df, min_len=2):
    session_len = df.groupby('sessionId', sort=False).size()
    long_sessions = session_len[session_len >= min_len].index
    df_long = df[df.sessionId.isin(long_sessions)]
    print(
        f'removed {len(session_len) - len(long_sessions)}/{len(session_len)} sessions shorter than {min_len}'
    )
    return df_long


def filter_infreq_items(df, min_support=5):
    item_support = df.groupby('itemId', sort=False).size()
    freq_items = item_support[item_support >= min_support].index
    df_freq = df[df.itemId.isin(freq_items)]
    print(
        f'removed {len(item_support) - len(freq_items)}/{len(item_support)} items with supprot < {min_support}'
    )
    return df_freq


def filter_isolated_users(df_clicks, df_edges):
    num_sessions = df_clicks.sessionId.nunique()
    num_edges = len(df_edges)
    while True:
        sess_users = df_clicks.userId.unique()
        soci_users = np.unique(df_edges[['follower', 'followee']].values)
        # users must be followed by or follow at least one user
        df_clicks_new = df_clicks[df_clicks.userId.isin(soci_users)]
        # users must have at least one session
        df_edges_new = df_edges[df_edges.follower.isin(sess_users)
                                & df_edges.followee.isin(sess_users)]
        if len(df_clicks_new) == len(df_clicks) and len(df_edges) == len(df_edges_new):
            break
        df_clicks = df_clicks_new
        df_edges = df_edges_new
    print(
        f'removed {num_sessions - df_clicks.sessionId.nunique()}/{num_sessions}'
        f' sessions and {num_edges - len(df_edges)}/{num_edges} edges of isolated users'
    )
    return df_clicks, df_edges


def filter_loop(df_clicks, df_edges, args):
    while True:
        df_long = filter_short_sessions(df_clicks)
        df_freq = filter_infreq_items(df_long)
        df_conn, df_edges = filter_isolated_users(df_freq, df_edges)
        if len(df_conn) == len(df_clicks):
            break
        df_clicks = df_conn
    return df_clicks, df_edges


def truncate_long_sessions(df, max_len, is_sorted=False):
    if not is_sorted:
        df = df.sort_values(['sessionId', 'timestamp'])
    itemIdx = df.groupby('sessionId').cumcount()
    df_t = df[itemIdx < max_len]
    print(
        f'removed {len(df) - len(df_t)}/{len(df)} clicks in sessions longer than {max_len}'
    )
    return df_t


def update_id(*dataframes, colnames, mapping=None):
    """
    Map the values in the columns `colnames` of `dataframes` according to `mapping`.
    If `mapping` is `None`, a dictionary that maps the values in column `colnames[0]`
    of `dataframes[0]` to unique integers will be used.
    Note that values not appear in `mapping` will be mapped to `NaN`.

    Args
    ----
    dataframes : list[DataFrame]
        A list of dataframes.
    colnames: str, list[str]
        The names of columns.
    mapping: function, dict, optional
        Mapping correspondence.

    Returns
    -------
    DataFrame, list[DataFrame]
        A dataframe (if there is only one input dataframe) or a list of dataframes
        with columns in `colnames` updated according to `mapping`.
    """
    if type(colnames) is str:
        colnames = [colnames]
    if mapping is None:
        uniques = dataframes[0][colnames[0]].unique()
        mapping = {oid: i for i, oid in enumerate(uniques)}
    results = []
    for df in dataframes:
        columns = {}
        for name in colnames:
            if name in df.columns:
                columns[name] = df[name].map(mapping)
        df = df.assign(**columns)
        results.append(df)
    if len(results) == 1:
        return results[0]
    else:
        return results


def remove_immediate_repeats(df):
    df_prev = df.shift()
    is_not_repeat = (df.sessionId != df_prev.sessionId) | (df.itemId != df_prev.itemId)
    df_no_repeat = df[is_not_repeat]
    print(
        f'removed {len(df) - len(df_no_repeat)}/{len(df)} immediate repeat consumptions'
    )
    return df_no_repeat


def reorder_sessions_by_endtime(df):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    df_endtime = endtime.sort_values().reset_index()
    oid2nid = dict(zip(df_endtime.sessionId, df_endtime.index))
    sessionId_new = df.sessionId.map(oid2nid)
    df = df.assign(sessionId=sessionId_new)
    df = df.sort_values(['sessionId', 'timestamp'])
    return df


def train_test_split(df, test_split=0.2):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    endtime = endtime.sort_values()
    num_tests = int(len(endtime) * test_split)
    test_session_ids = endtime.index[-num_tests:]
    df_train = df[~df.sessionId.isin(test_session_ids)]
    df_test = df[df.sessionId.isin(test_session_ids)]
    return df_train, df_test


def save_sessions(df_clicks, filepath):
    df_clicks = df_clicks.groupby('sessionId').agg({
        'userId':
        lambda col: col.iloc[0],
        'itemId':
        lambda col: ','.join(col.astype(str)),
    })
    df_clicks.to_csv(filepath, sep='\t', header=['userId', 'items'], index=True)


def keep_valid_sessions(df_train, df_test, train_split):
    print('\nprocessing test sets...')
    uid = df_train.userId.unique()
    iid = df_train.itemId.unique()
    df_test = df_test[df_test.userId.isin(uid) & df_test.itemId.isin(iid)]
    df_test = filter_short_sessions(df_test)
    return df_test


def keep_top_n(df_clicks, n, colname):
    print(f'keeping top {n} most frequent values in column {colname}')
    supports = df_clicks.groupby(colname, sort=False).size()
    top_values = supports.nlargest(n).index
    df_top = df_clicks[df_clicks[colname].isin(top_values)]
    print(f'removed {len(supports) - len(top_values)}/{len(supports)} values')
    return df_top


def update_session_id(df_train, df_test):
    df_train = reorder_sessions_by_endtime(df_train)
    df_test = reorder_sessions_by_endtime(df_test)
    num_train_sessions = df_train.sessionId.max() + 1
    df_test = df_test.assign(sessionId=df_test.sessionId + num_train_sessions)
    return df_train, df_test


def print_stats(df_clicks, name):
    print(
        f'{name}:\n'
        f'No. of clicks: {len(df_clicks)}\n'
        f'No. of sessions: {df_clicks.sessionId.nunique()}\n'
        f'No. of users: {df_clicks.userId.nunique()}\n'
        f'No. of items: {df_clicks.itemId.nunique()}\n'
        f'Avg. session length: {len(df_clicks) / df_clicks.sessionId.nunique():.3f}\n'
    )


def save_dataset(df_train, df_test, df_edges, df_loc, args):
    df_test = keep_valid_sessions(df_train, df_test, args.train_split)

    print(f'No. of Clicks: {len(df_train) + len(df_test)}')
    print_stats(df_train, 'Training set')
    print_stats(df_test, 'Test set')
    print(f'No. of Connections: {len(df_edges)}')
    print(f'No. of Followers: {df_edges.follower.nunique()}')
    print(f'No. of Followees: {df_edges.followee.nunique()}')
    num_users = df_train.userId.nunique()
    print(f'Avg. Followers: {len(df_edges) / num_users:.3f}')

    df_train, df_test = update_session_id(df_train, df_test)

    # update userId
    df_train, df_test, df_edges = update_id(
        df_train, df_test, df_edges, colnames=['userId', 'followee', 'follower']
    )

    # update itemId
    if df_loc is None:
        df_train, df_test = update_id(df_train, df_test, colnames='itemId')
    else:
        df_loc = df_loc[df_loc.itemId.isin(df_train.itemId.unique())]
        df_train, df_test, df_loc = update_id(
            df_train, df_test, df_loc, colnames='itemId'
        )
        df_loc = df_loc.sort_values('itemId')

    dataset_dir = args.output_dir / args.dataset
    print(f'saving dataset to {dataset_dir}')
    # save sessions
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_sessions(df_train, dataset_dir / 'train.txt')
    # randomly and evenly split df_test into df_valid and df_test
    valid_test_sids = df_test.sessionId.unique()
    num_valid_sessions = len(valid_test_sids) // 2
    valid_sids = np.random.choice(valid_test_sids, num_valid_sessions, replace=False)
    df_valid = df_test[df_test.sessionId.isin(valid_sids)]
    df_test = df_test[~df_test.sessionId.isin(valid_sids)]
    save_sessions(df_valid, dataset_dir / 'valid.txt')
    save_sessions(df_test, dataset_dir / 'test.txt')

    if df_loc is not None:
        df_loc.to_csv(
            dataset_dir / 'loc.txt',
            sep='\t',
            index=False,
            header=True,
            float_format='%.2f'
        )

    # save social network
    df_edges = df_edges.sort_values(['followee', 'follower'])
    df_edges.to_csv(dataset_dir / 'edges.txt', sep='\t', header=True, index=False)

    # save stats
    num_users = df_train.userId.nunique()
    num_items = df_train.itemId.nunique()
    with open(dataset_dir / 'stats.txt', 'w') as f:
        f.write('num_users\tnum_items\tmax_len\n')
        f.write(f'{num_users}\t{num_items}\t{args.max_len}')


def preprocess(df_clicks, df_edges, df_loc, args):
    print('arguments: ', args)
    if 'sessionId' in df_clicks.columns:
        print('clicks are already grouped into sessions')
        df_clicks = df_clicks.sort_values(['userId', 'sessionId', 'timestamp'])
        sessionId = df_clicks.userId.astype(str) + '_' + df_clicks.sessionId.astype(str)
        df_clicks = df_clicks.assign(sessionId=sessionId)
        df_clicks = update_id(df_clicks, colnames='sessionId')
    else:
        df_clicks = group_sessions(df_clicks, args.interval)
    df_clicks = remove_immediate_repeats(df_clicks)
    if args.max_len > 0:
        df_clicks = truncate_long_sessions(df_clicks, args.max_len, is_sorted=True)
    if 'max_users' in args:
        df_clicks = keep_top_n(df_clicks, args.max_users, 'userId')
    if 'max_items' in args:
        df_clicks = keep_top_n(df_clicks, args.max_items, 'itemId')
    df_train, df_test = train_test_split(df_clicks, test_split=1 - args.train_split)
    df_train, df_edges = filter_loop(df_train, df_edges, args)
    save_dataset(df_train, df_test, df_edges, df_loc, args)
