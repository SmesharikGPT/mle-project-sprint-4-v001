"""Подготовка и первичный анализ музыкальных данных."""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def estimate_memory(data_dir):
    metadata = pq.ParquetFile(Path(data_dir) / "interactions.parquet")
    row_bytes = 0
    for field in metadata.schema_arrow:
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            row_bytes += field.type.bit_width // 8
        elif pa.types.is_timestamp(field.type):
            row_bytes += 8
        else:
            row_bytes += 40
    # Во время преобразований одновременно могут существовать несколько массивов.
    estimate_gb = metadata.metadata.num_rows * row_bytes * 2.5 / 1024 ** 3
    print("Приблизительная оценка памяти для обработки interactions: {:.1f} ГиБ.".format(estimate_gb))
    print("Это оценка: фактический расход зависит от типов и промежуточных таблиц.")
    return estimate_gb


def load_raw_data(data_dir="data/raw"):
    data_dir = Path(data_dir)
    estimate_memory(data_dir)
    tracks = pd.read_parquet(data_dir / "tracks.parquet")
    catalog_names = pd.read_parquet(data_dir / "catalog_names.parquet")
    interactions = pd.read_parquet(data_dir / "interactions.parquet")
    return tracks, catalog_names, interactions


def _sql_path(path):
    return "'" + str(Path(path).resolve()).replace("'", "''") + "'"


def _connect_events(path):
    connection = duckdb.connect()
    temporary_dir = Path(path).parent / "tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '3GB'")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET TimeZone = 'UTC'")
    connection.execute("SET temp_directory = " + _sql_path(temporary_dir))
    return connection


def event_summary(events_path):
    """Небольшая сводка; сами события в pandas не загружаются."""
    with _connect_events(events_path) as connection:
        return connection.execute("""
            SELECT COUNT(*) AS rows, COUNT(DISTINCT user_id) AS users,
                   MIN(started_at) AS min_date, MAX(started_at) AS max_date
            FROM read_parquet({})
        """.format(_sql_path(events_path))).df()


def prepare_data_from_files(raw_dir="data/raw", prepared_dir="data/prepared"):
    """Готовит полный набор событий средствами DuckDB с выгрузкой на диск."""
    raw_dir, prepared_dir = Path(raw_dir), Path(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    events_path = prepared_dir / "events.parquet"
    source_path = raw_dir / "interactions.parquet"
    with _connect_events(events_path) as connection:
        connection.execute("CREATE TEMP VIEW raw_events AS SELECT * FROM read_parquet({})".format(_sql_path(source_path)))
        # Равенство с исходным числом не даёт округлить дробный ID при CAST.
        connection.execute("""
            CREATE TEMP VIEW checked_events AS
            SELECT TRY_CAST(user_id AS INTEGER) AS user_id,
                   TRY_CAST(track_id AS INTEGER) AS item_id,
                   TRY_CAST(track_seq AS SMALLINT) AS track_seq,
                   TRY_CAST(started_at AS TIMESTAMP_NS) AS started_at,
                   COALESCE(TRY_CAST(user_id AS DOUBLE) = TRY_CAST(user_id AS INTEGER), FALSE) AS valid_user,
                   COALESCE(TRY_CAST(track_id AS DOUBLE) = TRY_CAST(track_id AS INTEGER), FALSE) AS valid_item,
                   COALESCE(TRY_CAST(track_seq AS DOUBLE) = TRY_CAST(track_seq AS SMALLINT)
                            AND TRY_CAST(track_seq AS SMALLINT) >= 0, FALSE) AS valid_sequence
            FROM raw_events
        """)
        report = connection.execute("""
            SELECT COUNT(*) AS raw_rows,
                   COUNT(*) FILTER (WHERE NOT valid_user) AS invalid_user_ids,
                   COUNT(*) FILTER (WHERE NOT valid_item) AS invalid_item_ids,
                   COUNT(*) FILTER (WHERE NOT valid_sequence) AS invalid_track_seq,
                   COUNT(*) FILTER (WHERE started_at IS NULL) AS invalid_dates,
                   COUNT(*) FILTER (WHERE NOT (valid_user AND valid_item AND valid_sequence)
                                         OR started_at IS NULL) AS invalid_required_rows_removed
            FROM checked_events
        """).df().iloc[0].map(int).to_dict()
        # Удаляем только совпадения всех четырёх полей. Другие прослушивания пары остаются.
        connection.execute("""
            COPY (
                SELECT DISTINCT user_id, item_id, track_seq, started_at
                FROM checked_events
                WHERE valid_user AND valid_item AND valid_sequence AND started_at IS NOT NULL
            ) TO {} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)
        """.format(_sql_path(events_path)))
        rows = pq.ParquetFile(events_path).metadata.num_rows
        report["rows"] = rows
        report["exact_duplicates_removed"] = report["raw_rows"] - report["invalid_required_rows_removed"] - rows
        report["repeat_listens_kept"] = int(connection.execute("""
            SELECT COALESCE(SUM(listens - 1), 0) FROM (
                SELECT COUNT(*) AS listens FROM read_parquet({}) GROUP BY user_id, item_id
            )
        """.format(_sql_path(events_path))).fetchone()[0])
        event_item_ids = connection.execute("SELECT DISTINCT item_id FROM read_parquet({})".format(_sql_path(events_path))).fetchnumpy()["item_id"]
        report["unique_items"] = len(event_item_ids)

    tracks = pd.read_parquet(raw_dir / "tracks.parquet")
    catalog_names = pd.read_parquet(raw_dir / "catalog_names.parquet")
    track_ids, valid_ids = _id_values(tracks["track_id"])
    missing_items = np.setdiff1d(event_item_ids, track_ids.loc[valid_ids].to_numpy())
    if len(missing_items):
        added_tracks = pd.DataFrame({"track_id": missing_items})
        for field in ["artists", "albums", "genres"]:
            added_tracks[field] = [[] for _ in range(len(added_tracks))]
        tracks = pd.concat([tracks, added_tracks], ignore_index=True)
    empty_events = pd.DataFrame({
        "user_id": pd.Series(dtype="int32"), "track_id": pd.Series(dtype="int32"),
        "track_seq": pd.Series(dtype="int16"), "started_at": pd.Series(dtype="datetime64[ns]"),
    })
    items, _, quality_report = prepare_data(tracks, catalog_names, empty_events)
    quality_report["items"]["missing_tracks_added"] = len(missing_items)
    quality_report["events"] = report
    items.to_parquet(prepared_dir / "items.parquet", index=False)
    return items, events_path, quality_report


def _id_values(values):
    numbers = pd.to_numeric(values, errors="coerce")
    valid = numbers.notna()
    if not pd.api.types.is_integer_dtype(numbers.dtype):
        valid &= np.isfinite(numbers) & (numbers % 1 == 0)
    return numbers, valid


def _compact_ids(values):
    if len(values) == 0:
        return values.astype("int64")
    limits = np.iinfo(np.int32)
    wide_limits = np.iinfo(np.int64)
    if values.min() < wide_limits.min or values.max() > wide_limits.max:
        raise ValueError("Идентификатор не помещается в целый тип int64.")
    dtype = "int32" if values.min() >= limits.min and values.max() <= limits.max else "int64"
    return values.astype(dtype)


def _id_list(value):
    if not isinstance(value, (list, tuple, np.ndarray)):
        return []
    result = []
    for item in value:
        if pd.isna(item) or isinstance(item, (bool, np.bool_)):
            continue
        try:
            number = int(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if isinstance(item, (float, np.floating)) and number != item:
            continue
        if number not in result:
            result.append(number)
    return result


def prepare_data(tracks, catalog_names, interactions):
    report = {"items": {}, "events": {}, "catalog": {}, "attributes": {}}
    items = tracks.rename(columns={"track_id": "item_id"}, copy=False)
    item_ids, valid_items = _id_values(items["item_id"])
    report["items"]["invalid_ids_removed"] = int((~valid_items).sum())
    if not valid_items.all():
        items = items.loc[valid_items].copy()
    items["item_id"] = _compact_ids(item_ids.loc[valid_items])

    fields = ["artists", "albums", "genres"]
    for field in fields:
        items[field] = items[field].map(_id_list)

    duplicate_items = items[items.duplicated("item_id", keep=False)]
    report["items"]["duplicate_ids_merged"] = int(items.duplicated("item_id").sum())
    for item_id, group in duplicate_items.groupby("item_id", sort=False):
        first_index = group.index[0]
        for field in fields:
            items.at[first_index, field] = list(dict.fromkeys(item for values in group[field] for item in values))
    if len(duplicate_items):
        items = items.drop_duplicates("item_id").reset_index(drop=True)

    event_columns = ["user_id", "item_id", "track_seq", "started_at"]
    events = interactions.rename(columns={"track_id": "item_id"}, copy=False)
    extra_columns = [column for column in events.columns if column not in event_columns]
    if extra_columns:
        events = events.drop(columns=extra_columns)
    report["events"]["raw_rows"] = len(events)
    valid_events = pd.Series(True, index=events.index)
    for column in ["user_id", "item_id", "track_seq"]:
        values, valid = _id_values(events[column])
        events[column] = values
        valid_events &= valid
    events["started_at"] = pd.to_datetime(events["started_at"], errors="coerce", utc=True).dt.tz_convert(None)
    valid_events &= events["started_at"].notna() & (events["track_seq"] >= 0)
    report["events"]["invalid_required_rows_removed"] = int((~valid_events).sum())
    if not valid_events.all():
        events = events.loc[valid_events].copy()
    for column in ["user_id", "item_id", "track_seq"]:
        events[column] = _compact_ids(events[column])
    duplicates = events.duplicated(event_columns)
    report["events"]["exact_duplicates_removed"] = int(duplicates.sum())
    if duplicates.any():
        events = events.loc[~duplicates].reset_index(drop=True)
    report["events"]["repeat_listens_kept"] = int(events.duplicated(["user_id", "item_id"]).sum())
    report["events"]["rows"] = len(events)

    missing_items = np.setdiff1d(events["item_id"].unique(), items["item_id"].to_numpy())
    report["items"]["missing_tracks_added"] = len(missing_items)
    if len(missing_items):
        added_items = pd.DataFrame({"item_id": missing_items})
        for field in fields:
            added_items[field] = [[] for _ in range(len(added_items))]
        items = pd.concat([items, added_items], ignore_index=True)
    items["item_id"] = _compact_ids(items["item_id"])

    catalog = catalog_names[["id", "type", "name"]].copy()
    catalog_ids, valid_catalog = _id_values(catalog["id"])
    report["catalog"]["invalid_ids_removed"] = int((~valid_catalog).sum())
    catalog = catalog.loc[valid_catalog].copy()
    catalog["id"] = _compact_ids(catalog_ids.loc[valid_catalog])
    aliases = {"tracks": "track", "artists": "artist", "albums": "album", "genres": "genre"}
    catalog["type"] = catalog["type"].astype(str).str.strip().str.lower().replace(aliases)
    report["catalog"]["duplicate_type_ids"] = int(catalog.duplicated(["type", "id"]).sum())
    known_names = catalog["name"].notna() & catalog["name"].astype(str).str.strip().ne("")
    # Сначала убираем пустое имя, иначе оно может скрыть корректную запись того же ID.
    catalog = catalog.loc[known_names]
    catalog = catalog.drop_duplicates(["type", "id"])
    lookups = {}
    for kind in ["track", "artist", "album", "genre"]:
        selected = catalog.loc[catalog["type"] == kind, ["id", "name"]]
        lookups[kind] = selected.set_index("id")["name"].to_dict()

    items["name"] = items["item_id"].map(lookups["track"])
    report["items"]["unknown_names"] = int(items["name"].isna().sum())
    missing_names = items["name"].isna()
    items.loc[missing_names, "name"] = items.loc[missing_names, "item_id"].map(lambda value: "Неизвестный трек {}".format(value))
    for field, kind, label in [("artists", "artist", "исполнитель"), ("albums", "album", "альбом"), ("genres", "genre", "жанр")]:
        lookup = lookups[kind]
        unknown = items[field].map(lambda values: any(value not in lookup for value in values))
        empty = items[field].map(len).eq(0)
        report["attributes"][field] = {"empty_tracks": int(empty.sum()), "unknown_id_tracks": int(unknown.sum())}
        items[field + "_missing"] = empty
        items[field + "_unknown"] = unknown
        items[kind + "_names"] = items[field].map(lambda values: [lookup.get(value, "Неизвестный {} {}".format(label, value)) for value in values])

    report["items"]["rows"] = len(items)
    return items.reset_index(drop=True), events, report


def get_eda_tables(items, events):
    if isinstance(events, pd.DataFrame):
        user_activity = events.groupby("user_id").size().rename("listens").to_frame()
        item_listens = events.groupby("item_id").size().rename("listen_count")
    else:
        with _connect_events(events) as connection:
            user_activity = connection.execute("""
                SELECT user_id, COUNT(*) AS listens FROM read_parquet({}) GROUP BY user_id
            """.format(_sql_path(events))).df().set_index("user_id")
            item_listens = connection.execute("""
                SELECT item_id, COUNT(*) AS listen_count FROM read_parquet({}) GROUP BY item_id
            """.format(_sql_path(events))).df().set_index("item_id")["listen_count"]
    top_counts = item_listens.nlargest(20).reset_index()
    top_tracks = top_counts.merge(items[["item_id", "name", "artist_names"]], on="item_id", how="left")

    item_genres = items[["item_id", "genres", "genre_names"]].explode(["genres", "genre_names"]).dropna(subset=["genres"])
    item_genres["listen_count"] = item_genres["item_id"].map(item_listens).fillna(0)
    top_genres = item_genres.groupby("genres").agg(
        name=("genre_names", "first"), listen_count=("listen_count", "sum"), track_count=("item_id", "size")
    ).sort_values("listen_count", ascending=False).reset_index().rename(columns={"genres": "genre_id"})
    unheard_tracks = items.loc[~items["item_id"].isin(item_listens.index)]
    track_coverage = pd.DataFrame({
        "group": ["Есть прослушивания", "Нет прослушиваний"],
        "tracks": [len(items) - len(unheard_tracks), len(unheard_tracks)],
    })
    return {"user_activity": user_activity, "top_tracks": top_tracks, "top_genres": top_genres,
            "unheard_tracks": unheard_tracks, "track_coverage": track_coverage}


def plot_user_activity(user_activity):
    limit = user_activity["listens"].quantile(0.99)
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.hist(user_activity["listens"], bins=50, range=(0, max(1, limit)), color="#4C78A8", rwidth=0.98)
    axis.set(title="Количество прослушиваний на пользователя", xlabel="Прослушивания (до 99-го процентиля)", ylabel="Пользователи")
    figure.tight_layout()
    return figure


def plot_popular_tracks(top_tracks):
    selected = top_tracks.head(10).iloc[::-1]
    figure, axis = plt.subplots(figsize=(10, 5))
    labels = selected["name"].map(lambda name: str(name) if len(str(name)) <= 48 else str(name)[:45] + "…")
    positions = np.arange(len(selected))
    axis.barh(positions, selected["listen_count"], color="#4C78A8")
    axis.set_yticks(positions, labels=labels)
    axis.set(title="Десять самых прослушиваемых треков", xlabel="Прослушивания")
    figure.tight_layout()
    return figure


def plot_popular_genres(top_genres):
    selected = top_genres.head(10).iloc[::-1]
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(selected))
    axis.barh(positions, selected["listen_count"], color="#4C78A8")
    axis.set_yticks(positions, labels=selected["name"])
    axis.set(title="Популярные жанры по числу прослушиваний", xlabel="Прослушивания треков жанра")
    figure.tight_layout()
    return figure


def plot_unheard_tracks(track_coverage):
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(track_coverage["group"], track_coverage["tracks"], color=["#4C78A8", "#8A8A8A"])
    axis.set(title="Прослушанные и непрослушанные треки", ylabel="Треки каталога")
    figure.tight_layout()
    return figure


def plot_eda(eda_tables):
    return [plot_user_activity(eda_tables["user_activity"]), plot_popular_tracks(eda_tables["top_tracks"]),
            plot_popular_genres(eda_tables["top_genres"]), plot_unheard_tracks(eda_tables["track_coverage"])]
