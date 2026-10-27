"""Офлайн-рекомендации: ALS, похожие треки и ранжирование кандидатов.

Повторные прослушивания в музыке полезны, поэтому уже известные треки
не исключаются из рекомендаций. ALS обучается на бинарных взаимодействиях.
Для большого каталога кандидаты ищутся приближённо в FAISS HNSW, а их
оценки затем пересчитываются скалярным произведением факторов ALS.
"""

import gc
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix, csr_matrix


SEED = 42
FEATURES = ["als_score", "item_popularity", "user_activity", "genre_count"]


@dataclass
class ALSData:
    model: object
    user_ids: np.ndarray
    item_ids: np.ndarray
    matrix: object


def _is_file(events):
    return isinstance(events, (str, Path))


def _sql_path(path):
    return "'" + Path(path).resolve().as_posix().replace("'", "''") + "'"


def _connect_events(path):
    import duckdb

    connection = duckdb.connect()
    temporary_dir = Path(path).resolve().parent / "tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    connection.execute("SET memory_limit = '3GB'")
    connection.execute("SET threads = 4")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET temp_directory = {}".format(_sql_path(temporary_dir)))
    return connection


def _event_count(events):
    return pq.read_metadata(events).num_rows if _is_file(events) else len(events)


def get_user_ids(events):
    """Отсортированные пользователи без загрузки всех событий в pandas."""
    if not _is_file(events):
        return np.sort(events["user_id"].unique())
    with _connect_events(events) as connection:
        return connection.execute(
            "SELECT DISTINCT user_id FROM read_parquet({}) ORDER BY user_id".format(_sql_path(events))
        ).fetchnumpy()["user_id"]


def _known_positions(ids, values):
    positions = np.searchsorted(ids, values)
    valid = positions < len(ids)
    if not valid.all() or not np.array_equal(ids[positions], values):
        raise ValueError("В событиях найден идентификатор вне переданного каталога.")
    return positions


def _csr_from_file(events, user_ids=None, item_ids=None, value=1.0):
    """Собираем CSR из отсортированных уникальных пар, без полного массива COO.

    DuckDB удаляет повторы на диске с ограничением памяти. В Python одновременно
    находятся только CSR, справочники идентификаторов и один пакет item_id.
    """
    work_dir = Path(events).resolve().parent / "tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pairs_", dir=work_dir) as temporary_dir:
        pairs_path = Path(temporary_dir) / "pairs.parquet"
        with _connect_events(events) as connection:
            source = "read_parquet({}) AS events".format(_sql_path(events))
            if user_ids is not None:
                connection.register("selected_users", pd.DataFrame({"user_id": user_ids}))
                source += " INNER JOIN selected_users USING (user_id)"
            print("Подготовка уникальных пар для CSR:", Path(events).name, flush=True)
            connection.execute("""
                COPY (
                    SELECT DISTINCT user_id, item_id FROM {}
                    ORDER BY user_id, item_id
                ) TO {} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
            """.format(source, _sql_path(pairs_path)))
            counts = connection.execute("""
                SELECT user_id, COUNT(*) AS pair_count
                FROM read_parquet({}) GROUP BY user_id ORDER BY user_id
            """.format(_sql_path(pairs_path))).fetchnumpy()
            if user_ids is None:
                user_ids = counts["user_id"]
            if item_ids is None:
                item_ids = connection.execute("""
                    SELECT DISTINCT item_id FROM read_parquet({}) ORDER BY item_id
                """.format(_sql_path(pairs_path))).fetchnumpy()["item_id"]

        pair_count = int(counts["pair_count"].sum())
        index_type = "int32" if max(pair_count, len(user_ids), len(item_ids)) < 2 ** 31 else "int64"
        row_counts = np.zeros(len(user_ids), dtype=index_type)
        row_positions = _known_positions(user_ids, counts["user_id"])
        row_counts[row_positions] = counts["pair_count"]
        indptr = np.empty(len(user_ids) + 1, dtype=index_type)
        indptr[0] = 0
        np.cumsum(row_counts, out=indptr[1:])
        del counts, row_positions, row_counts
        indices = np.empty(pair_count, dtype=index_type)
        data = np.full(pair_count, value, dtype="float32")
        offset = 0
        parquet_file = pq.ParquetFile(pairs_path)
        try:
            for batch in parquet_file.iter_batches(batch_size=250000, columns=["item_id"]):
                batch_ids = batch.column(0).to_numpy(zero_copy_only=False)
                stop = offset + len(batch_ids)
                indices[offset:stop] = _known_positions(item_ids, batch_ids)
                offset = stop
        finally:
            parquet_file.close()
        if offset != pair_count:
            raise RuntimeError("Количество уникальных пар изменилось при чтении CSR.")
        matrix = csr_matrix((data, indices, indptr), shape=(len(user_ids), len(item_ids)), copy=False)
        print("CSR готова: {} пользователей, {} треков, {} пар.".format(
            len(user_ids), len(item_ids), pair_count), flush=True)
    return matrix, user_ids, item_ids


def split_events(events, cutoff="2022-12-16"):
    """Граница относится к тесту: train строго раньше cutoff."""
    cutoff = pd.Timestamp(cutoff)
    if _is_file(events):
        events = Path(events)
        suffix = cutoff.strftime("%Y%m%d")
        train_events = events.with_name(events.stem + "_before_" + suffix + ".parquet")
        test_events = events.with_name(events.stem + "_from_" + suffix + ".parquet")
        with _connect_events(events) as connection:
            for path, operator in [(train_events, "<"), (test_events, ">=")]:
                connection.execute("""
                    COPY (SELECT * FROM read_parquet({}) WHERE started_at {} TIMESTAMP '{}')
                    TO {} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
                """.format(_sql_path(events), operator, cutoff.isoformat(), _sql_path(path)))
        if _event_count(train_events) == 0 or _event_count(test_events) == 0:
            raise ValueError("После временного разбиения одна из выборок пуста.")
        return train_events, test_events
    train_events = events.loc[events["started_at"] < cutoff].copy()
    test_events = events.loc[events["started_at"] >= cutoff].copy()
    if train_events.empty or test_events.empty:
        raise ValueError("После временного разбиения одна из выборок пуста.")
    return train_events, test_events


def fit_als(events, factors=32, iterations=15):
    """Фиксированная ALS; повторные пары имеют ту же уверенность, что и одна."""
    from implicit.als import AlternatingLeastSquares
    from threadpoolctl import threadpool_limits

    if _event_count(events) == 0:
        raise ValueError("Нельзя обучить ALS на пустой истории.")
    if _is_file(events):
        matrix, user_ids, item_ids = _csr_from_file(events, value=20.0)
    else:
        user_ids = np.sort(events["user_id"].unique())
        item_ids = np.sort(events["item_id"].unique())
        user_codes = np.searchsorted(user_ids, events["user_id"].to_numpy()).astype("int32")
        item_codes = np.searchsorted(item_ids, events["item_id"].to_numpy()).astype("int32")
        matrix = coo_matrix(
            (np.ones(len(events), dtype="float32"), (user_codes, item_codes)),
            shape=(len(user_ids), len(item_ids)),
        ).tocsr()
        del user_codes, item_codes
        matrix.data[:] = 20.0
    # ALS сам распараллеливает пользователей; вложенные потоки BLAS мешают ему.
    with threadpool_limits(limits=1, user_api="blas"):
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=0.1,
            iterations=iterations,
            random_state=SEED,
            num_threads=min(8, os.cpu_count() or 1),
        )
        model.fit(matrix, show_progress=True)
    return ALSData(model, user_ids, item_ids, matrix)


class FactorSearch:
    """HNSW с фиксированными параметрами; маленький пример ищется точно."""

    def __init__(self, factors, cosine=False):
        self.factors = np.ascontiguousarray(factors, dtype="float32")
        if cosine:
            self.factors = self.factors.copy()
            norms = np.linalg.norm(self.factors, axis=1, keepdims=True)
            self.factors /= np.maximum(norms, 1e-12)
        self.index = None
        if len(self.factors) > 1000:
            try:
                import faiss
            except ImportError as error:
                raise ImportError("Установите faiss-cpu==1.7.4 для поиска по каталогу.") from error
            faiss.omp_set_num_threads(min(8, os.cpu_count() or 1))
            self.index = faiss.IndexHNSWFlat(
                self.factors.shape[1], 32, faiss.METRIC_INNER_PRODUCT
            )
            self.index.hnsw.efConstruction = 80
            self.index.hnsw.efSearch = 64
            self.index.add(self.factors)

    def search(self, queries, k):
        queries = np.ascontiguousarray(queries, dtype="float32")
        k = min(k, len(self.factors))
        if self.index is not None:
            _, positions = self.index.search(queries, k)
        else:
            scores = queries @ self.factors.T
            positions = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        # Даже после приближённого поиска сохраняем исходные оценки модели.
        selected = self.factors[positions]
        scores = np.einsum("bf,bkf->bk", queries, selected)
        order = np.argsort(-scores, axis=1, kind="stable")
        positions = np.take_along_axis(positions, order, axis=1)
        scores = np.take_along_axis(scores, order, axis=1)
        return positions, scores


def _feature_arrays(items, als_data):
    popularity = als_data.matrix.getnnz(axis=0).astype("float32")
    activity = np.diff(als_data.matrix.indptr).astype("float32")
    genre_counts = items.set_index("item_id")["genres"].map(len)
    genre_counts = genre_counts.reindex(als_data.item_ids).fillna(0).to_numpy("float32")
    return popularity, activity, genre_counts


def _popular_items(als_data, limit=1000):
    popularity = als_data.matrix.getnnz(axis=0)
    # При равной популярности порядок определяется идентификатором трека.
    order = np.lexsort((als_data.item_ids, -popularity))[:limit]
    return order, pd.DataFrame({
        "item_id": als_data.item_ids[order],
        "score": popularity[order].astype("float32"),
        "rank": np.arange(1, len(order) + 1, dtype="int32"),
    })


def get_popular_items(train_events, limit=1000):
    """Популярность трека — число разных слушателей в обучающем периоде."""
    if _is_file(train_events):
        with _connect_events(train_events) as connection:
            popular = connection.execute("""
                SELECT item_id, COUNT(DISTINCT user_id) AS score
                FROM read_parquet({}) GROUP BY item_id
                ORDER BY score DESC, item_id LIMIT {}
            """.format(_sql_path(train_events), int(limit))).fetchdf()
    else:
        popular = train_events.groupby("item_id")["user_id"].nunique().rename("score").reset_index()
    popular.sort_values(["score", "item_id"], ascending=[False, True], inplace=True)
    popular = popular.head(limit).reset_index(drop=True)
    popular["score"] = popular["score"].astype("float32")
    popular["rank"] = np.arange(1, len(popular) + 1, dtype="int32")
    return popular


def _candidate_batch(positions, als_data, search, feature_arrays, popular_positions, n=100):
    """Кандидаты одного батча: ALS и 20 популярных, без повторов внутри пользователя."""
    popularity, activity, genre_counts = feature_arrays
    vectors = als_data.model.user_factors[positions]
    candidate_positions, _ = search.search(vectors, n)
    user_parts, item_parts, score_parts = [], [], []
    for user_position, candidates in zip(positions, candidate_positions):
        candidates = np.unique(np.concatenate([candidates, popular_positions[:20]]))
        scores = als_data.model.item_factors[candidates] @ als_data.model.user_factors[user_position]
        user_parts.append(np.full(len(candidates), user_position, dtype="int32"))
        item_parts.append(candidates.astype("int32"))
        score_parts.append(scores.astype("float32"))
    user_positions = np.concatenate(user_parts)
    item_positions = np.concatenate(item_parts)
    return pd.DataFrame({
        "user_id": als_data.user_ids[user_positions],
        "item_id": als_data.item_ids[item_positions],
        "als_score": np.concatenate(score_parts),
        "item_popularity": np.log1p(popularity[item_positions]),
        "user_activity": np.log1p(activity[user_positions]),
        "genre_count": genre_counts[item_positions],
    })


def train_ranker(items, train_events, output_dir=None, user_limit=30000, batch_size=256):
    """Учим ranker на внутреннем отложенном периоде 9–15 декабря.

    Внешний тест не передаётся в функцию. Детерминированная выборка пользователей
    ограничивает размер обучающей таблицы; ALS использует всю внутреннюю историю.
    """
    from catboost import CatBoostRanker, Pool

    if _is_file(train_events):
        with _connect_events(train_events) as connection:
            last_date = connection.execute(
                "SELECT MAX(started_at) FROM read_parquet({})".format(_sql_path(train_events))
            ).fetchone()[0]
    else:
        last_date = train_events["started_at"].max()
    if pd.Timestamp(last_date) >= pd.Timestamp("2022-12-16"):
        raise ValueError("В train_ranker попали события внешнего тестового периода.")
    cutoff = pd.Timestamp("2022-12-09")
    if _is_file(train_events):
        base_events, label_events = split_events(train_events, cutoff=cutoff)
    else:
        base_events = train_events.loc[train_events["started_at"] < cutoff]
        label_events = train_events.loc[train_events["started_at"] >= cutoff]
    if _event_count(base_events) == 0 or _event_count(label_events) == 0:
        raise ValueError("Для обучения ranker нужны события до и после 9 декабря 2022 года.")
    als_data = fit_als(base_events)
    eligible_users = np.intersect1d(als_data.user_ids, get_user_ids(label_events))
    if user_limit is not None and len(eligible_users) > user_limit:
        eligible_users = np.sort(np.random.default_rng(SEED).choice(
            eligible_users, user_limit, replace=False
        ))
    if len(eligible_users) == 0:
        raise ValueError("Во внутреннем периоде нет пользователей с обучающей историей.")
    if _is_file(label_events):
        with _connect_events(label_events) as connection:
            connection.register("selected_users", pd.DataFrame({"user_id": eligible_users}))
            selected_labels = connection.execute("""
                SELECT DISTINCT user_id, item_id FROM read_parquet({})
                INNER JOIN selected_users USING (user_id)
            """.format(_sql_path(label_events))).fetchdf()
    else:
        selected_labels = label_events.loc[
            label_events["user_id"].isin(eligible_users), ["user_id", "item_id"]
        ].drop_duplicates()
    truth = selected_labels.groupby("user_id")["item_id"].agg(set).to_dict()
    user_positions = pd.Index(als_data.user_ids).get_indexer(eligible_users)
    feature_arrays = _feature_arrays(items, als_data)
    popular_positions, _ = _popular_items(als_data)
    search = FactorSearch(als_data.model.item_factors)
    tables = []
    for start in range(0, len(user_positions), batch_size):
        table = _candidate_batch(
            user_positions[start:start + batch_size], als_data, search,
            feature_arrays, popular_positions,
        )
        table["target"] = [
            int(item_id in truth[user_id])
            for user_id, item_id in zip(table["user_id"], table["item_id"])
        ]
        # Группа без единого положительного кандидата не даёт обучающих пар.
        positive_counts = table.groupby("user_id")["target"].transform("sum")
        candidate_counts = table.groupby("user_id")["target"].transform("size")
        tables.append(table.loc[(positive_counts > 0) & (positive_counts < candidate_counts)])
    candidates = pd.concat(tables, ignore_index=True)
    if candidates.empty:
        raise ValueError("В кандидатах внутреннего периода нет групп для обучения ranker.")
    candidates.sort_values(["user_id", "item_id"], inplace=True)
    info = {
        "base_end_exclusive": "2022-12-09",
        "labels_end_exclusive": "2022-12-16",
        "sampled_users": int(len(eligible_users)),
        "training_users": int(candidates["user_id"].nunique()),
        "training_candidates": int(len(candidates)),
        "positive_candidates": int(candidates["target"].sum()),
        "features": FEATURES,
    }
    # Освобождаем внутреннюю ALS до обучения деревьев и финальной ALS.
    del search, als_data, feature_arrays, tables, selected_labels, truth, base_events, label_events
    gc.collect()
    ranker = CatBoostRanker(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function="YetiRank",
        random_seed=SEED,
        thread_count=min(8, os.cpu_count() or 1),
        allow_writing_files=False,
        verbose=25,
    )
    ranker.fit(Pool(
        candidates[FEATURES], label=candidates["target"], group_id=candidates["user_id"]
    ))
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ranker.save_model(str(output_dir / "ranker.cbm"))
        (output_dir / "ranker_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return ranker, info


def _rank_table(table, score_column, k):
    table = table.sort_values(
        ["user_id", score_column, "item_id"], ascending=[True, False, True]
    ).copy()
    table["rank"] = (table.groupby("user_id", sort=False).cumcount() + 1).astype("int32")
    table = table.loc[table["rank"] <= k, ["user_id", "item_id", score_column, "rank"]]
    return table.rename(columns={score_column: "score"})


def _write_table(writer, table, path):
    arrow_table = pa.Table.from_pandas(table, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(str(path), arrow_table.schema, compression="zstd")
    writer.write_table(arrow_table)
    return writer


def save_recommendations(items, train_events, als_data, ranker, output_dir,
                         all_user_ids=None, batch_size=512):
    """Сохраняем рекомендации батчами; холодным пользователям выдаём популярное.

    Файлы содержат 50 ALS-кандидатов и 20 итоговых рекомендаций на пользователя.
    Оценки холодных пользователей равны популярности трека. В similar сохраняются
    10 соседей для треков, которые встречались в обучении, без самого трека.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _event_count(train_events) == 0:
        raise ValueError("Обучающая история пуста.")
    if all_user_ids is None:
        all_user_ids = als_data.user_ids
    all_user_ids = np.sort(np.unique(all_user_ids))
    if len(all_user_ids) == 0:
        raise ValueError("Список пользователей для рекомендаций пуст.")
    popular_positions, popular = _popular_items(als_data)
    popular.to_parquet(output_dir / "top_popular.parquet", index=False, compression="zstd")
    feature_arrays = _feature_arrays(items, als_data)
    search = FactorSearch(als_data.model.item_factors)
    writers = {"personal_als": None, "recommendations": None}
    counts = {"top_popular": len(popular), "personal_als": 0, "recommendations": 0}
    known_user_index = pd.Index(als_data.user_ids)
    cold_users = 0
    try:
        for start in range(0, len(all_user_ids), batch_size):
            batch_users = all_user_ids[start:start + batch_size]
            positions = known_user_index.get_indexer(batch_users)
            known_positions = positions[positions >= 0]
            tables = {"personal_als": [], "recommendations": []}
            if len(known_positions):
                candidates = _candidate_batch(
                    known_positions, als_data, search, feature_arrays, popular_positions
                )
                tables["personal_als"].append(_rank_table(candidates, "als_score", 50))
                candidates["ranker_score"] = ranker.predict(
                    candidates[FEATURES], thread_count=min(8, os.cpu_count() or 1)
                ).astype("float32")
                tables["recommendations"].append(_rank_table(candidates, "ranker_score", 20))
            for user_id in batch_users[positions < 0]:
                cold_users += 1
                for name, k in [("personal_als", 50), ("recommendations", 20)]:
                    fallback = popular.head(k).copy()
                    fallback.insert(0, "user_id", user_id)
                    tables[name].append(fallback)
            for name in tables:
                table = pd.concat(tables[name], ignore_index=True).sort_values(["user_id", "rank"])
                # Типы батчей должны совпадать, включая батч только холодных пользователей.
                table["user_id"] = table["user_id"].astype(all_user_ids.dtype)
                table["item_id"] = table["item_id"].astype(als_data.item_ids.dtype)
                table["score"] = table["score"].astype("float32")
                writers[name] = _write_table(writers[name], table, output_dir / (name + ".parquet"))
                counts[name] += len(table)
            if start == 0 or (start // batch_size + 1) % 100 == 0:
                print("Рекомендации: {} из {} пользователей".format(
                    min(start + batch_size, len(all_user_ids)), len(all_user_ids)
                ))
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()
    del search
    gc.collect()
    similar_info = save_similar(als_data, output_dir / "similar.parquet", batch_size)
    counts["similar"] = similar_info["rows"]
    return {
        "users": int(len(all_user_ids)), "cold_users": cold_users,
        "rows": {name: int(count) for name, count in counts.items()},
        "candidate_search": "FAISS HNSW32, efConstruction=80, efSearch=64; для <=1000 треков точный поиск",
    }


def save_similar(als_data, path, batch_size=512, k=10):
    """Соседи ALS по косинусному сходству, исходный трек исключается."""
    search = FactorSearch(als_data.model.item_factors, cosine=True)
    writer = None
    rows = 0
    try:
        for start in range(0, len(als_data.item_ids), batch_size):
            stop = min(start + batch_size, len(als_data.item_ids))
            positions, scores = search.search(search.factors[start:stop], k + 1)
            parts = []
            for source, neighbors, neighbor_scores in zip(range(start, stop), positions, scores):
                valid = neighbors != source
                neighbors, neighbor_scores = neighbors[valid][:k], neighbor_scores[valid][:k]
                parts.append(pd.DataFrame({
                    "item_id_1": np.repeat(als_data.item_ids[source], len(neighbors)),
                    "item_id_2": als_data.item_ids[neighbors],
                    "score": neighbor_scores.astype("float32"),
                    "rank": np.arange(1, len(neighbors) + 1, dtype="int32"),
                }))
            table = pd.concat(parts, ignore_index=True)
            writer = _write_table(writer, table, path)
            rows += len(table)
            if start == 0 or (start // batch_size + 1) % 200 == 0:
                print("Похожие треки: {} из {}".format(stop, len(als_data.item_ids)))
    finally:
        if writer is not None:
            writer.close()
    return {"rows": rows}


def _truth_matrix(events, user_ids, item_ids):
    if _is_file(events):
        matrix, _, _ = _csr_from_file(events, user_ids=user_ids, item_ids=item_ids)
        return matrix
    user_positions = pd.Index(user_ids).get_indexer(events["user_id"])
    valid = user_positions >= 0
    item_positions = _known_positions(item_ids, events.loc[valid, "item_id"].to_numpy())
    matrix = coo_matrix((
        np.ones(valid.sum(), dtype="float32"),
        (user_positions[valid].astype("int32"), item_positions.astype("int32")),
    ), shape=(len(user_ids), len(item_ids))).tocsr()
    matrix.data[:] = 1
    return matrix


def evaluate_recommendations(items, train_events, test_events, output_dir,
                             k=20, user_limit=None, als_data=None):
    """Одинаковые test-пользователи и top-k для трёх методов.

    Recall и precision усредняются по пользователям; precision делится на k,
    даже если список короче. Coverage — доля каталога в рекомендациях этой
    тестовой группы. Novelty — средняя доля рекомендаций, которых не было
    в истории пользователя до 16 декабря. Цели и история считаются как множества.
    """
    output_dir = Path(output_dir)
    user_ids = get_user_ids(test_events)
    if user_limit is not None and len(user_ids) > user_limit:
        user_ids = np.sort(np.random.default_rng(SEED).choice(user_ids, user_limit, replace=False))
    if len(user_ids) == 0:
        raise ValueError("Нет пользователей для оценки.")
    item_ids = np.sort(items["item_id"].unique())
    truth_matrix = _truth_matrix(test_events, user_ids, item_ids)
    history_matrix = None
    if als_data is None:
        history_matrix = _truth_matrix(train_events, user_ids, item_ids)
        history_positions = np.arange(len(user_ids))
    else:
        history_positions = pd.Index(als_data.user_ids).get_indexer(user_ids)
    truth_counts = np.diff(truth_matrix.indptr)
    user_index = pd.Index(user_ids)
    popular_ids = pd.read_parquet(output_dir / "top_popular.parquet")["item_id"].head(k).to_numpy()
    results = []

    for method, filename in [
        ("Популярные", "top_popular.parquet"),
        ("ALS", "personal_als.parquet"),
        ("Ранжирование", "recommendations.parquet"),
    ]:
        recalls = np.zeros(len(user_ids), dtype="float64")
        precisions = np.zeros(len(user_ids), dtype="float64")
        novelties = np.zeros(len(user_ids), dtype="float64")
        covered = np.zeros(len(item_ids), dtype=bool)
        evaluated = np.zeros(len(user_ids), dtype=bool)

        def add_user(position, recommended_ids):
            recommended_ids = np.asarray(recommended_ids)[:k]
            columns = _known_positions(item_ids, recommended_ids)
            target = truth_matrix.indices[truth_matrix.indptr[position]:truth_matrix.indptr[position + 1]]
            hits = np.isin(columns, target).sum()
            recalls[position] = hits / truth_counts[position]
            precisions[position] = hits / k
            covered[columns] = True
            history_position = history_positions[position]
            if history_position < 0:
                known_ids = np.array([], dtype=item_ids.dtype)
            elif als_data is not None:
                row = als_data.matrix
                known_columns = row.indices[row.indptr[history_position]:row.indptr[history_position + 1]]
                known_ids = als_data.item_ids[known_columns]
            else:
                known_columns = history_matrix.indices[
                    history_matrix.indptr[position]:history_matrix.indptr[position + 1]
                ]
                known_ids = item_ids[known_columns]
            if len(recommended_ids):
                novelties[position] = (~np.isin(recommended_ids, known_ids)).mean()
            evaluated[position] = True

        if method != "Популярные":
            parquet_file = pq.ParquetFile(output_dir / filename)
            # Каждая записанная строковая группа содержит целые списки пользователей.
            for row_group in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group).to_pandas()
                evaluation_positions = user_index.get_indexer(table["user_id"])
                table = table.loc[(table["rank"] <= k) & (evaluation_positions >= 0)]
                for user_id, group in table.groupby("user_id", sort=False):
                    position = user_index.get_loc(user_id)
                    add_user(position, group.sort_values("rank")["item_id"].to_numpy())
        # Также явно включаем холодных пользователей, которых нет в офлайн-файле.
        for position in np.flatnonzero(~evaluated):
            add_user(position, popular_ids)
        results.append({
            "method": method,
            "recall": recalls.mean(),
            "precision": precisions.mean(),
            "coverage": covered.mean(),
            "novelty": novelties.mean(),
            "users": len(user_ids),
            "k": k,
        })
    return pd.DataFrame(results).set_index("method")
