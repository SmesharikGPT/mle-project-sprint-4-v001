import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import duckdb
from fastapi import FastAPI, HTTPException, Query


logger = logging.getLogger("uvicorn.error")


class RecommendationStore:
    def __init__(self, data_directory):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect()
        self.connection.execute("SET memory_limit = '1GB'")
        self.connection.execute("SET threads = 2")
        self.lock = Lock()
        self.user_events = {}
        self.request_counts = {
            "personal": 0,
            "popular": 0,
            "online": 0,
            "similar": 0,
            "events": 0,
            "errors": 0,
        }

    def prepare(self):
        files = {
            "recommendations.parquet": {"user_id", "item_id", "rank"},
            "top_popular.parquet": {"item_id", "rank"},
            "similar.parquet": {"item_id_1", "item_id_2", "rank"},
        }
        missing_files = [name for name in files if not (self.data_directory / name).exists()]
        if missing_files:
            from storage import download_file, ensure_storage

            ensure_storage()
            prefix = os.getenv("S3_RECOMMENDATIONS_PREFIX", "recsys/recommendations").strip("/")
            for name in missing_files:
                logger.info("Загрузка файла %s", name)
                download_file(f"{prefix}/{name}", self.data_directory / name)

        # Проверяем только схему файлов, не загружая все рекомендации в память.
        for name, required_columns in files.items():
            columns = self.connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(self.data_directory / name)]
            ).fetchall()
            actual_columns = {row[0] for row in columns}
            if not required_columns.issubset(actual_columns):
                raise ValueError(f"В файле {name} отсутствуют обязательные столбцы")

    def add_event(self, user_id, item_id):
        with self.lock:
            events = self.user_events.get(user_id, [])
            # Повторное прослушивание перемещает трек в начало истории.
            self.user_events[user_id] = [item_id] + [
                previous_item for previous_item in events if previous_item != item_id
            ][:19]
            self.request_counts["events"] += 1

    def get_events(self, user_id):
        with self.lock:
            return list(self.user_events.get(user_id, []))

    def delete_events(self, user_id):
        with self.lock:
            self.user_events.pop(user_id, None)

    def _personal_items(self, user_id, limit):
        rows = self.connection.execute(
            """
            SELECT item_id
            FROM read_parquet(?)
            WHERE user_id = ? AND item_id IS NOT NULL
            GROUP BY item_id
            ORDER BY min(rank), item_id
            LIMIT ?
            """,
            [str(self.data_directory / "recommendations.parquet"), user_id, limit],
        ).fetchall()
        return [int(row[0]) for row in rows]

    def _popular_items(self, limit):
        rows = self.connection.execute(
            """
            SELECT item_id
            FROM read_parquet(?)
            WHERE item_id IS NOT NULL
            GROUP BY item_id
            ORDER BY min(rank), item_id
            LIMIT ?
            """,
            [str(self.data_directory / "top_popular.parquet"), limit],
        ).fetchall()
        return [int(row[0]) for row in rows]

    def _similar_rows(self, item_id, limit):
        return self.connection.execute(
            """
            SELECT item_id_2, min(rank) AS neighbor_rank
            FROM read_parquet(?)
            WHERE item_id_1 = ? AND item_id_2 != ? AND item_id_2 IS NOT NULL
            GROUP BY item_id_2
            ORDER BY neighbor_rank, item_id_2
            LIMIT ?
            """,
            [str(self.data_directory / "similar.parquet"), item_id, item_id, limit],
        ).fetchall()

    def _online_items(self, events):
        scores = {}
        listened_items = set(events)
        for event_position, item_id in enumerate(events[:3]):
            for neighbor_id, neighbor_rank in self._similar_rows(item_id, 10):
                if neighbor_id in listened_items or neighbor_rank is None or neighbor_rank <= 0:
                    continue
                score = 1 / ((event_position + 1) * neighbor_rank)
                scores[int(neighbor_id)] = scores.get(int(neighbor_id), 0) + score
        return sorted(scores, key=lambda item_id: (-scores[item_id], item_id))

    def get_recommendations(self, user_id, k):
        with self.lock:
            try:
                events = self.user_events.get(user_id, [])
                listened_items = set(events)
                online = [(item_id, "online") for item_id in self._online_items(events)]

                personal = self._personal_items(user_id, k + len(events))
                offline = [
                    (item_id, "personal") for item_id in personal if item_id not in listened_items
                ][:k]
                offline_ids = {item_id for item_id, _ in offline}
                popular = []
                if len(offline) < k:
                    # Запас компенсирует пересечение популярных треков с историей и personal.
                    popular = self._popular_items(k + len(events) + len(offline))
                    for item_id in popular:
                        if item_id not in listened_items and item_id not in offline_ids:
                            offline.append((item_id, "popular"))
                            offline_ids.add(item_id)
                        if len(offline) == k:
                            break

                recommendations = []
                selected_items = set(listened_items)
                used_sources = set()
                sources = [online, offline]
                positions = [0, 0]
                # Чередуем online и offline; повторы не занимают место в выдаче.
                while len(recommendations) < k and any(
                    positions[index] < len(source) for index, source in enumerate(sources)
                ):
                    for index, source in enumerate(sources):
                        while positions[index] < len(source):
                            item_id, source_name = source[positions[index]]
                            positions[index] += 1
                            if item_id not in selected_items:
                                recommendations.append(item_id)
                                selected_items.add(item_id)
                                used_sources.add(source_name)
                                break
                        if len(recommendations) == k:
                            break

                # Если кандидатов мало, используем оставшиеся популярные треки.
                for item_id in popular:
                    if len(recommendations) == k:
                        break
                    if item_id not in selected_items:
                        recommendations.append(item_id)
                        selected_items.add(item_id)
                        used_sources.add("popular")

                for source_name in used_sources:
                    self.request_counts[source_name] += 1
                return recommendations
            except duckdb.Error:
                self.request_counts["errors"] += 1
                logger.exception("Не удалось прочитать рекомендации")
                raise HTTPException(status_code=503, detail="Рекомендации временно недоступны")

    def get_similar(self, item_id, k):
        with self.lock:
            try:
                rows = self._similar_rows(item_id, k)
                self.request_counts["similar"] += 1
                return [int(row[0]) for row in rows]
            except duckdb.Error:
                self.request_counts["errors"] += 1
                logger.exception("Не удалось прочитать похожие треки")
                raise HTTPException(status_code=503, detail="Похожие треки временно недоступны")

    def stats(self):
        with self.lock:
            return dict(self.request_counts)

    def close(self):
        self.connection.close()


def create_app(data_directory=None):
    if data_directory is None:
        data_directory = os.getenv(
            "RECOMMENDATIONS_DIR", str(Path(__file__).parent / "data/recommendations")
        )

    @asynccontextmanager
    async def lifespan(app):
        store = RecommendationStore(data_directory)
        try:
            store.prepare()
            app.state.store = store
            yield
        finally:
            store.close()

    app = FastAPI(title="Рекомендации музыкальных треков", lifespan=lifespan)

    @app.post("/recommendations")
    def recommendations(
        user_id: int = Query(..., ge=0), k: int = Query(20, ge=1, le=100)
    ):
        return {"recs": app.state.store.get_recommendations(user_id, k)}

    @app.post("/similar_items")
    def similar_items(
        item_id: int = Query(..., ge=0), k: int = Query(10, ge=1, le=100)
    ):
        return {"recs": app.state.store.get_similar(item_id, k)}

    @app.post("/events")
    def add_event(user_id: int = Query(..., ge=0), item_id: int = Query(..., ge=0)):
        app.state.store.add_event(user_id, item_id)
        return {"status": "ok"}

    @app.get("/events")
    def get_events(user_id: int = Query(..., ge=0)):
        return {"events": app.state.store.get_events(user_id)}

    @app.delete("/events")
    def delete_events(user_id: int = Query(..., ge=0)):
        app.state.store.delete_events(user_id)
        return {"status": "ok"}

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.get("/stats")
    def stats():
        return app.state.store.stats()

    return app


app = create_app()
