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
        self.request_counts = {"personal": 0, "popular": 0, "similar": 0, "errors": 0}

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

    def get_recommendations(self, user_id, k):
        with self.lock:
            try:
                rows = self.connection.execute(
                    """
                    SELECT item_id
                    FROM read_parquet(?)
                    WHERE user_id = ? AND item_id IS NOT NULL
                    GROUP BY item_id
                    ORDER BY min(rank), item_id
                    LIMIT ?
                    """,
                    [str(self.data_directory / "recommendations.parquet"), user_id, k],
                ).fetchall()
                if rows:
                    self.request_counts["personal"] += 1
                else:
                    rows = self.connection.execute(
                        """
                        SELECT item_id
                        FROM read_parquet(?)
                        WHERE item_id IS NOT NULL
                        GROUP BY item_id
                        ORDER BY min(rank), item_id
                        LIMIT ?
                        """,
                        [str(self.data_directory / "top_popular.parquet"), k],
                    ).fetchall()
                    self.request_counts["popular"] += 1
                return [int(row[0]) for row in rows]
            except duckdb.Error:
                self.request_counts["errors"] += 1
                logger.exception("Не удалось прочитать рекомендации")
                raise HTTPException(status_code=503, detail="Рекомендации временно недоступны")

    def get_similar(self, item_id, k):
        with self.lock:
            try:
                rows = self.connection.execute(
                    """
                    SELECT item_id_2
                    FROM read_parquet(?)
                    WHERE item_id_1 = ? AND item_id_2 != ? AND item_id_2 IS NOT NULL
                    GROUP BY item_id_2
                    ORDER BY min(rank), item_id_2
                    LIMIT ?
                    """,
                    [str(self.data_directory / "similar.parquet"), item_id, item_id, k],
                ).fetchall()
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

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.get("/stats")
    def stats():
        return app.state.store.stats()

    return app


app = create_app()
