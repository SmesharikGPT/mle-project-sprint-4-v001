import argparse
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests
from fastapi.testclient import TestClient

from recommendations_service import create_app


class RecommendationServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work_directory = Path(os.getenv("TEST_DATA_ROOT", "work/service_tests")).resolve()
        work_directory.mkdir(parents=True, exist_ok=True)
        cls.temporary_directory = tempfile.TemporaryDirectory(dir=work_directory)
        cls.data_directory = Path(cls.temporary_directory.name).resolve()
        if work_directory not in cls.data_directory.parents:
            raise ValueError("Временные файлы должны находиться внутри рабочего каталога")

        pd.DataFrame(
            {
                "user_id": [1, 1, 1, 1, 2],
                "item_id": [12, 11, 12, 13, 15],
                "score": [0.8, 0.9, 0.7, 0.6, 0.5],
                "rank": [2, 1, 3, 4, 1],
            }
        ).to_parquet(cls.data_directory / "recommendations.parquet", index=False)
        pd.DataFrame(
            {"item_id": [10, 20, 10, 30], "score": [8, 7, 6, 5], "rank": [1, 2, 3, 4]}
        ).to_parquet(cls.data_directory / "top_popular.parquet", index=False)
        pd.DataFrame(
            {
                "item_id_1": [11, 11, 11, 11],
                "item_id_2": [11, 12, 12, 14],
                "score": [1.0, 0.9, 0.8, 0.7],
                "rank": [1, 2, 3, 4],
            }
        ).to_parquet(cls.data_directory / "similar.parquet", index=False)

        cls.client_context = TestClient(create_app(cls.data_directory))
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        cls.temporary_directory.cleanup()

    def test_personal_order_and_limit(self):
        response = self.client.post("/recommendations", params={"user_id": 1, "k": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"recs": [11, 12]})

    def test_personal_duplicates_are_removed(self):
        response = self.client.post("/recommendations", params={"user_id": 1, "k": 10})
        self.assertEqual(response.json(), {"recs": [11, 12, 13]})

    def test_unknown_user_gets_popular(self):
        response = self.client.post("/recommendations", params={"user_id": 999, "k": 3})
        self.assertEqual(response.json(), {"recs": [10, 20, 30]})

    def test_single_personal_recommendation(self):
        response = self.client.post("/recommendations", params={"user_id": 2, "k": 5})
        self.assertEqual(response.json(), {"recs": [15]})

    def test_similar_tracks_without_itself_and_duplicates(self):
        response = self.client.post("/similar_items", params={"item_id": 11, "k": 10})
        self.assertEqual(response.json(), {"recs": [12, 14]})

    def test_unknown_track(self):
        response = self.client.post("/similar_items", params={"item_id": 999})
        self.assertEqual(response.json(), {"recs": []})

    def test_invalid_parameters(self):
        for params in [
            {"user_id": 1, "k": 0},
            {"user_id": 1, "k": 101},
            {"user_id": -1},
            {"user_id": "не число"},
            {"k": 10},
        ]:
            with self.subTest(params=params):
                self.assertEqual(self.client.post("/recommendations", params=params).status_code, 422)
        self.assertEqual(
            self.client.post("/similar_items", params={"item_id": 11, "k": -1}).status_code, 422
        )

    def test_health_and_statistics(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "healthy"})
        before = self.client.get("/stats").json()
        self.client.post("/recommendations", params={"user_id": 1})
        after = self.client.get("/stats").json()
        self.assertEqual(after["personal"], before["personal"] + 1)
        self.assertEqual(after["errors"], 0)


def run_live_checks(url, data_directory, output_path):
    """Проверяет запущенный HTTP-сервис на готовых файлах рекомендаций."""
    data_directory = Path(data_directory)
    output_path = Path(output_path)
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "checks": [],
    }

    def check(condition, message):
        if not condition:
            raise AssertionError(message)

    def request(session, name, method, path, params=None):
        result = {"name": name, "method": method, "path": path, "passed": False}
        if params is not None:
            result["parameters"] = params
        report["checks"].append(result)
        started = time.perf_counter()
        try:
            response = session.request(method, url.rstrip("/") + path, params=params, timeout=60)
        except requests.RequestException as error:
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            result["error_type"] = type(error).__name__
            raise RuntimeError("Не удалось выполнить HTTP-запрос.") from None
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["status_code"] = response.status_code
        try:
            result["response"] = response.json()
        except ValueError:
            result["response"] = None
        return result

    def check_recommendations(result, expected, k, excluded_item=None):
        check(result["status_code"] == 200, result["name"] + ": ожидался HTTP 200.")
        body = result["response"]
        check(isinstance(body, dict) and set(body) == {"recs"}, "Неверная структура ответа рекомендаций.")
        recommendations = body["recs"]
        check(isinstance(recommendations, list), "Поле recs должно содержать список.")
        check(all(type(item_id) is int for item_id in recommendations), "Идентификаторы должны быть целыми числами.")
        check(len(recommendations) <= k, "Размер ответа превысил k.")
        check(len(set(recommendations)) == len(recommendations), "В ответе есть повторные треки.")
        if excluded_item is not None:
            check(excluded_item not in recommendations, "Исходный трек оказался в похожих.")
        check(recommendations == expected, result["name"] + ": ответ не совпал с сохранёнными рекомендациями.")
        result["passed"] = True

    try:
        # Читаем только идентификаторы и нужные строки, не загружая файлы в pandas.
        with duckdb.connect() as connection:
            connection.execute("SET memory_limit = '1GB'")
            connection.execute("SET threads = 2")
            personal_file = str(data_directory / "recommendations.parquet")
            popular_file = str(data_directory / "top_popular.parquet")
            similar_file = str(data_directory / "similar.parquet")
            first_user = connection.execute(
                "SELECT user_id FROM read_parquet(?) WHERE user_id IS NOT NULL LIMIT 1", [personal_file]
            ).fetchone()
            first_item = connection.execute(
                "SELECT item_id_1 FROM read_parquet(?) WHERE item_id_1 IS NOT NULL LIMIT 1", [similar_file]
            ).fetchone()
            check(first_user is not None and first_item is not None, "Файлы рекомендаций не должны быть пустыми.")
            user_id, item_id = int(first_user[0]), int(first_item[0])
            cold_user_id = int(connection.execute(
                "SELECT CAST(MAX(user_id) AS BIGINT) + 1 FROM read_parquet(?)", [personal_file]
            ).fetchone()[0])
            unknown_item_id = int(connection.execute(
                "SELECT CAST(MAX(item_id_1) AS BIGINT) + 1 FROM read_parquet(?)", [similar_file]
            ).fetchone()[0])
            personal = [int(row[0]) for row in connection.execute(
                """
                SELECT item_id FROM read_parquet(?) WHERE user_id = ? AND item_id IS NOT NULL
                GROUP BY item_id ORDER BY MIN(rank), item_id LIMIT 20
                """, [personal_file, user_id]
            ).fetchall()]
            popular = [int(row[0]) for row in connection.execute(
                """
                SELECT item_id FROM read_parquet(?) WHERE item_id IS NOT NULL
                GROUP BY item_id ORDER BY MIN(rank), item_id LIMIT 20
                """, [popular_file]
            ).fetchall()]
            similar = [int(row[0]) for row in connection.execute(
                """
                SELECT item_id_2 FROM read_parquet(?)
                WHERE item_id_1 = ? AND item_id_2 != ? AND item_id_2 IS NOT NULL
                GROUP BY item_id_2 ORDER BY MIN(rank), item_id_2 LIMIT 10
                """, [similar_file, item_id, item_id]
            ).fetchall()]
        check(bool(personal) and bool(popular) and bool(similar), "Для проверки нужны непустые ожидаемые списки.")
        report["examples"] = {"user_id": user_id, "cold_user_id": cold_user_id,
                              "item_id": item_id, "unknown_item_id": unknown_item_id}

        with requests.Session() as session:
            health = request(session, "health", "GET", "/health")
            check(health["status_code"] == 200 and health["response"] == {"status": "healthy"}, "Проверка health не прошла.")
            health["passed"] = True
            before = request(session, "stats_before", "GET", "/stats")
            check(before["status_code"] == 200 and isinstance(before["response"], dict), "Счётчики недоступны.")
            required_counts = {"personal", "popular", "similar", "errors"}
            check(required_counts.issubset(before["response"]), "В stats отсутствуют счётчики.")
            check(all(type(before["response"][name]) is int for name in required_counts), "Счётчики должны быть целыми числами.")
            before["passed"] = True

            result = request(session, "personal", "POST", "/recommendations", {"user_id": user_id, "k": 20})
            check_recommendations(result, personal, 20)
            result = request(session, "cold_user", "POST", "/recommendations", {"user_id": cold_user_id, "k": 20})
            check_recommendations(result, popular, 20)
            result = request(session, "similar", "POST", "/similar_items", {"item_id": item_id, "k": 10})
            check_recommendations(result, similar, 10, excluded_item=item_id)
            result = request(session, "unknown_track", "POST", "/similar_items", {"item_id": unknown_item_id, "k": 10})
            check_recommendations(result, [], 10, excluded_item=unknown_item_id)
            result = request(session, "invalid_k", "POST", "/recommendations", {"user_id": user_id, "k": 0})
            check(result["status_code"] == 422, "Для k=0 ожидался HTTP 422.")
            result["passed"] = True

            after = request(session, "stats_after", "GET", "/stats")
            check(after["status_code"] == 200 and isinstance(after["response"], dict), "Итоговые счётчики недоступны.")
            check(required_counts.issubset(after["response"]), "В итоговых stats отсутствуют счётчики.")
            check(all(type(after["response"][name]) is int for name in required_counts), "Итоговые счётчики должны быть целыми числами.")
            for name, increase in [("personal", 1), ("popular", 1), ("similar", 2)]:
                check(after["response"][name] >= before["response"][name] + increase, "Не обновился счётчик " + name + ".")
            check(after["response"]["errors"] == before["response"]["errors"], "За время проверки появились ошибки сервиса.")
            after["passed"] = True
        report["status"] = "passed"
    except AssertionError as error:
        report["error"] = str(error)
        report["error_type"] = type(error).__name__
    except Exception as error:
        # В отчёт не попадают системные сообщения с URL, ключами или путями файлов.
        report["error"] = "Проверка не завершена. Проверьте запуск сервиса и наличие готовых Parquet-файлов."
        report["error_type"] = type(error).__name__
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Проверка запущенного сервиса:", report["status"])
    for result in report["checks"]:
        print("{}: HTTP {}, {} мс".format(result["name"], result.get("status_code", "нет ответа"), result.get("latency_ms", 0)))
    if report["status"] != "passed":
        print(report["error"])
    return report


if __name__ == "__main__":
    if "--live" in sys.argv:
        parser = argparse.ArgumentParser(description="Проверить запущенный сервис на рассчитанных рекомендациях.")
        parser.add_argument("--live", action="store_true", help="Проверить настоящий HTTP-сервис.")
        parser.add_argument("--url", default="http://127.0.0.1:8000", help="Адрес запущенного сервиса.")
        parser.add_argument("--data-dir", default="data/recommendations", help="Каталог с готовыми рекомендациями.")
        parser.add_argument("--output", default="service_test_results.json", help="JSON-файл с результатами.")
        arguments = parser.parse_args()
        results = run_live_checks(arguments.url, arguments.data_dir, arguments.output)
        raise SystemExit(0 if results["status"] == "passed" else 1)
    unittest.main(verbosity=2)
