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

        pd.DataFrame({
            "user_id": [1, 1, 1, 1, 2],
            "item_id": [12, 11, 12, 13, 15],
            "score": [0.8, 0.9, 0.7, 0.6, 0.5],
            "rank": [2, 1, 3, 4, 1],
        }).to_parquet(cls.data_directory / "recommendations.parquet", index=False)
        pd.DataFrame({
            "item_id": [10, 20, 10, 30],
            "score": [8, 7, 6, 5], "rank": [1, 2, 3, 4],
        }).to_parquet(cls.data_directory / "top_popular.parquet", index=False)
        pd.DataFrame({
            "item_id_1": [11, 11, 11, 11, 11, 21, 21, 21, 22, 22, 23],
            "item_id_2": [11, 12, 12, 14, 16, 14, 17, 12, 18, 14, 19],
            "score": [1.0, 0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.9, 0.8, 0.9],
            "rank": [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1],
        }).to_parquet(cls.data_directory / "similar.parquet", index=False)
        cls.client_context = TestClient(create_app(cls.data_directory))
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        cls.temporary_directory.cleanup()

    def setUp(self):
        for user_id in [1, 2, 999]:
            self.client.delete("/events", params={"user_id": user_id})

    def add_event(self, item_id, user_id=1):
        response = self.client.post("/events", params={"user_id": user_id, "item_id": item_id})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def recommendations(self, user_id=1, k=20):
        response = self.client.post("/recommendations", params={"user_id": user_id, "k": k})
        self.assertEqual(response.status_code, 200)
        return response.json()["recs"]

    def test_personal_without_online_history(self):
        self.assertEqual(self.client.get("/events", params={"user_id": 1}).json(), {"events": []})
        self.assertEqual(self.recommendations(k=2), [11, 12])

    def test_personal_duplicates_and_popular_completion(self):
        self.assertEqual(self.recommendations(k=10), [11, 12, 13, 10, 20, 30])

    def test_user_without_personal_recommendations(self):
        self.assertEqual(self.recommendations(user_id=999, k=3), [10, 20, 30])

    def test_short_personal_list_is_completed(self):
        self.assertEqual(self.recommendations(user_id=2, k=5), [15, 10, 20, 30])

    def test_personal_with_online_history(self):
        self.add_event(11)
        # Из offline убран 11. Общий для источников трек 12 выдаётся один раз.
        self.assertEqual(self.recommendations(k=6), [12, 13, 14, 10, 16, 20])

    def test_unknown_user_with_history(self):
        self.add_event(11, user_id=999)
        self.assertEqual(self.recommendations(user_id=999, k=6), [12, 10, 14, 20, 16, 30])

    def test_history_without_similar_items_keeps_offline(self):
        self.add_event(99999)
        self.assertEqual(self.recommendations(k=3), [11, 12, 13])

    def test_all_recent_items_are_excluded(self):
        for item_id in [11, 12, 14, 10]:
            self.add_event(item_id)
        recommendations = self.recommendations(k=20)
        self.assertTrue(recommendations)
        self.assertFalse(set(recommendations) & {11, 12, 14, 10})
        self.assertEqual(len(recommendations), len(set(recommendations)))

    def test_recency_changes_the_first_online_candidate(self):
        self.add_event(23)
        self.add_event(22)
        self.assertEqual(self.recommendations(k=1), [18])
        self.add_event(23)
        self.assertEqual(self.recommendations(k=1), [19])

    def test_only_three_latest_events_generate_candidates(self):
        for item_id in [23, 22, 21, 11]:
            self.add_event(item_id)
        self.assertNotIn(19, self.recommendations(k=100))

    def test_history_is_unique_bounded_and_can_be_cleared(self):
        for item_id in range(500, 525):
            self.add_event(item_id)
        self.add_event(510)
        expected = [510] + [item_id for item_id in range(524, 504, -1) if item_id != 510]
        self.assertEqual(self.client.get("/events", params={"user_id": 1}).json(), {"events": expected})
        response = self.client.delete("/events", params={"user_id": 1})
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(self.client.get("/events", params={"user_id": 1}).json(), {"events": []})

    def test_similar_tracks_without_itself_and_duplicates(self):
        response = self.client.post("/similar_items", params={"item_id": 11, "k": 10})
        self.assertEqual(response.json(), {"recs": [12, 14, 16]})

    def test_unknown_track(self):
        response = self.client.post("/similar_items", params={"item_id": 999})
        self.assertEqual(response.json(), {"recs": []})

    def test_invalid_parameters(self):
        for params in [{"user_id": 1, "k": 0}, {"user_id": 1, "k": 101},
                       {"user_id": -1}, {"user_id": "не число"}, {"k": 10}]:
            with self.subTest(params=params):
                self.assertEqual(self.client.post("/recommendations", params=params).status_code, 422)
        for params in [{"user_id": 1, "item_id": -1}, {"user_id": -1, "item_id": 11},
                       {"user_id": 1}, {"user_id": 1, "item_id": "не число"}]:
            with self.subTest(params=params):
                self.assertEqual(self.client.post("/events", params=params).status_code, 422)
        self.assertEqual(self.client.get("/events", params={"user_id": -1}).status_code, 422)
        self.assertEqual(self.client.delete("/events", params={"user_id": -1}).status_code, 422)
        self.assertEqual(self.client.post("/similar_items", params={"item_id": 11, "k": -1}).status_code, 422)

    def test_health_and_statistics(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "healthy"})
        before = self.client.get("/stats").json()
        self.add_event(11)
        self.recommendations(k=6)
        after = self.client.get("/stats").json()
        for name in ["personal", "popular", "online", "events"]:
            self.assertEqual(after[name], before[name] + 1)
        self.assertEqual(after["errors"], before["errors"])
        self.assertEqual(after["similar"], before["similar"])


def run_live_checks(url, data_directory, output_path, log_path="test_service.log"):
    """Проверяет три сценария задания через настоящий HTTP-сервис."""
    data_directory = Path(data_directory)
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "failed", "checks": []}

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

    def check_body(result, expected):
        check(result["status_code"] == 200, result["name"] + ": ожидался HTTP 200.")
        check(result["response"] == expected, result["name"] + ": неожиданный ответ.")
        result["passed"] = True

    def check_recommendations(result, expected, k, excluded=()):
        check_body(result, {"recs": expected})
        recommendations = result["response"]["recs"]
        check(all(type(item_id) is int for item_id in recommendations), "Идентификаторы должны быть целыми числами.")
        check(len(recommendations) <= k, "Размер ответа превысил k.")
        check(len(set(recommendations)) == len(recommendations), "В ответе есть повторные треки.")
        check(not set(recommendations).intersection(excluded), "В ответ попали треки из текущей истории.")

    try:
        # Для эталонных списков читаем только нужные строки из рассчитанных файлов.
        with duckdb.connect() as connection:
            connection.execute("SET memory_limit = '1GB'")
            connection.execute("SET threads = 2")
            personal_file = str(data_directory / "recommendations.parquet")
            popular_file = str(data_directory / "top_popular.parquet")
            similar_file = str(data_directory / "similar.parquet")
            first_user = connection.execute(
                "SELECT user_id FROM read_parquet(?) WHERE user_id IS NOT NULL LIMIT 1", [personal_file]
            ).fetchone()
            check(first_user is not None, "Файл персональных рекомендаций пуст.")
            user_id = int(first_user[0])
            cold_user_id = int(connection.execute(
                "SELECT CAST(MAX(user_id) AS BIGINT) + 1 FROM read_parquet(?)", [personal_file]
            ).fetchone()[0])
            unknown_item_id = int(connection.execute(
                "SELECT GREATEST(MAX(item_id_1), MAX(item_id_2))::BIGINT + 1 FROM read_parquet(?)", [similar_file]
            ).fetchone()[0])
            personal = [int(row[0]) for row in connection.execute(
                """SELECT item_id FROM read_parquet(?) WHERE user_id = ? AND item_id IS NOT NULL
                   GROUP BY item_id ORDER BY MIN(rank), item_id LIMIT 20""", [personal_file, user_id]
            ).fetchall()]
            popular = [int(row[0]) for row in connection.execute(
                """SELECT item_id FROM read_parquet(?) WHERE item_id IS NOT NULL
                   GROUP BY item_id ORDER BY MIN(rank), item_id LIMIT 100""", [popular_file]
            ).fetchall()]
            check(len(personal) == 20 and len(popular) >= 20, "Для проверки нужны полные списки по 20 треков.")

            # Выбираем событие, чей первый сосед добавит новый трек к offline-топу.
            candidates = connection.execute(
                """SELECT DISTINCT item_id_1 FROM read_parquet(?)
                   WHERE item_id_1 NOT IN (SELECT unnest(?)) AND item_id_1 IS NOT NULL
                   ORDER BY item_id_1 LIMIT 100""", [similar_file, personal]
            ).fetchall()
            similar = []
            for candidate in candidates:
                item_id = int(candidate[0])
                similar = [int(row[0]) for row in connection.execute(
                    """SELECT item_id_2 FROM read_parquet(?)
                       WHERE item_id_1 = ? AND item_id_2 != ? AND item_id_2 IS NOT NULL
                       GROUP BY item_id_2 ORDER BY MIN(rank), item_id_2 LIMIT 10""",
                    [similar_file, item_id, item_id]
                ).fetchall()]
                if similar and similar[0] not in personal:
                    break
            check(bool(similar) and similar[0] not in personal, "Не найден пример нового online-кандидата.")

        # При одном событии online-порядок совпадает с порядком его соседей.
        offline = personal[:]
        expected_mixed = []
        online_index = offline_index = 0
        while len(expected_mixed) < 20 and (online_index < len(similar) or offline_index < len(offline)):
            for source in ["online", "offline"]:
                values = similar if source == "online" else offline
                index = online_index if source == "online" else offline_index
                while index < len(values) and values[index] in expected_mixed:
                    index += 1
                if index < len(values) and len(expected_mixed) < 20:
                    expected_mixed.append(values[index])
                    index += 1
                if source == "online":
                    online_index = index
                else:
                    offline_index = index
        report["examples"] = {"user_id": user_id, "cold_user_id": cold_user_id,
                              "history_item_id": item_id, "new_online_item_id": similar[0]}

        with requests.Session() as session:
            try:
                result = request(session, "Проверка готовности", "GET", "/health")
                check_body(result, {"status": "healthy"})
                before = request(session, "Счётчики до проверки", "GET", "/stats")
                check(before["status_code"] == 200, "Счётчики недоступны.")
                required_counts = {"personal", "popular", "online", "events", "similar", "errors"}
                check(required_counts.issubset(before["response"]), "В stats отсутствуют счётчики.")
                before["passed"] = True
                for test_user in [user_id, cold_user_id]:
                    result = request(session, "Очистка тестовой истории", "DELETE", "/events", {"user_id": test_user})
                    check_body(result, {"status": "ok"})

                result = request(session, "1. Пользователь без персональных рекомендаций", "POST", "/recommendations",
                                 {"user_id": cold_user_id, "k": 20})
                check_recommendations(result, popular[:20], 20)

                result = request(session, "Пустая online-история", "GET", "/events", {"user_id": user_id})
                check_body(result, {"events": []})
                result = request(session, "2. Персональные рекомендации без online-истории", "POST", "/recommendations",
                                 {"user_id": user_id, "k": 20})
                check_recommendations(result, personal, 20)

                result = request(session, "Похожие треки", "POST", "/similar_items", {"item_id": item_id, "k": 10})
                check_recommendations(result, similar, 10, excluded=[item_id])
                result = request(session, "Неизвестный трек", "POST", "/similar_items", {"item_id": unknown_item_id, "k": 10})
                check_recommendations(result, [], 10)
                result = request(session, "Добавление нового прослушивания", "POST", "/events",
                                 {"user_id": user_id, "item_id": item_id})
                check_body(result, {"status": "ok"})
                result = request(session, "История после прослушивания", "GET", "/events", {"user_id": user_id})
                check_body(result, {"events": [item_id]})
                result = request(session, "3. Персональные рекомендации с online-историей", "POST", "/recommendations",
                                 {"user_id": user_id, "k": 20})
                check_recommendations(result, expected_mixed, 20, excluded=[item_id])
                mixed = result["response"]["recs"]
                check(mixed != personal, "Новое событие не изменило выдачу.")
                check(similar[0] in mixed and similar[0] not in personal, "В выдачу не добавился новый online-трек.")
                check(bool(set(mixed) & (set(personal) - set(similar))), "В смешанной выдаче нет самостоятельного offline-кандидата.")
                result = request(session, "Неверный размер выдачи", "POST", "/recommendations", {"user_id": user_id, "k": 0})
                check(result["status_code"] == 422, "Для k=0 ожидался HTTP 422.")
                result["passed"] = True
            finally:
                # Удаляем только события пользователей, использованных в этом запуске.
                for test_user in [user_id, cold_user_id]:
                    result = request(session, "Удаление тестовой истории", "DELETE", "/events", {"user_id": test_user})
                    check_body(result, {"status": "ok"})

            result = request(session, "Возврат к offline после очистки истории", "POST", "/recommendations",
                             {"user_id": user_id, "k": 20})
            check_recommendations(result, personal, 20)
            after = request(session, "Счётчики после проверки", "GET", "/stats")
            check(after["status_code"] == 200 and required_counts.issubset(after["response"]), "Итоговые счётчики недоступны.")
            for name, increase in [("personal", 3), ("popular", 1), ("online", 1), ("events", 1), ("similar", 2)]:
                check(after["response"][name] >= before["response"][name] + increase, "Не обновился счётчик " + name + ".")
            check(after["response"]["errors"] == before["response"]["errors"], "За время проверки появились ошибки сервиса.")
            after["passed"] = True
        report["status"] = "passed"
    except AssertionError as error:
        report["error"] = str(error)
        report["error_type"] = type(error).__name__
    except Exception as error:
        report["error"] = "Проверка не завершена. Проверьте запуск сервиса и наличие готовых Parquet-файлов."
        report["error_type"] = type(error).__name__

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["Проверка микросервиса рекомендаций", "Начало (UTC): " + report["started_at_utc"]]
    for result in report["checks"]:
        lines.append("\n" + result["name"])
        lines.append("{} {} {}".format(result["method"], result["path"],
                     json.dumps(result.get("parameters", {}), ensure_ascii=False)))
        lines.append("HTTP {}, {} мс; проверка: {}".format(result.get("status_code", "нет ответа"),
                     result.get("latency_ms", 0), "пройдена" if result["passed"] else "не пройдена"))
        lines.append("Ответ: " + json.dumps(result.get("response"), ensure_ascii=False))
    passed = sum(result["passed"] for result in report["checks"])
    lines.append("\nИтог: {}. Пройдено HTTP-проверок: {}/{}.".format(report["status"], passed, len(report["checks"])))
    if report["status"] == "passed":
        lines.append("Все три сценария выполнены. Новое прослушивание изменило выдачу; online- и offline-треки присутствуют.")
    else:
        lines.append(report["error"])
    text = "\n".join(lines) + "\n"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")
    print(text, end="")
    return report


if __name__ == "__main__":
    if "--live" in sys.argv:
        parser = argparse.ArgumentParser(description="Проверить три сценария на запущенном сервисе.")
        parser.add_argument("--live", action="store_true", help="Проверить настоящий HTTP-сервис.")
        parser.add_argument("--url", default="http://127.0.0.1:8000", help="Адрес запущенного сервиса.")
        parser.add_argument("--data-dir", default="data/recommendations", help="Каталог с готовыми рекомендациями.")
        parser.add_argument("--output", default="service_test_results.json", help="JSON-файл с результатами.")
        parser.add_argument("--log", default="test_service.log", help="Текстовый вывод проверки в UTF-8.")
        arguments = parser.parse_args()
        results = run_live_checks(arguments.url, arguments.data_dir, arguments.output, arguments.log)
        raise SystemExit(0 if results["status"] == "passed" else 1)
    unittest.main(verbosity=2)
