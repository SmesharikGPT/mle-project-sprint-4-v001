"""Загрузка исходных данных проекта."""

from pathlib import Path

import pyarrow.parquet as pq
import requests


DATA_URLS = {
    "tracks.parquet": "https://mle-data.storage.yandexcloud.net/ym/tracks.parquet",
    "catalog_names.parquet": "https://mle-data.storage.yandexcloud.net/ym/catalog_names.parquet",
    "interactions.parquet": "https://mle-data.storage.yandexcloud.net/ym/interactions.parquet",
}


def download_data(data_dir="data/raw"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for filename, url in DATA_URLS.items():
        path = data_dir / filename
        if path.exists():
            try:
                pq.read_metadata(path)
                print("Файл уже загружен:", filename)
                paths[filename] = path
                continue
            except Exception:
                print("Повторная загрузка файла:", filename)

        partial_path = path.with_suffix(path.suffix + ".part")
        with requests.Session() as session:
            head = session.head(url, timeout=(15, 30))
            head.raise_for_status()
            total_size = int(head.headers["Content-Length"])
            start = partial_path.stat().st_size if partial_path.exists() else 0
            if start > total_size:
                raise RuntimeError("Размер частично загруженного файла превышает исходный: " + filename)
            # Небольшие диапазоны устойчивее одного запроса на файл размером более гигабайта.
            while start < total_size:
                for attempt in range(3):
                    start = partial_path.stat().st_size if partial_path.exists() else 0
                    end = min(start + 32 * 1024 * 1024 - 1, total_size - 1)
                    try:
                        with session.get(
                            url, headers={"Range": "bytes={}-{}".format(start, end)},
                            stream=True, timeout=(15, 90),
                        ) as response:
                            response.raise_for_status()
                            expected_range = "bytes {}-{}/{}".format(start, end, total_size)
                            if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                                raise RuntimeError("Сервер вернул неверный диапазон байтов: " + filename)
                            with partial_path.open("ab") as file:
                                for block in response.iter_content(chunk_size=1024 * 1024):
                                    if block:
                                        file.write(block)
                        start = partial_path.stat().st_size
                        print("{}: {:.1f}% ({:.0f} МБ)".format(
                            filename, 100 * start / total_size, start / 1024 ** 2), flush=True)
                        break
                    except (requests.RequestException, OSError) as error:
                        if attempt == 2:
                            raise RuntimeError("Не удалось загрузить {}. Повторите запуск загрузки.".format(filename)) from error
                        print("Соединение прервано. Повторная попытка для", filename, flush=True)
        if partial_path.stat().st_size != total_size:
            raise RuntimeError("Не совпал размер загруженного файла: " + filename)
        pq.read_metadata(partial_path)
        partial_path.replace(path)
        print("Загружен файл:", filename)
        paths[filename] = path
    return paths


if __name__ == "__main__":
    download_data()
