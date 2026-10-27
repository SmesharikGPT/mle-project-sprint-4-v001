"""Сохранение результатов проекта в персональный S3-бакет."""

import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

import boto3
from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound


class BrowserStorage:
    """Доступ через уже настроенный аккаунт CS Browser в Windows."""

    def __init__(self, account):
        self.account = account
        self.executable = Path(os.getenv(
            "S3_BROWSER_CLI", r"C:\Program Files\CS Browser\s3browser-cli.exe"
        ))
        if not self.executable.is_file():
            raise FileNotFoundError("Не найден командный интерфейс CS Browser.")

    def run(self, *arguments):
        result = subprocess.run(
            [str(self.executable), "/file", arguments[0], self.account, *arguments[1:]],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output = result.stdout + result.stderr
        if result.returncode or re.search(r"\[E\]|\bERROR\b|\bFAILED\b", output, re.I):
            raise RuntimeError("Ошибка CS Browser:\n" + output[-2500:])
        return output

    def head_bucket(self, Bucket):
        self.run("list", Bucket + "/recsys", "s")

    def head_object(self, Bucket, Key):
        key = PurePosixPath(Key)
        folder = Bucket if str(key.parent) == "." else Bucket + "/" + str(key.parent)
        output = self.run("list", folder, "s")
        match = re.search(r"^" + re.escape(key.name) + r"\s+(\d+)\s*$", output, re.M)
        if match is None:
            raise FileNotFoundError("Объект не найден в S3: " + Key)
        return {"ContentLength": int(match.group(1))}

    def upload_file(self, filename, bucket, key):
        path, object_key = Path(filename), PurePosixPath(key)
        if path.name != object_key.name:
            raise ValueError("При загрузке через CS Browser имя файла должно совпадать с именем объекта.")
        folder = bucket if str(object_key.parent) == "." else bucket + "/" + str(object_key.parent)
        self.run("upload", str(path.resolve()), folder)

    def download_file(self, bucket, key, filename):
        path = Path(filename).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
            directory = Path(temporary).resolve()
            if directory.parent != path.parent:
                raise ValueError("Временная папка должна находиться рядом с целевым файлом.")
            self.run("download", bucket + "/" + key, str(directory))
            downloaded = directory / PurePosixPath(key).name
            if not downloaded.is_file():
                raise FileNotFoundError("CS Browser не сохранил скачанный файл.")
            downloaded.replace(path)


def ensure_storage():
    bucket = os.getenv("S3_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError(
            "Не задан S3_BUCKET. Укажите имя персонального бакета в переменной "
            "окружения перед запуском ноутбука. Ключи доступа в код добавлять не нужно."
        )

    browser_account = os.getenv("S3_BROWSER_ACCOUNT", "").strip()
    if browser_account:
        client = BrowserStorage(browser_account)
        client.head_bucket(Bucket=bucket)
        return client, bucket

    profile = os.getenv("AWS_PROFILE") or None
    endpoint = os.getenv("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
    try:
        session = boto3.Session(profile_name=profile)
        client = session.client("s3", endpoint_url=endpoint, region_name=session.region_name or "ru-central1")
        client.head_bucket(Bucket=bucket)
    except ProfileNotFound as error:
        raise RuntimeError("Не найден профиль AWS_PROFILE. Проверьте настройку S3 на виртуальной машине.") from error
    except (BotoCoreError, ClientError) as error:
        raise RuntimeError(
            "Не удалось подключиться к S3-бакету. Проверьте S3_BUCKET, "
            "S3_ENDPOINT_URL и доступ к бакету в настройках окружения."
        ) from error
    return client, bucket


def upload_file(path, key):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Файл для загрузки не найден: {}".format(path))
    client, bucket = ensure_storage()
    client.upload_file(str(path), bucket, key)
    saved_size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    if saved_size != path.stat().st_size:
        raise RuntimeError("Размер сохранённого в S3 файла не совпал с исходным.")
    return "s3://{}/{}".format(bucket, key)


def download_file(key, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    client, bucket = ensure_storage()
    client.download_file(bucket, key, str(path))
    if client.head_object(Bucket=bucket, Key=key)["ContentLength"] != path.stat().st_size:
        raise RuntimeError("Размер скачанного из S3 файла не совпал с размером объекта.")
    return path
