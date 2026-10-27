"""Последовательное выполнение основного ноутбука из командной строки."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description="Выполнить основной ноутбук проекта.")
    parser.add_argument(
        "--kernel",
        default="music-recsys",
        help="Название установленного Jupyter-ядра (по умолчанию music-recsys).",
    )
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent
    notebook_path = project_dir / "recommendations.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)

    def report_cell(cell, cell_index, **kwargs):
        if cell.cell_type == "code" and cell.source.strip():
            first_line = cell.source.splitlines()[0][:100]
            print("Ячейка {}: {}".format(cell_index + 1, first_line), flush=True)

    def save_progress(**kwargs):
        nbformat.write(notebook, notebook_path)

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(project_dir)}},
        allow_errors=False,
        on_cell_start=report_cell,
        on_cell_executed=save_progress,
    )
    try:
        client.execute()
    finally:
        # При ошибке сохраняется и уже выполненная часть с текстом ошибки.
        nbformat.write(notebook, notebook_path)
    print(f"Ноутбук выполнен и сохранён: {notebook_path.name}")


if __name__ == "__main__":
    main()
