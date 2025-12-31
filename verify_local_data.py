import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = r"G:\My Drive\SuperPeterTrader"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def find_data_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Data folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Data path is not a folder: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def read_data(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(file_path)
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding="latin-1")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def print_summary(file_path: Path, df: pd.DataFrame) -> None:
    print("=" * 80)
    print(f"File: {file_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Columns: {', '.join(df.columns.astype(str).tolist())}")
    missing = df.isna().sum()
    if missing.any():
        missing_cols = ", ".join(f"{col}={int(count)}" for col, count in missing.items() if count)
        print(f"Missing values: {missing_cols}")
    else:
        print("Missing values: none")
    print("Head:")
    print(df.head(5).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local data files and show a quick summary.")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Path to data folder (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = find_data_files(data_dir)
    if not files:
        print(f"No data files found in: {data_dir}")
        return 1

    print(f"Found {len(files)} file(s) in {data_dir}")
    for file_path in files:
        df = read_data(file_path)
        print_summary(file_path, df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
