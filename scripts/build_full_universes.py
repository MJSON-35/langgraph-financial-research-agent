"""Build free full-universe CSV snapshots for S&P500 and KOSPI200."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_DIR = PROJECT_ROOT / "data" / "universes"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
KOSPI200_URL = "https://en.wikipedia.org/wiki/KOSPI_200"
ALLOWED_SOURCE_HOSTS = {"en.wikipedia.org"}


def read_html_tables(url: str) -> list[pd.DataFrame]:
    """Fetch HTML with a browser-like user-agent before parsing tables."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_SOURCE_HOSTS:
        raise ValueError("Universe sources must use an approved HTTPS host.")
    if parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment:
        raise ValueError("Universe source URL contains unsupported components.")
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")
    return pd.read_html(StringIO(html))


def build_sp500_frame() -> pd.DataFrame:
    tables = read_html_tables(SP500_URL)
    components = tables[0].copy()
    frame = pd.DataFrame(
        {
            "ticker": components["Symbol"].astype(str).str.strip(),
            "name": components["Security"].astype(str).str.strip(),
            "sector": components["GICS Sector"].astype(str).str.strip(),
            "market": "SP500",
        }
    )
    return frame.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def build_kospi200_frame() -> pd.DataFrame:
    tables = read_html_tables(KOSPI200_URL)
    candidates = [table.copy() for table in tables if len(table.columns) >= 3]
    if not candidates:
        raise ValueError("No KOSPI200 composition table was found.")

    for table in candidates:
        renamed_columns = {str(column).strip().lower(): column for column in table.columns}
        symbol_column = next(
            (
                renamed_columns[key]
                for key in renamed_columns
                if "symbol" in key or "ticker" in key or "code" in key
            ),
            None,
        )
        name_column = next(
            (
                renamed_columns[key]
                for key in renamed_columns
                if "company" in key or "constituent" in key or "name" in key
            ),
            None,
        )
        sector_column = next(
            (
                renamed_columns[key]
                for key in renamed_columns
                if "sector" in key or "industry" in key or "gics" in key
            ),
            None,
        )

        if symbol_column is None and len(table.columns) >= 3:
            possible_symbol = table.iloc[:, 1].astype(str).str.extract(r"(\d{6})", expand=False)
            if possible_symbol.notna().sum() >= 50:
                name_column = table.columns[0]
                symbol_column = table.columns[1]
                sector_column = table.columns[2]

        if symbol_column is None or name_column is None:
            continue

        raw_symbol = table[symbol_column].astype(str).str.strip()
        numeric_symbol = raw_symbol.str.extract(r"(\d{6})", expand=False)
        frame = pd.DataFrame(
            {
                "ticker": numeric_symbol.fillna("").astype(str).str.zfill(6) + ".KS",
                "name": table[name_column].astype(str).str.strip(),
                "sector": table[sector_column].astype(str).str.strip() if sector_column is not None else "Unknown",
                "market": "KOSPI200",
            }
        )
        frame = frame[frame["ticker"].str.match(r"^\d{6}\.KS$")].copy()
        if len(frame) >= 150:
            return frame.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    raise ValueError("Could not identify symbol/name columns in the KOSPI200 table.")


def main() -> None:
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    sp500_frame = build_sp500_frame()
    kospi200_frame = build_kospi200_frame()

    sp500_path = UNIVERSE_DIR / "sp500_full.csv"
    kospi200_path = UNIVERSE_DIR / "kospi200_full.csv"

    sp500_frame.to_csv(sp500_path, index=False, encoding="utf-8")
    kospi200_frame.to_csv(kospi200_path, index=False, encoding="utf-8")

    print(f"Saved {len(sp500_frame)} S&P500 rows to {sp500_path}")
    print(f"Saved {len(kospi200_frame)} KOSPI200 rows to {kospi200_path}")


if __name__ == "__main__":
    main()
