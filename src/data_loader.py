"""Load and clean raw DSE data."""
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Scrip", "Date"]).reset_index(drop=True)

    # Basic sanity checks
    assert {"Date", "Scrip", "Open", "High", "Low", "Close", "Volume"} \
        .issubset(df.columns), "Missing required columns"

    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Total rows: {len(df)}")
    print(f"Unique companies: {df['Scrip'].nunique()}")
    print("\nRows per year:")
    print(df["Date"].dt.year.value_counts().sort_index())

    return df


def filter_valid_companies(df: pd.DataFrame, min_records: int = 500) -> pd.DataFrame:
    counts = df.groupby("Scrip").size()
    valid = counts[counts >= min_records].index
    print(f"Valid companies (>={min_records} records): {len(valid)}")
    return df[df["Scrip"].isin(valid)].copy()