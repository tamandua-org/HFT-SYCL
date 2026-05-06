import pandas as pd
import os

INPUT_FILE = "stockData20202024.csv"
OUTPUT_DIR = "."

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load CSV
    df = pd.read_csv(INPUT_FILE)

    # Drop unnamed index column if it exists
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(df.columns[0], axis=1)

    # Reverse order (first becomes last)
    df = df.iloc[::-1].reset_index(drop=True)

    # Select only price columns
    price_columns = [c for c in df.columns if c.endswith("_Price")]

    print(f"Found {len(price_columns)} price columns")

    # Process each asset
    for col in price_columns:
        asset_name = col.replace("_Price", "").upper()

        output_path = f"{asset_name}"

        # Extract column, drop NaNs
        series = df[col].dropna()

        # Write one price per line
        with open(output_path, "w") as f:
            for value in series:
                f.write(f"{value}\n")

        print(f"Wrote {output_path} ({len(series)} values)")


if __name__ == "__main__":
    main()
