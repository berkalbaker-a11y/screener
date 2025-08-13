from __future__ import annotations
from pathlib import Path
import pandas as pd
from .data_loader_yahoo import fetch_ohlcv_yahoo, resample_hourly_to_daily

SYMS = ["AKBNK", "THYAO", "EREGL"]  # örnek 3 BIST sembolü
SECTOR_MAP = {"AKBNK":"Bankacılık","THYAO":"Ulaştırma","EREGL":"Metal"}

def main():
    print("↳ 1s OHLCV indiriliyor...")
    hourly = fetch_ohlcv_yahoo(SYMS, period="60d", interval="60m", sector_map=SECTOR_MAP)
    print("hourly shape:", hourly.shape)
    # CSV çıktı (yerel çalıştırırsan): ./out/
    outdir = Path("out"); outdir.mkdir(parents=True, exist_ok=True)
    hourly.reset_index().to_csv(outdir/"hourly.csv", index=False)

    print("↳ Günlüğe resample ediliyor...")
    daily = resample_hourly_to_daily(hourly, market_tz="Europe/Istanbul")
    print("daily shape:", daily.shape)
    daily.reset_index().to_csv(outdir/"daily.csv", index=False)

    print("✔️ Tamam. Örnek son gün:", daily.index.get_level_values(0).max())

if __name__ == "__main__":
    main()
