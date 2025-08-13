from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Iterable, Optional

# BIST Yahoo kodları genelde ".IS": AKBNK.IS, THYAO.IS, EREGL.IS
def to_yahoo_symbol(sym: str, suffix: str = ".IS") -> str:
    s = sym.strip().upper()
    return s if s.endswith(suffix) else s + suffix

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    rename = {"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"}
    out = df.rename(columns=rename).copy()
    out.index = pd.to_datetime(out.index)  # saat bilgisi korunur (çoğu zaman UTC)
    return out

def fetch_ohlcv_yahoo(
    symbols: Iterable[str],
    *,
    period: str = "60d",          # 60m barlar için tipik limit
    interval: str = "60m",        # 1 saatlik bar
    auto_adjust: bool = True,
    sector_map: Optional[Dict[str, str]] = None,
    suffix: str = ".IS",
) -> pd.DataFrame:
    """
    Yahoo’dan çoklu sembol 1s OHLCV indir → MultiIndex (date, symbol)
    Dönüş: ['open','high','low','close','volume','sector']
    """
    ysyms = [to_yahoo_symbol(s, suffix=suffix) for s in symbols]

    raw = yf.download(
        ysyms,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=auto_adjust,
        threads=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo indirme sonucu boş. Sembol/period/interval'i kontrol et.")

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        for t in sorted(set(raw.columns.get_level_values(0))):
            sub = raw[t].dropna(how="all")
            if sub.empty: 
                continue
            sub = _normalize_ohlcv(sub)
            sub["symbol"] = t
            sec = None
            if sector_map:
                sec = sector_map.get(t) or sector_map.get(t.replace(suffix, ""))
            sub["sector"] = sec if sec is not None else "Unknown"
            frames.append(sub[["open","high","low","close","volume","sector"]])
    else:
        sub = _normalize_ohlcv(raw)
        t = ysyms[0]
        sub["symbol"] = t
        sub["sector"] = sector_map.get(t.replace(suffix,""), "Unknown") if sector_map else "Unknown"
        frames.append(sub[["open","high","low","close","volume","sector"]])

    all_df = pd.concat(frames)
    all_df = all_df.reset_index().rename(columns={"index":"date","Date":"date"})
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.set_index(["date","symbol"]).sort_index()
    return all_df

def _convert_to_tz(idx: pd.DatetimeIndex, market_tz: str) -> pd.DatetimeIndex:
    # yfinance genelde UTC verir. TZ varsa convert, yoksa önce UTC'ye localize et, sonra convert.
    if getattr(idx, "tz", None) is None:
        return idx.tz_localize("UTC").tz_convert(market_tz)
    return idx.tz_convert(market_tz)

def resample_hourly_to_daily(df: pd.DataFrame, market_tz: str = "Europe/Istanbul") -> pd.DataFrame:
    """
    1 saatlik OHLCV → günlük OHLCV (EOD) resample.
    open=ilk, high=max, low=min, close=son, volume=toplam, sector=son
    """
    # (date,symbol) -> tz düzelt
    d0 = df.copy()
    dates = d0.index.get_level_values(0)
    new_dates = _convert_to_tz(pd.DatetimeIndex(dates), market_tz)
    d0.index = pd.MultiIndex.from_arrays([new_dates, d0.index.get_level_values(1)],
                                         names=["date","symbol"])

    s = d0.swaplevel(0,1).sort_index()  # (symbol, date)
    agg = s.groupby(level=0).resample("1D", level=1).agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum","sector":"last"
    })
    # geri dön: (date, symbol) ve tz-free tarih
    dts = agg.index.get_level_values(1).tz_localize(None)
    agg.index = pd.MultiIndex.from_arrays([agg.index.get_level_values(0), dts],
                                          names=["symbol","date"])
    daily = agg.swaplevel(0,1).sort_index()
    daily = daily.dropna(subset=["open","high","low","close"])
    return daily[["open","high","low","close","volume","sector"]]
