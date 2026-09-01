from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

def preprocess_player_data(df):
    """Preprocess the player data for modeling"""
    
    # 1. Sort and filter
    df = df.sort_values(["player_id", "date"])
    df = df[     # Keep rows where team_id matches t1 or t2 OR where both t1 and t2 are missing
        (df["team_id"] == df["t1"]) |
        (df["team_id"] == df["t2"]) |
        (df["t1"].isna() & df["t2"].isna())
    ]

    # Convert date columns to datetime
    df["date"] = pd.to_datetime(df["date"])
    df["md"] = pd.to_datetime(df["md"])

    # 2. Date and matchday calculations 
    df["next_day"] = df.groupby("player_id")["date"].shift(-1) 
    df["next_md"] = df.groupby("player_id")["md"].transform(
        lambda x: x.shift(-1).where(x.shift(-1) != x).bfill()
    )
    df["days_to_next"] = (df["next_md"] - df["date"]).dt.days

    # 3. Next day market value
    df["mv_next_day"] = df.groupby("player_id")["mv"].shift(-1)
    df["mv_target"] = df["mv_next_day"] - df["mv"]
    df = df[df["mv"] != 0.0]

    # 4. Feature engineering 
    # Market value trend 1d
    df["mv_change_1d"] = df["mv"] - df.groupby("player_id")["mv"].shift(1)
    df["mv_trend_1d"] = df.groupby("player_id")["mv"].pct_change(fill_method=None)
    df["mv_trend_1d"] = df["mv_trend_1d"].replace([np.inf, -np.inf], 0).fillna(0)

    # Market value trend 3d
    df["mv_change_3d"] = df["mv"] - df.groupby("player_id")["mv"].shift(3)
    df["mv_vol_3d"] = df.groupby("player_id")["mv"].rolling(3).std().reset_index(0,drop=True)

    # Market value trend 7d. Kickbase's app displays the absolute change over
    # seven daily MV points; keep both the absolute and relative variants.
    mv_7d_ago = df.groupby("player_id")["mv"].shift(7)
    df["mv_change_7d"] = df["mv"] - mv_7d_ago
    df["mv_trend_7d"] = df["mv_change_7d"] / mv_7d_ago
    df["mv_trend_7d"] = df["mv_trend_7d"].replace([np.inf, -np.inf], 0).fillna(0)
    df["mv_avg_daily_change_7d"] = df["mv_change_7d"] / 7

    ## League-wide market context
    df["market_divergence"] = (df["mv"] / df.groupby("md")["mv"].transform("mean")).rolling(3).mean()

    # 5. Clip outliers in mv_target
    Q1 = df["mv_target"].quantile(0.25)
    Q3 = df["mv_target"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR

    df["mv_target_clipped"] = df["mv_target"].clip(lower_bound, upper_bound)

    # 6. Fill missing values
    df = df.fillna({
        "market_divergence": 1,
        "mv_change_3d": 0,
        "mv_change_7d": 0,
        "mv_avg_daily_change_7d": 0,
        "mv_vol_3d": 0,
        "p": 0,
        "ppm": 0,
        "won": -1
    })

    # 7. Cutout todays values and store them
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    cutoff_time = now.replace(hour=22, minute=15, second=0, microsecond=0)
    max_date = (now - timedelta(days=1)) if now <= cutoff_time else now
    max_date = max_date.date()

    today_df = df[df["date"].dt.date >= max_date]

    # Drop those values from today from df
    df = df[df["date"].dt.date < max_date]

    # 8. Drop rows with NaN in critical columns
    df = df.dropna(subset=["mv_change_1d", "next_day", "next_md", "days_to_next", "mv_next_day", "mv_target", "mv_target_clipped"])

    return df, today_df


def split_data(df, features, target):
    """Split the data into training and testing sets based on date to avoid data leakage"""

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * 0.75)
    split_date = df["date"].iloc[split_idx]

    # Split by time, to avoid data leakage
    train = df[df["date"] < split_date]
    test = df[(df["date"] >= split_date)]

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    return X_train, X_test, y_train, y_test
