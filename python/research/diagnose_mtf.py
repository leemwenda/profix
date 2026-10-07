from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profx.config import load_config
from profx.mtf import load_mtf
from profx.mtf_backtest import build_mtf_signals, summarize_records
from profx.engine import MarketData, BacktestEngine


ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "config/xauusd.toml")
DATA_DIR = ROOT / "python/data/mtf"


def money(x):
    return f"${x:,.2f}"


def pf(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    gp = series[series > 0].sum()
    gl = abs(series[series < 0].sum())
    return gp / gl if gl else float("inf")


def stats(df, label=""):
    if df.empty:
        print(f"{label}: NO TRADES")
        return

    p = pd.to_numeric(df["pnl"], errors="coerce").dropna()

    print(
        f"{label:<18} "
        f"trades={len(df):3d} "
        f"P/L={money(p.sum()):>12} "
        f"PF={pf(p):>6.3f} "
        f"win={(p > 0).mean()*100:6.2f}% "
        f"avg={money(p.mean()):>10}"
    )


print("=" * 78)
print("PROFX MTF ROOT-CAUSE DIAGNOSTIC")
print("=" * 78)

print("\n[1] Loading data...")
mtf = load_mtf(DATA_DIR, "XAUUSD")

print("H1 :", len(mtf.frames["H1"]))
print("M15:", len(mtf.frames["M15"]))
print("M5 :", len(mtf.frames["M5"]))
print("M1 :", len(mtf.frames["M1"]))

print("\n[2] Building MTF decisions...")
signals, records = build_mtf_signals(
    mtf=mtf,
    cfg=CFG,
    symbol="XAUUSD",
)

print(summarize_records(records))

print("\n[3] Running MTF backtest...")

market = MarketData(
    {"XAUUSD": mtf.frames["H1"].copy()},
    htf_hours=CFG.htf_hours,
    htf_offset=CFG.htf_offset_hours,
)

result = BacktestEngine(
    market,
    CFG,
    signal_override=signals,
).run()

final_equity = float(result.equity.iloc[-1])

print(f"Initial equity : {money(result.initial_balance)}")
print(f"Final equity   : {money(final_equity)}")
print(f"Net P/L        : {money(final_equity - result.initial_balance)}")
print(
    f"Return         : "
    f"{(final_equity / result.initial_balance - 1) * 100:.2f}%"
)
print(f"Trades         : {len(result.trades)}")
print(f"Halted         : {result.halted}")
print(f"Halted at      : {result.halted_at}")
print(f"Rejections     : {result.rejections}")

trades = pd.DataFrame(result.trades)

if trades.empty:
    print("\nNo trades were generated.")
    raise SystemExit(0)

print("\n[4] Available trade fields:")
print(", ".join(trades.columns))

# Normalize timestamps.
for col in ["entry_time", "exit_time"]:
    if col in trades.columns:
        trades[col] = pd.to_datetime(
            trades[col],
            utc=True,
            errors="coerce",
        )

# ------------------------------------------------------------------
# BASIC PERFORMANCE
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("OVERALL PERFORMANCE")
print("=" * 78)

stats(trades, "ALL")

if "side" in trades.columns:
    print("\nLONG / SHORT")
    for side, name in [(1, "LONG"), (-1, "SHORT")]:
        stats(trades[trades["side"] == side], name)

# ------------------------------------------------------------------
# YEAR
# ------------------------------------------------------------------

if "entry_time" in trades.columns:
    trades["year"] = trades["entry_time"].dt.year

    print("\nYEARLY PERFORMANCE")
    for year, group in trades.groupby("year"):
        stats(group, str(year))

# ------------------------------------------------------------------
# DECISION RECORD MAP
# ------------------------------------------------------------------

decision_df = pd.DataFrame(
    [
        {
            "decision_time": r.decision_time,
            "signal_time": r.signal_time,
            "decision": r.decision,
        }
        for r in records
    ]
)

if not decision_df.empty:
    decision_df["decision_time"] = pd.to_datetime(
        decision_df["decision_time"],
        utc=True,
        errors="coerce",
    )

    # Inspect actual DecisionAssessment object if available.
    # Rebuild a compact lookup from the record objects.
    rows = []

    for r in records:
        d = r.decision

        row = {
            "decision_time": r.decision_time,
            "signal_time": r.signal_time,
        }

        for attr in [
            "direction",
            "score",
            "tier",
            "regime_score",
            "setup_score",
            "volatility_score",
            "spread_score",
            "session_score",
            "executable",
        ]:
            row[attr] = getattr(d, attr, None)

        rows.append(row)

    decisions = pd.DataFrame(rows)

    if not decisions.empty:
        decisions["decision_time"] = pd.to_datetime(
            decisions["decision_time"],
            utc=True,
            errors="coerce",
        )

        print("\n" + "=" * 78)
        print("DECISION QUALITY")
        print("=" * 78)

        print("\nBy tier:")
        print(
            decisions.groupby("tier", dropna=False)
            .size()
            .sort_values(ascending=False)
            .to_string()
        )

        # Attach decision information to each trade.
        if "entry_time" in trades.columns:
            # Existing engine enters one H1 bar after the signal.
            trades["decision_time"] = trades["entry_time"]

            decision_lookup = decisions.drop_duplicates(
                "decision_time"
            ).copy()

            trades = trades.merge(
                decision_lookup,
                on="decision_time",
                how="left",
                suffixes=("", "_decision"),
            )

            print("\nTrade performance by tier:")
            for tier, group in trades.groupby(
                "tier",
                dropna=False,
            ):
                stats(group, str(tier))

            print("\nTrade performance by direction:")
            for direction, group in trades.groupby(
                "direction",
                dropna=False,
            ):
                stats(group, str(direction))

            print("\nTrade performance by regime score:")
            bins = [-np.inf, 50, 60, 70, 80, 90, np.inf]
            labels = [
                "<=50",
                "51-60",
                "61-70",
                "71-80",
                "81-90",
                ">90",
            ]

            trades["regime_band"] = pd.cut(
                pd.to_numeric(
                    trades["regime_score"],
                    errors="coerce",
                ),
                bins=bins,
                labels=labels,
            )

            for band, group in trades.groupby(
                "regime_band",
                observed=False,
                dropna=False,
            ):
                stats(group, str(band))

            print("\nTrade performance by setup score:")
            setup_values = pd.to_numeric(
                trades["setup_score"],
                errors="coerce",
            )

            trades["setup_band"] = pd.cut(
                setup_values,
                bins=[-np.inf, 15, 20, 25, np.inf],
                labels=["<15", "15-20", "20-25", "25+"],
            )

            for band, group in trades.groupby(
                "setup_band",
                observed=False,
                dropna=False,
            ):
                stats(group, str(band))

            print("\nTrade performance by total decision score:")

            score_values = pd.to_numeric(
                trades["score"],
                errors="coerce",
            )

            trades["score_band"] = pd.cut(
                score_values,
                bins=[-np.inf, 64, 74, 84, np.inf],
                labels=[
                    "<65",
                    "65-74",
                    "75-84",
                    "85+",
                ],
            )

            for band, group in trades.groupby(
                "score_band",
                observed=False,
                dropna=False,
            ):
                stats(group, str(band))

            # ------------------------------------------------------
            # YEAR + SCORE
            # ------------------------------------------------------

            print("\n2022 / 2023 by score band:")

            if "year" in trades.columns:
                for year in sorted(trades["year"].dropna().unique()):
                    print(f"\nYEAR {int(year)}")

                    year_df = trades[
                        trades["year"] == year
                    ]

                    for band, group in year_df.groupby(
                        "score_band",
                        observed=False,
                        dropna=False,
                    ):
                        stats(group, str(band))

            # ------------------------------------------------------
            # WINNERS / LOSERS CHARACTERISTICS
            # ------------------------------------------------------

            print("\n" + "=" * 78)
            print("WINNERS VS LOSERS DECISION CHARACTERISTICS")
            print("=" * 78)

            numeric_fields = [
                "score",
                "regime_score",
                "setup_score",
                "volatility_score",
                "spread_score",
                "session_score",
            ]

            for field in numeric_fields:
                if field not in trades.columns:
                    continue

                values = pd.to_numeric(
                    trades[field],
                    errors="coerce",
                )

                win_mean = values[
                    pd.to_numeric(trades["pnl"], errors="coerce") > 0
                ].mean()

                loss_mean = values[
                    pd.to_numeric(trades["pnl"], errors="coerce") < 0
                ].mean()

                print(
                    f"{field:<20} "
                    f"winner_mean={win_mean:8.3f} "
                    f"loser_mean={loss_mean:8.3f}"
                )

# ------------------------------------------------------------------
# WORST / BEST TRADES
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("WORST 15 TRADES")
print("=" * 78)

wanted = [
    "entry_time",
    "exit_time",
    "side",
    "entry_price",
    "exit_price",
    "pnl",
    "score",
    "tier",
    "regime_score",
    "setup_score",
    "volatility_score",
    "spread_score",
    "session_score",
    "year",
]

available = [c for c in wanted if c in trades.columns]

print(
    trades.sort_values("pnl")
    .loc[:, available]
    .head(15)
    .to_string(index=False)
)

print("\n" + "=" * 78)
print("BEST 15 TRADES")
print("=" * 78)

print(
    trades.sort_values("pnl", ascending=False)
    .loc[:, available]
    .head(15)
    .to_string(index=False)
)

# ------------------------------------------------------------------
# EXIT / HOLDING ANALYSIS
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("EXIT / HOLDING ANALYSIS")
print("=" * 78)

for col in [
    "exit_reason",
    "reason",
    "exit_type",
    "outcome",
]:
    if col in trades.columns:
        print(f"\n{col}:")
        print(trades[col].value_counts(dropna=False).to_string())

if "entry_time" in trades.columns and "exit_time" in trades.columns:
    trades["hours_held"] = (
        trades["exit_time"] - trades["entry_time"]
    ).dt.total_seconds() / 3600

    print("\nAverage hours held:")
    print(trades["hours_held"].describe().to_string())

# ------------------------------------------------------------------
# ENGINE SIGNAL REJECTIONS
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("ENGINE REJECTIONS")
print("=" * 78)
print(result.rejections)

# ------------------------------------------------------------------
# SAVE DIAGNOSTIC CSV
# ------------------------------------------------------------------

out = ROOT / "python/out"
out.mkdir(parents=True, exist_ok=True)

csv_path = out / "mtf_trade_diagnostic.csv"
trades.to_csv(csv_path, index=False)

print("\nSaved:")
print(csv_path)

print("\n" + "=" * 78)
print("DIAGNOSTIC COMPLETE")
print("=" * 78)
