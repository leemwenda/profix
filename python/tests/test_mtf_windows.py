import pandas as pd

from profx.mtf import MTFData
from profx.mtf_windows import build_research_windows


def frame(start, end, freq):
    idx = pd.date_range(
        start=start,
        end=end,
        freq=freq,
        tz="UTC",
    )

    values = range(1000, 1000 + len(idx))

    return pd.DataFrame(
        {
            "open": values,
            "high": [x + 1 for x in values],
            "low": values,
            "close": values,
        },
        index=idx,
    )


def test_builds_progressive_mtf_windows():
    mtf = MTFData(
        {
            "H1": frame(
                "2022-01-03",
                "2026-10-06",
                "1h",
            ),
            "M15": frame(
                "2022-07-05",
                "2026-10-06",
                "15min",
            ),
            "M5": frame(
                "2025-04-30",
                "2026-10-06",
                "5min",
            ),
            "M1": frame(
                "2026-06-22",
                "2026-10-06",
                "1min",
            ),
        }
    )

    windows = build_research_windows(mtf)

    assert len(windows) == 4

    assert windows[0].timeframes == ("H1",)
    assert windows[1].timeframes == ("H1", "M15")
    assert windows[2].timeframes == ("H1", "M15", "M5")
    assert windows[3].timeframes == (
        "H1",
        "M15",
        "M5",
        "M1",
    )


def test_no_lower_timeframe_is_fabricated():
    mtf = MTFData(
        {
            "H1": frame("2022-01-03", "2026-10-06", "1h"),
            "M15": frame("2022-07-05", "2026-10-06", "15min"),
            "M5": frame("2025-04-30", "2026-10-06", "5min"),
            "M1": frame("2026-06-22", "2026-10-06", "1min"),
        }
    )

    windows = build_research_windows(mtf)

    before_m15 = windows[0]
    before_m5 = windows[1]
    before_m1 = windows[2]

    assert "M15" not in before_m15.timeframes
    assert "M5" not in before_m15.timeframes
    assert "M1" not in before_m15.timeframes

    assert "M5" not in before_m5.timeframes
    assert "M1" not in before_m5.timeframes

    assert "M1" not in before_m1.timeframes
    assert "M1" in windows[3].timeframes
