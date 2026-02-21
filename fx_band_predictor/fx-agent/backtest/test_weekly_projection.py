"""
Test the three-force weekly projection model with 3 scenarios.

Scenario A: Current market (real data — neutral momentum, rate above MA)
Scenario B: Mock strong UP day (lstm_delta=+0.40, xgb=UP, conf=0.75)
Scenario C: Mock strong DOWN day (lstm_delta=-0.40, xgb=DOWN, conf=0.75)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from agents.fx_prediction_agent import FXPredictionAgent


def run_scenario_mock(label, current_rate, lstm_delta, xgb_dir, xgb_conf,
                      sent_score, ma_30, rate_trend_30d):
    """Run the three-force projection with mock values (no agent needed)."""
    # Force 1: Momentum
    xgb_dir_mult = 1.0 if xgb_dir == "UP" else -1.0 if xgb_dir == "DOWN" else 0.0
    momentum_per_day = (
        lstm_delta * 0.60
        + xgb_dir_mult * abs(lstm_delta) * 0.30 * xgb_conf
        + (-sent_score) * 0.05
    )

    # Force 2: Mean reversion
    reversion_gap = ma_30 - current_rate

    # Force 3: Trend drift
    capped_trend = min(0.10, max(-0.10, rate_trend_30d))

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  current_rate={current_rate:.4f}, ma_30={ma_30:.4f}")
    print(f"  lstm_delta={lstm_delta:+.4f}, xgb={xgb_dir}({xgb_conf:.2f}), sent_score={sent_score:+.2f}")
    print(f"  momentum/day={momentum_per_day:+.4f}, reversion_gap={reversion_gap:+.4f}, "
          f"trend_30d={rate_trend_30d:+.6f}")
    print()
    print(f"  {'Day':<5s} | {'Mom.wt':>6s} | {'Momentum':>9s} | {'Reversion':>9s} | "
          f"{'Trend':>7s} | {'Total':>7s} | {'most_likely':>11s} | {'Range'}")
    print(f"  {'─'*5}─┼─{'─'*6}─┼─{'─'*9}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*11}─┼─{'─'*20}")

    prev_ml = current_rate
    for day_offset in range(1, 8):
        momentum_weight = max(0.20, 1.0 - (day_offset - 1) * 0.13)
        momentum_contribution = momentum_per_day * day_offset * momentum_weight
        reversion_weight = min(0.40, day_offset * 0.06)
        reversion_contribution = reversion_gap * reversion_weight
        trend_contribution = capped_trend * day_offset

        total = momentum_contribution + reversion_contribution + trend_contribution
        total = max(-2.0, min(2.0, total))
        most_likely = round(current_rate + total, 4)

        # Range
        base_range = 0.20
        daily_expansion = 0.07
        range_buffer = base_range + (day_offset * daily_expansion)
        if momentum_per_day > 0.05:
            rng_lo = round(most_likely - range_buffer * 0.7, 4)
            rng_hi = round(most_likely + range_buffer * 1.3, 4)
        elif momentum_per_day < -0.05:
            rng_lo = round(most_likely - range_buffer * 1.3, 4)
            rng_hi = round(most_likely + range_buffer * 0.7, 4)
        else:
            rng_lo = round(most_likely - range_buffer, 4)
            rng_hi = round(most_likely + range_buffer, 4)

        day_names = ["", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        name = day_names[day_offset] if day_offset <= 7 else f"D{day_offset}"

        delta = most_likely - prev_ml
        print(f"  {name:<5s} | {momentum_weight:6.2f} | {momentum_contribution:+9.4f} | "
              f"{reversion_contribution:+9.4f} | {trend_contribution:+7.4f} | "
              f"{total:+7.4f} | {most_likely:11.4f} | [{rng_lo:.2f}, {rng_hi:.2f}]")
        prev_ml = most_likely

    print()


# =========================================================================
# Scenario A: Current market (run real agent)
# =========================================================================
print("\n" + "="*72)
print("  SCENARIO A — Current Market (live agent)")
print("="*72)

df = pd.read_csv("data/market_data.csv", parse_dates=["date"])
agent = FXPredictionAgent()
result = agent.predict_weekly(df)

print("\n  Summary:")
for f in result["daily_forecasts"]:
    if f["is_trading_day"]:
        p = f["prediction"]
        print(f"    {f['day_name']:>9s} {f['date']}: most_likely={p['most_likely']:8.4f}  "
              f"range=[{p['range_low']:.2f}, {p['range_high']:.2f}]  dir={p['direction']}")


# =========================================================================
# Scenario B: Strong UP day (mock)
# =========================================================================
run_scenario_mock(
    label="SCENARIO B — Strong UP (lstm_delta=+0.40, xgb=UP@0.75)",
    current_rate=90.79,
    lstm_delta=+0.40,
    xgb_dir="UP",
    xgb_conf=0.75,
    sent_score=-0.20,   # mild bearish sentiment (rate UP = INR weaker)
    ma_30=89.50,         # rate well above MA → reversion pulls DOWN
    rate_trend_30d=0.02, # uptrend: 2 paise/day
)


# =========================================================================
# Scenario C: Strong DOWN day (mock)
# =========================================================================
run_scenario_mock(
    label="SCENARIO C — Strong DOWN (lstm_delta=-0.40, xgb=DOWN@0.75)",
    current_rate=90.79,
    lstm_delta=-0.40,
    xgb_dir="DOWN",
    xgb_conf=0.75,
    sent_score=+0.30,    # bullish INR sentiment (rate DOWN = INR stronger)
    ma_30=89.50,          # rate above MA → reversion also pulls DOWN
    rate_trend_30d=-0.01, # downtrend: -1 paisa/day
)


# =========================================================================
# Validation checks
# =========================================================================
print("\n" + "="*72)
print("  VALIDATION CHECKS")
print("="*72)

# Check Scenario A from live agent
forecasts = [f for f in result["daily_forecasts"] if f["is_trading_day"]]
mls = [f["prediction"]["most_likely"] for f in forecasts]

# 1. All different
all_different = len(set(mls)) == len(mls)
print(f"\n  1. most_likely all DIFFERENT:  {'PASS' if all_different else 'FAIL'} — {mls}")

# 2. Monotonically reasonable (no jumps > 0.5 between adjacent days)
max_jump = max(abs(mls[i+1] - mls[i]) for i in range(len(mls)-1)) if len(mls) > 1 else 0
reasonable = max_jump < 0.50
print(f"  2. No sudden jumps (max={max_jump:.4f}): {'PASS' if reasonable else 'FAIL'}")

# 3. Within ±2.0 of current rate
rate = result["current_rate"]
within_bounds = all(abs(ml - rate) <= 2.0 for ml in mls)
print(f"  3. All within ±2.0 of {rate:.2f}: {'PASS' if within_bounds else 'FAIL'}")

# 4 & 5 verified visually from scenarios B and C above
print(f"  4. Scenario B: verify UP momentum → increasing early, stabilizing late")
print(f"  5. Scenario A: verify reversion pulls DOWN (rate {rate:.2f} above MA)")

print("\n--- Test complete ---")
