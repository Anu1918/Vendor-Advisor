"""Deterministic vendor scoring: no AI involved, so results are reproducible and auditable."""
import pandas as pd

REQUIRED = ["vendor", "cost_per_complete_usd", "quality_score",
            "lead_time_days", "on_time_delivery_pct"]
# criterion -> (column, higher_is_better)
CRITERIA = {
    "Cost": ("cost_per_complete_usd", False),
    "Quality": ("quality_score", True),
    "Lead time": ("lead_time_days", False),
    "Reliability": ("on_time_delivery_pct", True),
}
RANGES = {  # sanity ranges used by validation
    "cost_per_complete_usd": (0, 10000),
    "quality_score": (0, 10),
    "lead_time_days": (0, 365),
    "on_time_delivery_pct": (0, 100),
}


def validate(df: pd.DataFrame):
    """Return (clean_df, errors, warnings). Never raises on bad user data."""
    errors, warnings = [], []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return df, [f"Missing required column(s): {', '.join(missing)}"], warnings
    df = df[REQUIRED].copy()
    df["vendor"] = df["vendor"].fillna("").astype(str).str.strip()
    for col in REQUIRED[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[df[REQUIRED].isna().any(axis=1) | (df["vendor"] == "")]
    if len(bad):
        warnings.append(f"Dropped {len(bad)} row(s) with missing or non-numeric values: "
                        + ", ".join(bad["vendor"].replace("", "(blank)").tolist()))
        df = df.drop(bad.index)
    dup = df[df["vendor"].duplicated(keep="first")]
    if len(dup):
        warnings.append("Dropped duplicate vendor name(s): " + ", ".join(dup["vendor"]))
        df = df.drop(dup.index)
    for col, (lo, hi) in RANGES.items():
        out = df[(df[col] < lo) | (df[col] > hi)]
        if len(out):
            warnings.append(f"Dropped {len(out)} row(s) with {col} outside {lo}-{hi}: "
                            + ", ".join(out["vendor"]))
            df = df.drop(out.index)
    if len(df) < 2:
        errors.append("Need at least 2 valid vendors to compare.")
    return df.reset_index(drop=True), errors, warnings


def _normalise(series: pd.Series, higher_is_better: bool) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:                      # all vendors identical on this criterion
        return pd.Series(1.0, index=series.index)
    n = (series - lo) / (hi - lo)
    return n if higher_is_better else 1 - n


def score(df: pd.DataFrame, weights: dict, max_cost=None, min_quality=None, max_lead=None):
    """Weighted min-max score, 0-100. Returns ranked DataFrame incl. per-criterion contributions."""
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("At least one weight must be above zero.")
    out = df.copy()
    for name, (col, hib) in CRITERIA.items():
        out[f"n_{name}"] = _normalise(out[col], hib)
        out[f"c_{name}"] = out[f"n_{name}"] * weights[name] / total_w * 100
    out["score"] = out[[f"c_{n}" for n in CRITERIA]].sum(axis=1).round(1)
    out["eligible"] = True
    out["excluded_reason"] = ""
    if max_cost is not None:
        m = out["cost_per_complete_usd"] > max_cost
        out.loc[m, ["eligible", "excluded_reason"]] = [False, f"cost above {max_cost}"]
    if min_quality is not None:
        m = out["quality_score"] < min_quality
        out.loc[m, ["eligible", "excluded_reason"]] = [False, f"quality below {min_quality}"]
    if max_lead is not None:
        m = out["lead_time_days"] > max_lead
        out.loc[m, ["eligible", "excluded_reason"]] = [False, f"lead time above {max_lead} days"]
    out = out.sort_values(["eligible", "score"], ascending=[False, False]).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    out.loc[~out["eligible"], "rank"] = pd.NA
    return out


def stability(df, weights, step=10, **constraints):
    """Nudge each weight by +/-step points; report how often the #1 vendor stays #1."""
    base = score(df, weights, **constraints)
    base_el = base[base["eligible"]]
    if base_el.empty:
        return None, 0, 0, []
    top = base_el.iloc[0]["vendor"]
    same, total, flips = 0, 0, []
    for k in weights:
        for d in (-step, step):
            w = dict(weights)
            w[k] = max(0, w[k] + d)
            if sum(w.values()) <= 0:
                continue
            r = score(df, w, **constraints)
            r = r[r["eligible"]]
            total += 1
            if r.iloc[0]["vendor"] == top:
                same += 1
            else:
                flips.append((k, d, r.iloc[0]["vendor"]))
    return top, same, total, flips


def expert_flags(ranked: pd.DataFrame):
    """Rule-of-thumb sanity checks a procurement expert would apply to the #1 pick."""
    el = ranked[ranked["eligible"]]
    flags = []
    if el.empty:
        return ["No vendor passes your hard constraints - relax them."]
    top = el.iloc[0]
    for name, (col, hib) in CRITERIA.items():
        if top[f"n_{name}"] == 0 and len(el) > 1:
            flags.append(f"Top pick is the WORST of all eligible vendors on {name}.")
    if top["on_time_delivery_pct"] < 85:
        flags.append("Top pick delivers on time less than 85% of the time - risky for deadline work.")
    if len(el) > 1 and el.iloc[0]["score"] - el.iloc[1]["score"] < 3:
        flags.append(f"Top two vendors are within 3 points ({el.iloc[0]['vendor']} vs "
                     f"{el.iloc[1]['vendor']}) - treat as a tie and look at qualitative factors.")
    return flags
