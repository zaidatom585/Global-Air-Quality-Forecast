"""The PM2.5 forecast model: training, honest testing against simple baselines, and what drives each city.

Model: LightGBM (gradient-boosted trees) on log(1 + PM2.5), one model for all cities and horizons; the city,
its latitude and longitude and the horizon are features. Working on the log scale makes an error of 10 in
Reykjavik (5 ug/m3) count as much as an error of 200 in Delhi (100 ug/m3).

Two tests, both on rows the model never saw:
  held-out months  every month is assigned to one of 5 groups; each group is predicted by a model trained on
                   the other four, leaving out 4 days on each side of the held-out months so that no training
                   row overlaps a test row's 72-hour window. Every city and season gets tested.
  recent           the last 90 days, predicted by a model trained only on data from before them: the closest
                   thing to running the forecast for real.

Baselines on the same rows:
  persistence      "the next days look like the last 24 hours" (the PM2.5 mean of the last 24 hours)
  cams             the official CAMS forecast for the target hours, from the forecasts saved by `update`.
                   These only exist from the day collection started, so this comparison grows over time.

Skill = 1 - (model error / persistence error): 0 means no better than persistence, 0.3 means 30% smaller
errors.

Bias correction: a model trained on the log scale predicts the typical value, which is below the average,
so it under-shoots smog peaks. Each city gets a scale factor (actual total / predicted total) learned from
test predictions on other data: for a held-out group, from the other four groups; for the recent test, from
held-out rows before it; for the final model, from all held-out rows. Both versions are scored ("model" and
"adjusted"): the adjusted one has almost no bias, while the plain one can have the smaller typical error.

What drives each city: SHAP values (each feature's contribution to each forecast, from LightGBM's own
TreeSHAP) are added up per feature group: recent PM2.5, weather, fires, CAMS, time of day and season, place.
Shown twice: over all forecasts, and over each city's worst 10% (the forecasts for its highest PM2.5), where
smoke episodes are.

Outputs (models/):
  pm25_lgbm.txt              the model trained on all rows
  model_card.json            features, settings, data range and test scores
  report/metrics.csv         scores per test, city and horizon
  report/drivers.csv         share of each feature group in each city's forecasts
  report/predictions.parquet every test prediction next to the baselines and what happened
  report/summary.md          the same in readable tables
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from aqcast.settings import Settings
from aqcast.storage import as_utc, read_frame, write_frame

log = logging.getLogger(__name__)

ID_COLUMNS = ["city", "issue_utc", "target_utc", "y"]
GROUPS = {"pm_": "recent PM2.5", "wx_": "weather", "fire_": "fires", "cams_": "CAMS", "time_": "time",
          "horizon_h": "time", "city": "place", "lat": "place", "lon": "place"}
PARAMS = dict(objective="regression", learning_rate=0.05, n_estimators=800, num_leaves=63, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
FOLDS = 5
PURGE_DAYS = 4                  # > the 72-hour horizon, so no training target overlaps a test row
RECENT_DAYS = 90
RATIO_LIMITS = (0.8, 1.5)       # a city's bias factor is kept within these
WORST_SHARE = 0.10              # "worst days": each city's highest 10% of PM2.5
SHAP_ROWS_PER_CITY = 500        # TreeSHAP is slow on big models; 500 forecasts per city are plenty


def group_of(column: str) -> str:
    for prefix, group in GROUPS.items():
        if column == prefix or column.startswith(prefix):
            return group
    return "other"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLUMNS]


def design(df: pd.DataFrame, features: list[str], cities: list[str]) -> pd.DataFrame:
    """The model's input: features as floats, the city as a category with a fixed list of cities."""
    X = df[features].copy()
    for c in features:
        if c != "city":
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    if "city" in features:
        X["city"] = pd.Categorical(df["city"], categories=cities)
    return X


def make_model(**overrides) -> lightgbm.LGBMRegressor:
    return lightgbm.LGBMRegressor(**{**PARAMS, **overrides, "random_state": 42})


def fit(train: pd.DataFrame, features: list[str], cities: list[str], **overrides) -> lightgbm.LGBMRegressor:
    model = make_model(**overrides)
    model.fit(design(train, features, cities), np.log1p(train["y"].clip(lower=0)))
    return model


def predict(model, df: pd.DataFrame, features: list[str], cities: list[str]) -> np.ndarray:
    return np.expm1(model.predict(design(df, features, cities))).clip(min=0)


# ------------------------------------------------------------------------------------------------- splits
def _purged(train_mask: pd.Series, issue: pd.Series, test_mask: pd.Series) -> pd.Series:
    """Drop training rows within PURGE_DAYS of any test row."""
    test_days = pd.DatetimeIndex(issue[test_mask].dt.floor("D").unique())
    near = set()
    for k in range(-PURGE_DAYS, PURGE_DAYS + 1):
        near.update((test_days + pd.Timedelta(days=k)).tolist())
    return train_mask & ~issue.dt.floor("D").isin(near)


def month_folds(df: pd.DataFrame, k: int = FOLDS):
    """(name, train mask, test mask) for each group of held-out months."""
    issue = df["issue_utc"]
    month = issue.dt.year * 12 + issue.dt.month
    for i in range(k):
        test = (month % k) == i
        yield f"held-out months {i + 1}/{k}", _purged(~test, issue, test), test


def recent_split(df: pd.DataFrame, days: int = RECENT_DAYS):
    issue = df["issue_utc"]
    start = issue.max() - pd.Timedelta(days=days)
    test = issue > start
    train = issue < start - pd.Timedelta(days=PURGE_DAYS)
    return "recent", train, test


# --------------------------------------------------------------------------------------- bias correction
def city_ratios(df: pd.DataFrame) -> dict[str, float]:
    """Per city: actual total / predicted total, from test predictions (kept within RATIO_LIMITS)."""
    g = df.groupby("city")[["y", "model"]].sum()
    r = (g["y"] / g["model"].where(g["model"] > 0)).clip(*RATIO_LIMITS).fillna(1.0)
    return {k: float(v) for k, v in r.items()}


def adjust(df: pd.DataFrame, ratios: dict[str, float]) -> np.ndarray:
    return df["model"].to_numpy() * df["city"].map(ratios).fillna(1.0).to_numpy()


def adjusted_predictions(preds: pd.DataFrame) -> pd.Series:
    """Bias-corrected predictions, each from ratios learned on other test rows (see module notes)."""
    out = pd.Series(np.nan, index=preds.index)
    held = preds[preds["test"] == "held-out months"]
    for fold in held["fold"].unique():
        rows = held[held["fold"] == fold]
        out[rows.index] = adjust(rows, city_ratios(held[held["fold"] != fold]))
    recent = preds[preds["test"] == "recent"]
    if len(recent):
        before = held[held["issue_utc"] < recent["issue_utc"].min() - pd.Timedelta(days=PURGE_DAYS)]
        out[recent.index] = adjust(recent, city_ratios(before) if len(before) else {})
    return out


# ----------------------------------------------------------------------------------------------- baselines
def persistence(df: pd.DataFrame) -> pd.Series:
    return df["pm_24h"].fillna(df["pm_3h"])


def cams_forecasts(settings: Settings) -> pd.DataFrame:
    """Saved CAMS forecasts: city, issued_utc, time_utc, pm2_5."""
    files = sorted((settings.data_dir / "forecasts" / "cams").glob("issued=*/*.parquet"))
    if not files:
        return pd.DataFrame(columns=["city", "issued_utc", "time_utc", "pm2_5"])
    df = pd.concat([read_frame(p, columns=["city", "issued_utc", "time_utc", "pm2_5"]) for p in files],
                   ignore_index=True)
    df["issued_utc"], df["time_utc"] = as_utc(df["issued_utc"]), as_utc(df["time_utc"])
    return df


def cams_baseline(rows: pd.DataFrame, forecasts: pd.DataFrame) -> pd.Series:
    """CAMS forecast issued at a row's issue time, averaged over the row's six target hours."""
    out = pd.Series(np.nan, index=rows.index)
    if forecasts.empty or rows.empty:
        return out
    f = forecasts.dropna(subset=["pm2_5"])
    f = f.assign(lead=((f["time_utc"] - f["issued_utc"]) / pd.Timedelta(hours=1)).round().astype(int))
    pieces = []
    for h in sorted(rows["horizon_h"].unique()):
        block = f[(f["lead"] > h - 6) & (f["lead"] <= h)]
        m = block.groupby(["city", "issued_utc"])["pm2_5"].agg(["mean", "size"]).reset_index()
        m = m[m["size"] >= 3].assign(horizon_h=h)
        pieces.append(m)
    table = pd.concat(pieces, ignore_index=True).rename(columns={"issued_utc": "issue_utc", "mean": "cams"})
    merged = rows[["city", "issue_utc", "horizon_h"]].reset_index().merge(
        table[["city", "issue_utc", "horizon_h", "cams"]], on=["city", "issue_utc", "horizon_h"], how="left")
    return pd.Series(merged["cams"].to_numpy(), index=merged["index"]).reindex(rows.index)


# ------------------------------------------------------------------------------------------------- scores
def scores(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """MAE, RMSE and bias of the model and baselines, and skill against persistence, per group."""
    def one(g: pd.DataFrame) -> pd.Series:
        e = g["model"] - g["y"]
        out = {"n": len(g), "mean_pm25": g["y"].mean(), "mae_model": e.abs().mean(),
               "rmse_model": np.sqrt((e ** 2).mean()), "bias_model": e.mean(),
               "mae_persistence": (g["persistence"] - g["y"]).abs().mean()}
        out["skill_vs_persistence"] = 1 - out["mae_model"] / out["mae_persistence"] if out["mae_persistence"] else np.nan
        if "adjusted" in g:
            a = g["adjusted"] - g["y"]
            out["mae_adjusted"], out["bias_adjusted"] = a.abs().mean(), a.mean()
            out["skill_adjusted"] = 1 - out["mae_adjusted"] / out["mae_persistence"] if out["mae_persistence"] else np.nan
        c = g.dropna(subset=["cams"])
        out["n_cams"] = len(c)
        out["mae_cams"] = (c["cams"] - c["y"]).abs().mean() if len(c) else np.nan
        out["mae_model_on_cams_rows"] = (c["model"] - c["y"]).abs().mean() if len(c) else np.nan
        return pd.Series(out)
    if not by:
        return one(df).to_frame().T
    return df.groupby(by).apply(one, include_groups=False).reset_index()


# ------------------------------------------------------------------------------------------------ drivers
def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP value of every feature for every row (log scale), from LightGBM's built-in TreeSHAP."""
    values = model.booster_.predict(X, pred_contrib=True)
    return pd.DataFrame(np.asarray(values)[:, :-1], columns=list(X.columns), index=X.index)


def driver_shares(shap: pd.DataFrame, cities: pd.Series) -> pd.DataFrame:
    """Per city, each feature group's share (%) of the total absolute SHAP value."""
    by_group = shap.abs().T.groupby(lambda c: group_of(c)).sum().T
    by_group["city"] = cities.to_numpy()
    totals = by_group.groupby("city").mean()
    shares = totals.div(totals.sum(axis=1), axis=0).mul(100).round(1)
    others = [g for g in shares.columns if g not in ("recent PM2.5", "place", "time")]
    shares["main_outside_driver"] = shares[others].idxmax(axis=1) if others else ""
    return shares.reset_index()


# ---------------------------------------------------------------------------------------------------- run
def model_dir(settings: Settings) -> Path:
    return Path(os.environ.get("AQ_MODEL_DIR", settings.data_dir.parent / "models"))


def run(settings: Settings, quick: bool = False) -> dict:
    table = read_frame(settings.data_dir / "features" / "training.parquet")
    for c in ("issue_utc", "target_utc"):
        table[c] = as_utc(table[c])
    table = table.dropna(subset=["y"]).reset_index(drop=True)
    features = feature_columns(table)
    cities = sorted(table["city"].unique())
    overrides = {"n_estimators": 60} if quick else {}
    out = model_dir(settings)
    log.info("%d rows, %d features, %d cities, %s to %s", len(table), len(features), len(cities),
             table["issue_utc"].min(), table["issue_utc"].max())

    tests = []
    for i, (name, train, test) in enumerate([*month_folds(table), recent_split(table)]):
        model = fit(table[train], features, cities, **overrides)
        rows = table[test].copy()
        rows["model"] = predict(model, rows, features, cities)
        rows["test"] = "recent" if name == "recent" else "held-out months"
        rows["fold"] = i
        tests.append(rows)
        log.info("%-22s trained on %7d rows, tested on %7d", name, int(train.sum()), int(test.sum()))
    preds = pd.concat(tests, ignore_index=True)
    preds["persistence"] = persistence(preds)
    preds["cams"] = cams_baseline(preds, cams_forecasts(settings))
    preds["adjusted"] = adjusted_predictions(preds)
    ratios = city_ratios(preds[preds["test"] == "held-out months"])
    keep = ["test", "city", "issue_utc", "horizon_h", "target_utc", "y", "model", "adjusted", "persistence", "cams"]
    preds = preds[keep]

    overall = scores(preds, ["test"])
    by_horizon = scores(preds, ["test", "horizon_h"])
    by_city = scores(preds[preds["test"] == "held-out months"], ["city"])
    metrics = pd.concat([overall.assign(level="overall"), by_horizon.assign(level="horizon"),
                         by_city.assign(level="city", test="held-out months")], ignore_index=True)

    log.info("Final model on all %d rows", len(table))
    final = fit(table, features, cities, **overrides)
    shuffled = table.sample(frac=1.0, random_state=0)
    sample = shuffled.groupby("city").head(SHAP_ROWS_PER_CITY)
    worst_cut = table.groupby("city")["y"].transform(lambda y: y.quantile(1 - WORST_SHARE))
    worst = shuffled[shuffled["y"] >= worst_cut.loc[shuffled.index]].groupby("city").head(SHAP_ROWS_PER_CITY)
    drivers = pd.concat([
        driver_shares(contributions(final, design(sample, features, cities)), sample["city"]).assign(days="all"),
        driver_shares(contributions(final, design(worst, features, cities)), worst["city"]).assign(days="worst 10%"),
    ], ignore_index=True)

    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    final.booster_.save_model(str(out / "pm25_lgbm.txt"))
    write_frame(preds, rep / "predictions.parquet")
    metrics.to_csv(rep / "metrics.csv", index=False)
    drivers.to_csv(rep / "drivers.csv", index=False)
    pd.DataFrame({"feature": features, "group": [group_of(f) for f in features],
                  "importance_gain": final.booster_.feature_importance("gain")}) \
        .sort_values("importance_gain", ascending=False).to_csv(rep / "feature_importance.csv", index=False)
    card = {
        "model": "LightGBM regressor on log(1 + PM2.5)", "params": {**PARAMS, **overrides},
        "bias_factors_by_city": ratios,
        "features": features, "cities": cities, "rows": int(len(table)),
        "data_from": str(table["issue_utc"].min()), "data_to": str(table["issue_utc"].max()),
        "scores": json.loads(overall.to_json(orient="records")),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    (rep / "summary.md").write_text(summary_markdown(overall, by_horizon, by_city, drivers), encoding="utf-8")

    for line in summary_markdown(overall, by_horizon, by_city, drivers).splitlines():
        log.info("%s", line)
    o = overall.set_index("test")
    return {"rows": int(len(table)), "features": len(features),
            "skill_held_out": round(float(o.loc["held-out months", "skill_vs_persistence"]), 3),
            "skill_recent": round(float(o.loc["recent", "skill_vs_persistence"]), 3) if "recent" in o.index else None,
            "bias_held_out": round(float(o.loc["held-out months", "bias_model"]), 2),
            "bias_adjusted_held_out": round(float(o.loc["held-out months", "bias_adjusted"]), 2),
            "model": str(out / "pm25_lgbm.txt")}


def _table(df: pd.DataFrame, cols: list[str], fmt: dict) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt[c].format(v) if c in fmt and pd.notna(v) else ("" if pd.isna(v) else str(v)))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def summary_markdown(overall, by_horizon, by_city, drivers) -> str:
    f = {"n": "{:,.0f}", "mean_pm25": "{:.1f}", "mae_model": "{:.2f}", "mae_persistence": "{:.2f}",
         "skill_vs_persistence": "{:.0%}", "n_cams": "{:,.0f}", "mae_cams": "{:.2f}", "mae_model_on_cams_rows": "{:.2f}",
         "bias_model": "{:+.2f}", "mae_adjusted": "{:.2f}", "bias_adjusted": "{:+.2f}", "skill_adjusted": "{:.0%}"}
    s = "# PM2.5 forecast: test results\n\nErrors in ug/m3 (mean absolute error). Skill = how much smaller the " \
        "model's errors are than persistence's.\n\n## Overall\n\n"
    s += _table(overall, ["test", "n", "mean_pm25", "mae_model", "mae_persistence", "skill_vs_persistence",
                          "bias_model", "n_cams", "mae_cams", "mae_model_on_cams_rows"], f)
    s += "\nWith the bias correction:\n\n" + _table(overall, ["test", "mae_adjusted", "bias_adjusted",
                                                                "skill_adjusted"], f)
    s += "\n## By horizon\n\n" + _table(by_horizon, ["test", "horizon_h", "n", "mae_model", "mae_persistence",
                                                       "skill_vs_persistence"], f)
    s += "\n## By city (held-out months)\n\n" + _table(by_city.sort_values("mean_pm25", ascending=False),
                                                      ["city", "n", "mean_pm25", "mae_model", "mae_persistence",
                                                       "skill_vs_persistence"], f)
    groups = [c for c in drivers.columns if c not in ("city", "main_outside_driver", "days")]
    for days, title in (("all", "all forecasts"), ("worst 10%", "each city's worst 10% (highest PM2.5)")):
        d = drivers[drivers["days"] == days]
        s += f"\n## What drives each city's forecast: {title} (% of SHAP)\n\n" + _table(
            d, ["city", *groups, "main_outside_driver"], {g: "{:.0f}" for g in groups})
    return s
