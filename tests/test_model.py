"""Model splits, baselines, scores and drivers, on a small made-up training table."""
import numpy as np
import pandas as pd

from aqcast import model

T0 = pd.Timestamp("2025-01-01", tz="UTC")


def table(days=400, cities=("a", "b")):
    rng = np.random.default_rng(0)
    rows = []
    for city in cities:
        issue = pd.date_range(T0, periods=days * 4, freq="6h", tz="UTC")
        base = 20 + 10 * np.sin(np.arange(len(issue)) / 120) + (30 if city == "a" else 0)
        for h in (24, 48, 72):
            fire = rng.gamma(1.0, 50, len(issue))
            df = pd.DataFrame({"city": city, "issue_utc": issue, "horizon_h": h,
                               "target_utc": issue + pd.Timedelta(hours=h),
                               "pm_3h": base + rng.normal(0, 2, len(issue)), "pm_24h": base,
                               "wx_wind": rng.uniform(0, 8, len(issue)), "fire_upwind_24h_r100": fire,
                               "cams_pm25": base * 0.8, "time_season_sin": np.sin(np.arange(len(issue)) / 240),
                               "lat": 10.0, "lon": 20.0})
            df["y"] = base + 0.05 * fire - 1.5 * df["wx_wind"] + rng.normal(0, 2, len(issue))
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_feature_groups():
    assert model.group_of("pm_24h") == "recent PM2.5"
    assert model.group_of("fire_upwind_24h_r100") == "fires"
    assert model.group_of("wx_target_wind_u") == "weather"
    assert model.group_of("horizon_h") == "time" and model.group_of("city") == "place"
    assert "y" not in model.feature_columns(table(5)) and "issue_utc" not in model.feature_columns(table(5))


def test_held_out_months_are_purged_and_cover_everything():
    df = table(400)
    tested = pd.Series(False, index=df.index)
    for name, train, test in model.month_folds(df):
        assert not (train & test).any()
        tr = np.unique(df.loc[train, "issue_utc"].dt.tz_localize(None).to_numpy())
        te = np.unique(df.loc[test, "issue_utc"].dt.tz_localize(None).to_numpy())
        gap = np.abs(tr[:, None] - te[None, :]).min()
        assert pd.Timedelta(gap) > pd.Timedelta(days=model.PURGE_DAYS)
        tested |= test
    assert tested.all()


def test_recent_split_trains_only_on_the_past():
    df = table(400)
    _, train, test = model.recent_split(df)
    assert df.loc[train, "issue_utc"].max() + pd.Timedelta(days=model.PURGE_DAYS) < df.loc[test, "issue_utc"].min()
    assert df.loc[test, "issue_utc"].max() == df["issue_utc"].max()


def test_cams_baseline_averages_the_six_target_hours():
    rows = pd.DataFrame({"city": ["a", "a"], "issue_utc": [T0, T0], "horizon_h": [24, 48]})
    times = pd.date_range(T0, periods=80, freq="h", tz="UTC")
    fc = pd.DataFrame({"city": "a", "issued_utc": T0, "time_utc": times,
                       "pm2_5": np.arange(80, dtype=float)})           # value = lead hours
    out = model.cams_baseline(rows, fc)
    assert out.tolist() == [np.mean(range(19, 25)), np.mean(range(43, 49))]
    other = pd.DataFrame({"city": ["b"], "issue_utc": [T0], "horizon_h": [24]})
    assert model.cams_baseline(other, fc).isna().all()


def test_scores_and_skill():
    df = pd.DataFrame({"test": "t", "y": [10.0, 20.0], "model": [12.0, 18.0], "persistence": [14.0, 16.0],
                       "cams": [np.nan, 25.0]})
    s = model.scores(df, ["test"]).iloc[0]
    assert s["mae_model"] == 2.0 and s["mae_persistence"] == 4.0 and s["skill_vs_persistence"] == 0.5
    assert s["n_cams"] == 1 and s["mae_cams"] == 5.0 and s["mae_model_on_cams_rows"] == 2.0


def test_driver_shares_add_up():
    shap = pd.DataFrame({"pm_24h": [1.0, -1.0], "fire_upwind_24h_r100": [3.0, 0.0], "wx_wind": [0.0, 1.0]})
    out = model.driver_shares(shap, pd.Series(["a", "b"])).set_index("city")
    assert out.loc["a", "fires"] == 75.0 and out.loc["a", "main_outside_driver"] == "fires"
    assert out.loc["b", "main_outside_driver"] == "weather"


def test_model_learns_and_beats_persistence():
    df = table(300)
    features, cities = model.feature_columns(df), sorted(df["city"].unique())
    _, train, test = model.recent_split(df, days=60)
    m = model.fit(df[train], features, cities, n_estimators=150)
    pred = model.predict(m, df[test], features, cities)
    mae = np.abs(pred - df.loc[test, "y"]).mean()
    mae_p = np.abs(model.persistence(df[test]) - df.loc[test, "y"]).mean()
    assert mae < mae_p
    shap = model.contributions(m, model.design(df[test].head(50), features, cities))
    assert list(shap.columns) == features and len(shap) == 50


def test_bias_factors_come_from_other_rows():
    preds = pd.DataFrame({
        "test": ["held-out months"] * 4 + ["recent"],
        "fold": [0, 0, 1, 1, 5], "city": "a",
        "issue_utc": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-03-01", "2025-03-02", "2025-06-01"], utc=True),
        "y": [12.0, 12.0, 11.0, 11.0, 30.0], "model": [10.0, 10.0, 10.0, 10.0, 20.0]})
    out = model.adjusted_predictions(preds)
    assert out.iloc[0] == 10.0 * 1.1          # fold 0 uses fold 1's ratio (22/20)
    assert out.iloc[2] == 10.0 * 1.2          # fold 1 uses fold 0's ratio (24/20)
    assert np.isclose(out.iloc[4], 20.0 * 46 / 40)   # recent uses all earlier held-out rows
    assert model.city_ratios(pd.DataFrame({"city": ["b"], "y": [100.0], "model": [10.0]}))["b"] == model.RATIO_LIMITS[1]
