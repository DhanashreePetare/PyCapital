"""Offline tests for pyequity.processing.normalize (no network needed)."""
import logging

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyequity.processing.normalize import ValidationReport, normalize, validate
from pyequity.schema import SCHEMAS, empty_frame

LOGGER = "pyequity.processing.normalize"
DATES = ["2026-09-01", "2026-09-02", "2026-09-03"]


def _typed(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Cast a hand-built frame to the schema dtypes (test helper only)."""
    out = df.copy()
    for col, dtype in SCHEMAS[dataset].items():
        out[col] = out[col].astype(dtype)
    return out[list(SCHEMAS[dataset])]


def make(dataset: str) -> pd.DataFrame:
    """A small, valid, schema-typed frame for each dataset."""
    d = pd.to_datetime(DATES)
    if dataset == "equity":
        raw = pd.DataFrame({
            "date": d, "open": [100.0, 101.0, 102.0], "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0], "close": [104.0, 105.0, 106.0],
            "volume": [1000, 2000, 3000], "trades": [10, 20, 30],
            "symbol": "TCS", "source": "NSE"})
    elif dataset == "index":
        raw = pd.DataFrame({
            "date": d, "index_name": "NIFTY 50", "open": [25000.0, 25100.0, 25200.0],
            "high": [25200.0, 25300.0, 25400.0], "low": [24900.0, 25000.0, 25100.0],
            "close": [25100.0, 25200.0, 25300.0], "volume": [5, 6, 7], "source": "NSE"})
    elif dataset == "fno":
        raw = pd.DataFrame({
            "date": d, "symbol": "RELIANCE", "instrument_type": "FUTSTK",
            "expiry_date": pd.Timestamp("2026-10-27"), "option_type": "XX",
            "strike_price": 0.0, "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0], "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0], "settle_price": [104.0, 105.0, 106.0],
            "volume": [10, 20, 30], "value": [1000, 2000, 3000],
            "open_interest": [500, 600, 700], "change_in_oi": [5, 100, -50],
            "underlying_value": [103.0, 104.0, 105.0], "source": "NSE_FO"})
    elif dataset == "corporate_actions":
        raw = pd.DataFrame({
            "ex_date": d, "symbol": "RELIANCE",
            "action_type": ["DIVIDEND", "BONUS", "SPLIT"],
            "action_detail": ["Dividend - Rs 5", "Bonus 1:1", "Split Rs 10 to Rs 5"],
            "record_date": d, "source": "NSE"})
    elif dataset == "nav":
        raw = pd.DataFrame({
            "date": d, "scheme_code": 119551, "scheme_name": "Some Fund",
            "nav": [10.0, 10.1, 10.2], "source": "AMFI"})
    elif dataset == "bulk_deals":
        raw = pd.DataFrame({
            "date": d, "symbol": ["AAA", "BBB", "CCC"],
            "scrip_name": ["A Ltd", "B Ltd", "C Ltd"], "client_name": "Client X",
            "deal_type": ["BUY", "SELL", "BUY"], "quantity": [100, 200, 300],
            "price": [10.0, 20.0, 30.0], "source": "SEBI_NSE"})
    else:
        raise AssertionError(dataset)
    return _typed(raw, dataset)


ALL = list(SCHEMAS)


# ----------------------------- normalize ----------------------------------

@pytest.mark.parametrize("dataset", ALL)
def test_clean_frame_is_unchanged(dataset):
    df = make(dataset)
    assert_frame_equal(normalize(df, dataset), df)


@pytest.mark.parametrize("dataset", ALL)
def test_normalize_is_idempotent(dataset):
    once = normalize(make(dataset), dataset)
    assert_frame_equal(normalize(once, dataset), once)


@pytest.mark.parametrize("dataset", ALL)
def test_empty_frame_roundtrip_keeps_dtypes(dataset):
    assert_frame_equal(normalize(empty_frame(dataset), dataset), empty_frame(dataset))


@pytest.mark.parametrize("dataset", ALL)
def test_input_is_not_modified(dataset):
    df = make(dataset).iloc[::-1]
    before = df.copy()
    normalize(df, dataset)
    assert_frame_equal(df, before)


def test_unknown_dataset_raises_value_error():
    with pytest.raises(ValueError):
        normalize(make("equity"), "nope")
    with pytest.raises(ValueError):
        validate(make("equity"), "nope")


def test_non_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        normalize([1, 2], "equity")
    with pytest.raises(TypeError):
        validate(None, "equity")


def test_frame_without_columns_returns_empty_frame():
    assert_frame_equal(normalize(pd.DataFrame(), "equity"), empty_frame("equity"))


def test_missing_required_column_logs_error_and_returns_empty(caplog):
    df = make("equity").drop(columns=["volume"])
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        out = normalize(df, "equity")
    assert_frame_equal(out, empty_frame("equity"))
    assert "volume" in caplog.text


def test_duplicate_column_names_return_empty_frame(caplog):
    df = pd.concat([make("equity"), make("equity")[["close"]]], axis=1)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        out = normalize(df, "equity")
    assert_frame_equal(out, empty_frame("equity"))
    assert "duplicate column" in caplog.text


def test_dtypes_are_coerced_from_strings_and_int32():
    df = pd.DataFrame({
        "date": ["2026-09-01", "2026-09-02"],
        "open": ["100.5", "101"], "high": ["105", "106"], "low": ["99", "100"],
        "close": ["104", "105"],
        "volume": pd.Series([1000, 2000], dtype="int32"),
        "trades": ["10", "20"], "symbol": ["TCS", "TCS"], "source": ["NSE", "NSE"]})
    out = normalize(df, "equity")
    assert {c: str(t) for c, t in out.dtypes.items()} == {
        c: ("object" if t == "object" else t) for c, t in SCHEMAS["equity"].items()}
    assert out["open"].iloc[0] == 100.5


def test_columns_reordered_and_extras_kept_after_schema_columns():
    df = make("corporate_actions")
    df["ratio"] = [None, 2.0, 2.0]          # a future column from corporate.py
    df = df[["ratio"] + list(reversed(SCHEMAS["corporate_actions"]))]
    out = normalize(df, "corporate_actions")
    assert list(out.columns) == list(SCHEMAS["corporate_actions"]) + ["ratio"]
    assert out["ratio"].tolist()[1:] == [2.0, 2.0]


def test_sorted_by_date_and_duplicates_dropped():
    df = make("equity")
    messy = pd.concat([df.iloc[::-1], df.iloc[[0]]], ignore_index=True)
    out = normalize(messy, "equity")
    assert len(out) == 3
    assert out["date"].is_monotonic_increasing
    assert_frame_equal(out, df)


def test_dedupe_keeps_first_occurrence():
    df = make("equity")
    dup = df.iloc[[0]].copy()
    dup["close"] = 999.0
    out = normalize(pd.concat([df, dup], ignore_index=True), "equity")
    assert len(out) == 3
    assert out["close"].iloc[0] == 104.0


def test_same_date_different_symbols_are_not_duplicates():
    a, b = make("equity"), make("equity")
    b["symbol"] = "INFY"
    assert len(normalize(pd.concat([a, b], ignore_index=True), "equity")) == 6


def test_bulk_deals_identical_rows_are_never_dropped():
    df = make("bulk_deals")
    doubled = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert len(normalize(doubled, "bulk_deals")) == 4


def test_corporate_actions_exact_duplicates_dropped():
    df = make("corporate_actions")
    assert len(normalize(pd.concat([df, df.iloc[[1]]], ignore_index=True),
                         "corporate_actions")) == 3


def test_sort_is_stable_for_rows_sharing_a_date():
    df = make("bulk_deals")
    df["date"] = pd.Timestamp("2026-09-01")
    out = normalize(df.iloc[::-1].reset_index(drop=True), "bulk_deals")
    assert out["symbol"].tolist() == ["CCC", "BBB", "AAA"]  # input order kept


def test_missing_int_becomes_zero_with_warning_and_prices_stay_nan(caplog):
    df = make("equity").astype({"volume": "float64", "close": "float64"})
    df.loc[1, "volume"] = float("nan")
    df.loc[1, "close"] = float("nan")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = normalize(df, "equity")
    assert out["volume"].tolist() == [1000, 0, 3000]
    assert str(out["volume"].dtype) == "int64"
    assert pd.isna(out["close"].iloc[1])          # never filled
    assert "volume" in caplog.text and "not the same as a real zero" in caplog.text


def test_unparseable_price_becomes_nan_with_warning(caplog):
    df = make("equity").astype({"open": "object"})
    df.loc[0, "open"] = "abc"
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = normalize(df, "equity")
    assert pd.isna(out["open"].iloc[0])
    assert "could not be parsed" in caplog.text


def test_source_fills_only_missing_values():
    df = make("equity").astype({"source": "object"})
    df.loc[1, "source"] = None
    out = normalize(df, "equity", source="NSE_BACKUP")
    assert out["source"].tolist() == ["NSE", "NSE_BACKUP", "NSE"]


def test_source_added_when_column_absent():
    df = make("equity").drop(columns=["source"])
    out = normalize(df, "equity", source="NSE")
    assert set(out["source"]) == {"NSE"}
    assert list(out.columns) == list(SCHEMAS["equity"])


def test_source_never_overwrites_existing():
    out = normalize(make("equity"), "equity", source="OTHER")
    assert set(out["source"]) == {"NSE"}


def test_timezone_aware_dates_are_made_naive():
    df = make("equity")
    df["date"] = df["date"].dt.tz_localize("Asia/Kolkata")
    out = normalize(df, "equity")
    assert str(out["date"].dtype) == "datetime64[ns]"
    assert out["date"].iloc[0] == pd.Timestamp("2026-09-01")


# ----------------------------- validate -----------------------------------

@pytest.mark.parametrize("dataset", ALL)
def test_clean_frame_validates_ok(dataset):
    report = validate(make(dataset), dataset)
    assert isinstance(report, ValidationReport)
    assert report.ok
    assert report.errors == []


@pytest.mark.parametrize("dataset", ALL)
def test_empty_frame_validates_ok(dataset):
    assert validate(empty_frame(dataset), dataset).ok


@pytest.mark.parametrize("dataset", ALL)
def test_validate_does_not_modify_input(dataset):
    df = make(dataset)
    before = df.copy()
    validate(df, dataset, start="2026-09-01", end="2026-09-03")
    assert_frame_equal(df, before)


def test_missing_columns_is_error(caplog):
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        report = validate(make("equity").drop(columns=["close"]), "equity")
    assert not report.ok and report.has("missing_columns")
    assert "close" in caplog.text


def test_wrong_dtype_is_error():
    df = make("equity").astype({"volume": "float64"})
    assert validate(df, "equity").has("bad_dtype")


def test_high_below_low_is_error():
    df = make("equity")
    df.loc[0, "high"] = 90.0
    report = validate(df, "equity")
    assert not report.ok and report.has("high_lt_low")


def test_null_price_is_error():
    df = make("equity")
    df.loc[2, "close"] = float("nan")
    assert validate(df, "equity").has("null_prices")
    assert not validate(df, "equity").ok


def test_close_outside_range_is_warning_only():
    df = make("equity")
    df.loc[0, "close"] = 500.0
    report = validate(df, "equity")
    assert report.ok and report.has("close_outside_range")


def test_nonpositive_price_is_warning_only():
    df = make("fno")
    df.loc[0, ["open", "high", "low", "close"]] = 0.0
    report = validate(df, "fno")
    assert report.ok and report.has("nonpositive_price")


def test_duplicate_dates_is_error():
    df = pd.concat([make("equity"), make("equity").iloc[[0]]], ignore_index=True)
    assert validate(df, "equity").has("duplicate_keys")


def test_unsorted_dates_is_warning():
    report = validate(make("equity").iloc[::-1], "equity")
    assert report.ok and report.has("unsorted_dates")


def test_negative_volume_is_error_but_negative_change_in_oi_is_fine():
    assert validate(make("fno"), "fno").ok      # make() has change_in_oi = -50
    df = make("equity")
    df.loc[0, "volume"] = -1
    assert validate(df, "equity").has("negative_values")


def test_null_date_is_error_except_for_corporate_actions():
    eq = make("equity")
    eq.loc[0, "date"] = pd.NaT
    assert not validate(eq, "equity").ok
    ca = make("corporate_actions")
    ca.loc[0, "ex_date"] = pd.NaT
    report = validate(ca, "corporate_actions")
    assert report.ok and report.has("null_dates")


def test_rows_outside_requested_range_is_error():
    report = validate(make("equity"), "equity", start="2026-09-02", end="2026-09-03")
    assert not report.ok and report.has("rows_outside_range")


def test_rows_inside_requested_range_ok():
    assert validate(make("equity"), "equity", start="2026-08-31", end="2026-09-04").ok


def test_coverage_warnings_when_data_far_from_requested_bounds():
    report = validate(make("equity"), "equity", start="2026-08-01", end="2026-09-30")
    assert report.ok
    assert report.has("coverage_start") and report.has("coverage_end")


def test_no_coverage_warning_within_tolerance():
    report = validate(make("equity"), "equity", start="2026-08-28", end="2026-09-06")
    assert not report.has("coverage_start") and not report.has("coverage_end")


def test_coverage_not_checked_for_corporate_actions():
    report = validate(make("corporate_actions"), "corporate_actions",
                      start="2020-01-01", end="2030-01-01")
    assert not report.has("coverage_start")


def test_gap_warning_for_continuous_series():
    df = make("equity")
    df.loc[2, "date"] = pd.Timestamp("2026-09-20")
    report = validate(df, "equity")
    assert report.ok and report.has("date_gaps")


def test_no_gap_warning_across_a_weekend():
    df = make("equity")
    df["date"] = pd.to_datetime(["2026-09-03", "2026-09-04", "2026-09-07"])
    assert not validate(df, "equity").has("date_gaps")


def test_gaps_are_checked_per_symbol_not_across_symbols():
    a = make("equity")
    b = make("equity")
    b["symbol"] = "INFY"
    b["date"] = pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-30"])
    both = normalize(pd.concat([a, b], ignore_index=True), "equity")
    report = validate(both, "equity")
    assert report.has("date_gaps") and report.issues[-1].count == 1


def test_custom_gap_threshold():
    df = make("equity")
    df["date"] = pd.to_datetime(["2026-09-01", "2026-09-04", "2026-09-05"])
    assert not validate(df, "equity").has("date_gaps")
    assert validate(df, "equity", max_gap_days=2).has("date_gaps")


def test_bulk_deal_duplicates_are_warning_not_error():
    df = make("bulk_deals")
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    report = validate(df, "bulk_deals")
    assert report.ok and report.has("duplicate_rows")


def test_bulk_deal_unexpected_type_and_zero_quantity_are_warnings():
    df = make("bulk_deals")
    df.loc[0, "deal_type"] = "HOLD"
    df.loc[1, "quantity"] = 0
    report = validate(df, "bulk_deals")
    assert report.ok
    assert report.has("unexpected_deal_type") and report.has("nonpositive_quantity")


def test_bulk_deal_type_check_ignores_case_and_whitespace():
    df = make("bulk_deals")
    df["deal_type"] = [" buy", "Sell ", "BUY"]
    assert not validate(df, "bulk_deals").has("unexpected_deal_type")


def test_null_required_value_is_warning():
    df = make("equity").astype({"symbol": "object"})
    df.loc[0, "symbol"] = None
    report = validate(df, "equity")
    assert report.ok and report.has("null_values")


def test_invalid_bound_raises_value_error():
    with pytest.raises(ValueError):
        validate(make("equity"), "equity", start="01-09-2026")


def test_issues_are_logged(caplog):
    df = make("equity")
    df.loc[0, "high"] = 90.0
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        validate(df, "equity")
    assert "high_lt_low" in caplog.text


def test_summary_and_ok_properties():
    df = make("equity")
    df.loc[0, "high"] = 90.0
    report = validate(df, "equity")
    assert "1 errors" in report.summary()
    assert not report.ok


def test_validate_works_on_raw_unnormalized_frame():
    raw = make("equity").astype({"open": "object"})
    raw.loc[0, "open"] = "abc"
    report = validate(raw, "equity")
    assert report.has("bad_dtype") and report.has("null_prices")


def test_normalize_then_validate_clean_for_every_dataset():
    for dataset in ALL:
        assert validate(normalize(make(dataset).iloc[::-1], dataset), dataset).ok