from preprocessing.loader import (
    load_raw_data
)

from preprocessing.cleaner import (
    clean_data
)


def test_cleaner():

    path = (
        "data/raw/"
        "Scats Data October 2006.xls"
    )

    raw_df = load_raw_data(path)

    cleaned_df = clean_data(raw_df)

    assert cleaned_df is not None

    assert len(cleaned_df) > 0

    assert "traffic_flow" in cleaned_df.columns