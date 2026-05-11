from preprocessing.loader import load_raw_data


def test_loader():

    path = (
        "data/raw/"
        "Scats Data October 2006.xls"
    )

    df = load_raw_data(path)

    assert df is not None

    assert len(df) > 0