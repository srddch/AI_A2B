import pandas as pd


def load_raw_data(path):

    df = pd.read_excel(
        path,
        sheet_name="Data",
        header=1,
        engine="xlrd"
    )

    print(df.head())
    print(df.columns)

    return df