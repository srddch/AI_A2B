import pandas as pd


def clean_data(df):
    """
    Convert SCATS wide format
    into time-series long format
    """

    flow_columns = [
        f"V{i:02d}"
        for i in range(96)
    ]

    cleaned_rows = []

    for _, row in df.iterrows():

        scats_number = row["SCATS Number"]

        date = pd.to_datetime(
            row["Date"]
        )

        for i, col in enumerate(flow_columns):

            flow_value = row[col]

            timestamp = (
                date +
                pd.Timedelta(minutes=15 * i)
            )

            cleaned_rows.append({
                "datetime": timestamp,
                "scats_number": scats_number,
                "traffic_flow": flow_value
            })

    cleaned_df = pd.DataFrame(
        cleaned_rows
    )

    cleaned_df = cleaned_df.dropna()

    cleaned_df = cleaned_df.sort_values(
        by="datetime"
    )

    cleaned_df.reset_index(
        drop=True,
        inplace=True
    )

    return cleaned_df