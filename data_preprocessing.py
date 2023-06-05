import pandas as pd


def clean_excel(file_location):
    df = pd.read_csv(file_location, sep=" ")
    df = df[df["positive_labels"].str.contains("/m/05zppz|/m/02zsn|/m/0ytgt")]
    df = df.reset_index(drop=True)

    for i, row in df.iterrows():
        value = row["positive_labels"]
        x = []
        if "/m/0ytgt" in value:
            x.append("child")
        if "/m/05zppz" in value:
            x.append("male")
        if "/m/02zsn" in value:
            x.append("female")
        df.iat[i, 3] = x
    return df
