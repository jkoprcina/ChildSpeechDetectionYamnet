from __future__ import unicode_literals, division, print_function
from data_preprocessing import clean_excel
from inference import inference
import ast
import pandas as pd
from downloading_files import download_files


if __name__ == '__main__':
    file_location = "eval_segments.csv"
    df = clean_excel(file_location)
    download_files(df)
    inference(df)

    # data checking
    df = pd.read_csv('data.csv')
    df.drop(columns=df.columns[0], axis=1, inplace=True)

    count_dict = {
        "child_count": 0,
        "female_count": 0,
        "male_count": 0,
        "tp": 0,
        "male_fp": 0,
        "female_fp": 0,
        "tn": 0,
        "fn": 0,
    }
    vidim_duplo = 0

    for i, row in df.iterrows():
        if len(ast.literal_eval(row['positive_labels'])) > 1:
            vidim_duplo += 1
        positive_labels = ast.literal_eval(row['positive_labels'])

        for positive_label in positive_labels:
            if positive_label == 'child':
                count_dict['child_count'] += 1
                if 'Child speech, kid speaking' in row['labels']:
                    count_dict['tp'] += 1
                else:
                    count_dict['fn'] += 1

            elif positive_label == 'female':
                count_dict['female_count'] += 1
                if 'Child speech, kid speaking' in row['labels']:
                    count_dict['female_fp'] += 1
                else:
                    count_dict['tn'] += 1

            elif positive_label == 'male':
                count_dict['male_count'] += 1
                if 'Child speech, kid speaking' in row['labels']:
                    count_dict['male_fp'] += 1
                else:
                    count_dict['tn'] += 1
            else:
                print("Unexpected data")
    print(i)
    print(vidim_duplo)
    print(count_dict)
