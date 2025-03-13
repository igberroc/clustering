
import pandas as pd
if __name__ == '__main__':
    df = pd.read_csv('customer_dataset.csv', dayfirst = True)
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst = True)
    df['Dt_Customer'] = df['Dt_Customer'].dt.year
    df_tuples = list(df.itertuples(index = False, name = None))
    point = df_tuples[0]
    for coord in point:
        print(type(coord))

