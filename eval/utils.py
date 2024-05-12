import pandas as pd
import numpy as np

def calculate_SEM(ser_groupby):
    avg_of_best = ser_groupby.mean()
    df_avg_of_best = avg_of_best.reset_index()
    # print(avg_of_best)
    
    df_avg_of_best['standard_error'] = ser_groupby.std().values / np.sqrt(df_avg_of_best.shape[0])

    return df_avg_of_best