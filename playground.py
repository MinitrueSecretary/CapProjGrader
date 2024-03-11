import numpy as np
import pandas as pd

df = pd.read_csv('qsanswers.HW3_Hardware_beyond-467245-17100579891898.csv')

df.loc[0:12].to_csv('test.csv',index=False)