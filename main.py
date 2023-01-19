import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
df = pd.read_csv('Test.csv')
print(df)

profile = ProfileReport(df)
profile.to_file(output_file="Test.html")