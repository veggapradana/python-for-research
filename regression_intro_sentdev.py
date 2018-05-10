#Source: PythonProgramming.net
#Links: https://pythonprogramming.net/regression-introduction-machine-learning-tutorial/?completed=/machine-learning-tutorial-python-introduction/
#testing edit
import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL')

print(df.head())
