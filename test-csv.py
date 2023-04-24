import pandas as pd

pd.options.display.max_rows = 20

#data = pd.read_csv("hems.csv", usecols= ['column_name1','column_name2'])
data = pd.read_csv('hems.csv',sep=';',usecols=['day1','day2','occupation1','occupation2','power1','power2'])
print(data)
