# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-NML4j4_NclRWzrMPiEFjAh40N_rFX41
"""

import pandas as pd

data = pd.read_csv("iris1.csv")

data.head()

data.sample(10)

data.columns

data.shape

print(data)

#data[start:end]
#start is inclusive whereas end is exclusive
print(data[10:21])
# it will print the rows from 10 to 20.

# you can also save it in a variable for further use in analysis
sliced_data=data[10:21]
print(sliced_data)

specific_data=data[["Id","species"]]
print(specific_data.head(10))

data.iloc[5]

data.loc[data["species"] == "Iris-setosa"]

#In this dataset we will work on the Species column, it will count number of times a particular species has occurred.
data["species"].value_counts()

# data["column_name"].sum()

sum_data = data["sepal_length"].sum()
mean_data = data["sepal_length"].mean()

print("Sum:",sum_data, "\nMean:", mean_data,)

min_data=data["sepal_length"].min()
max_data=data["sepal_length"].max()

print("Minimum:",min_data, "\nMaximum:", max_data)

# For example, if we want to add a column let say "total_values",
# that means if you want to add all the integer value of that particular
# row and get total answer in the new column "total_values".
# first we will extract the columns which have integer values.
cols = data.columns

# it will print the list of column names.
print(cols)

# we will take that columns which have integer values.
cols = cols[1:5]

# we will save it in the new dataframe variable
data1 = data[cols]

# now adding new column "total_values" to dataframe data.
data["total_values"]=data1[cols].sum(axis=1)

# here axis=1 means you are working in rows,
# whereas axis=0 means you are working in columns.

newcols={
"Id":"id",
"sepal_length":"sepallength",
"sepal_width" :"sepalwidth"
}

data.rename(columns=newcols,inplace=True)

print(data.head())

data.style

data.head(10).style.highlight_max(color='lightgreen', axis=0)

data.head(10).style.highlight_max(color='lightgreen', axis=1)

data.head(10).style.highlight_max(color='lightgreen', axis=None)

data.isnull()
#if there is data is missing, it will display True else False.

data.isnull.sum()

import seaborn as sns

iris = sns.load_dataset("iris")
sns.heatmap(iris.corr(),camp = "YlGnBu", linecolor = 'white', linewidths = 1)

sns.heatmap(iris.corr(),camp = "YlGnBu", linecolor = 'white', linewidths = 1, annot = True )

data.corr(method='pearson')

g = sns.pairplot(data,hue="Species")