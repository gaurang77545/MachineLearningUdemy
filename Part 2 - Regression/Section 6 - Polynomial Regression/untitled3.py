# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('haha.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

