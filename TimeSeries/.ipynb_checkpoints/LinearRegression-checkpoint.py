import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import random as rd
import pandas as pd


# Init
sns.set_style("darkgrid") # Style du graphe
N = 1000

# Init X et Y (Y = 2*X + 3 + bruit)
X = np.array([
   np.random.normal(0,1) for _ in range(N)
]).reshape(-1,1)
Y = 2*X + 3 + np.array([
   np.random.normal(0,1) for _ in range(N)
]).reshape(-1,1)

df = pd.DataFrame({
    "Y" : Y.reshape(-1),
    "X" : X.reshape(-1)
})
df.head()
print(df.head())