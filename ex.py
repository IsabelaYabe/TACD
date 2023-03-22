import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
#Ler csv e transformar num DataFrame
iris = pd.read_csv("./Iris.csv")

print(iris["SepalLengthCm"].value_counts().sort_index())
print(sns.countplot(x=iris["SepalLengthCm"].value_counts().sort_index()))

'''print(iris["SepalLengthCm"])'''
'''print(iris["SepalLengthCm"].value_counts())
print(type(iris["SepalLengthCm"].value_counts()))'''
'''
plot_iris = plt.hist([1,2,3,4],[10,15,20,25,30], density = True, color="g")
print(plot_iris)
print("foi")'''

              

