import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
#Data Loading and Visualisation
iris = pd.read_csv("./Iris.csv")

#Plot the histograms 
iris_hist = sns.histplot(data = iris["SepalLengthCm"])
#print(iris_hist)

iris_spe_hist = sns.histplot(x = iris["SepalLengthCm"], hue = "Species", data = iris)
#print(iris_spe_hist)

#Binary Classification Building a ROC curve
# Loading the pre-trained classifiers and testing data
open_file_class = open("classifiers_dict.p", "rb")
open_file_test_data = open("mushroom_test_data.p", "rb")
classifiers = pkl.load(open_file_class)
mushroom_test_data = pkl.load(open_file_test_data)

#print(f"Classifiers: {classifiers} \nData: {mushroom_test_data} \n") #Descomenta se for necessário
#print(classifiers.keys())
#print(classifiers.values())
#print(mushroom_test_data.keys()) #Chaves: ['X_test', 'y_test']
#print(type(mushroom_test_data["X_test"])) #DataFrame
#print(type(mushroom_test_data["y_test"])) #SeriePandas
#Valores são dicionários
#print(mushroom_test_data["X_test"])
#print(mushroom_test_data["y_test"])

#print(f"Classifiers type: {type(classifiers)}")
#print(f"Mushroom test data type: {type(mushroom_test_data)}\n")

cnb_clf = classifiers["Categorical NB"]
#print(f"Classifier[Categorical NB]: {cnb_clf}")
#print(f"Classifier[Categorical NB] type: {type(cnb_clf)}\n")
lr_clf = classifiers["Logistic Regression"]
#print(f"Classifier[Logistic Regression]: {lr_clf}")
#print(f"Classifier[Logistic Regression] type: {type(lr_clf)}\n")
svm_clf = classifiers["SVM"]
#print(f"Classifier[SVM]: {svm_clf}")
#print(f"Classifier[SVM] type: {type(svm_clf)}\n")
gb_clf = classifiers["Gradient Boosting"]
#print(f"Classifier[SVM]: {svm_clf}")
#print(f"Classifier[Gradient Boosting] type: {type(gb_clf)}\n")

#Definindo variáveis
X_test = mushroom_test_data["X_test"] #DataFrame
y_test = mushroom_test_data["y_test"] #SeriePandas
#print(X_test.head())
#print(X_test.columns)


#We are now going to obtain the predicted probabilities from our different classifiers
#To do this we are using the method predict_proba()
#This is a method specific to each classifier and it requires as input argument the datapoints of our testing set with their features (X_test).
y_proba_cnb = cnb_clf.predict_proba(X_test)
#print(f"y proba cnb: {y_proba_cnb}\n")
y_proba_svm = svm_clf.predict_proba(X_test)
#print(f"y proba svm: {y_proba_svm}\n")
y_proba_lr = lr_clf.predict_proba(X_test)
#print(f"y proba lr: {y_proba_lr}\n")
y_proba_gb = gb_clf.predict_proba(X_test)
#print(f"y proba gb: {y_proba_gb}\n")
'''matriz = np.array([[2, 3, 4],[9, 8, 7]])
print(matriz[1, 1])
b = np.array([[0, 1, 2], [3, 4, 5], [0, 1, 2]])    # 2 x 3 array
print(b.ndim)
print(b.shape)
print(len(b))
count = 0
while count < len(b):
    print(count)
    count+=1'''
# The following is just example code, it is not meant to be executed.
def get_fpr_tpr(predicted_values, true_labels, threshold):
    TP = 0 #True Positive
    #print(f"TP: {TP}")
    FN = 0 #False Negative
    #print(f"FN: {FN}")
    FP = 0 #False Positive
    #print(f"FP: {FP}")
    TN = 0 #True Negativo
    #print(f"TN: {TN}")
    
    lem = len(predicted_values)
    #print(f"Números de elementos em predicted_values: {lem}")
    lemm = len(true_labels)
    #print(f"Números de elementos em true_labels: {lemm}")
    count = 0
    #print(f"Count: {count}")
    #print("=====================================")
    
    while count < lem:
        if predicted_values[count] <= threshold:
            #print(f"Valores avaliados como 0")
            #print(f"Valores analisados: \n  Predicted_values: {predicted_values[count]}\n   True_labels: {true_labels[count]}")
            if 0 == true_labels[count]: #Falso Negative
                FN+=1
                #print(f"Valor adicionado em FN, FN atualisado: {FN}")                
            elif 0 != true_labels[count]: #True Negative
                TN+=1
                """print(f"Valor adicionado em TN, TN atualisado: {TN}")            
            print(f"FN: {FN}")
            print(f"TN: {TN}")
            print(f"TP: {TP}")
            print(f"FP: {FP}") 
            print(f"Count: {count}")
            print("=====================================")"""   
            count+=1
        else:
            #print(f"Valores avaliados como positivos")
            #print(f"Valores analisados:    \n   Predicted_values: {predicted_values[count]}\n   True_labels: {true_labels[count]}")
            if 1 == true_labels[count]: #True Positive
                TP+=1
                #print(f"Valor adicionado em TP, TP atualisado: {TP}")
                    
            elif 1 != true_labels[count]: #False Positive
                FP+=1
                """print(f"Valor adicionado em FP, FP atualisado: {FP}")
            print(f"FN: {FN}")
            print(f"TN: {TN}")
            print(f"TP: {TP}")
            print(f"FP: {FP}")
            print(f"Count: {count}")
            print("=====================================")"""    
            count+=1
    P = TP+FN #Total Positive Predict
    #print(f"P: {P}")
    N = FP+TN #Total Negativo Predict
    #print(f"N: {N}")
    fpr = FP/N
    #print(f"False Positive Rate: {fpr}")
    tpr = TP/P
    #print(f"True Positive Rate: {tpr}")
    return fpr, tpr

def get_class_1(array):
    list = []
    for i in array:
        list.append(i[1])
    array_1 = np.array(list)
    return array_1

def roc_serie(predicted_values, true_labels, threshold_array):
    x_label = []
    y_label = []
    for i in threshold_array:
        img = get_fpr_tpr(predicted_values, true_labels, i)
        x_label.append(img[0])
        y_label.append(img[1])
    serie = pd.Series(data=y_label, index=x_label).sort_index()
    return serie
           
'''a = np.array([1,2,3,4,5,6,7,8,9,0])
b = np.array([1,1,3,1,5,1,7,1,9,1])

print(roc_x_label(a, b))
print("==========")
print(roc_y_label(a,b))
'''
#print(y_test)
#print(type(y_test))
true_labels = pd.Series.to_numpy(y_test)
#print(true_labels)
#print(type(true_labels))
"""print(len(y_test))
print(y_proba_cnb)
print(len(y_proba_gb))
print(len(y_proba_lr))
print(len(y_proba_svm))
"""



predicted_values_cnb = get_class_1(y_proba_cnb)
#print(get_fpr_tpr(predicted_values_cnb, true_labels, 0.3))
print("==============================================")
print("==============================================")
#print(x_label_cnb)
threshold_array = np.sort(np.random.rand(100))
print(threshold_array)
serie = roc_serie(predicted_values_cnb, true_labels, threshold_array)
print(serie)

plt.plot(get_fpr_tpr(predicted_values_cnb, true_labels, 0.3))




"""# Here you must complete the calls to plt.plot() with the right input arguments
# Following that you will need to generate the correct plot properties below
fig = plt.figure(figsize=(8,6))
true_labels = pd.Series.to_numpy(y_test)
threshold_array = np.sort(np.random.rand(70))
# ROC Curve for the Categorical Naive Bayes
predicted_values_cnb = get_class_1(y_proba_cnb)
serie_cnb = roc_serie(predicted_values_cnb, true_labels, threshold_array)
plt.plot(serie_cnb.index, serie_cnb.values)
# ROC Curve for Logistic Regression
predicted_values_lr = get_class_1(y_proba_lr)
serie_lr = roc_serie(predicted_values_lr, true_labels, threshold_array)
plt.plot(serie_lr.index, serie_lr.values)
# ROC Curve for SVM
predicted_values_svm = get_class_1(y_proba_svm)
serie_svm = roc_serie(predicted_values_svm, true_labels, threshold_array)
plt.plot(serie_svm.index, serie_svm.values)
# ROC Curve for Gradient Boosting
predicted_values_gb = get_class_1(y_proba_gb)
serie_gb = roc_serie(predicted_values_gb, true_labels, threshold_array)
plt.plot(serie_gb.index, serie_gb.values)

# Plot properties
# Title

# X-Ticks and X-label 

# Y-Ticks and Y-label

# Legend

plt.show()"""

# Here you must complete the calls to plt.plot() with the right input arguments
# Following that you will need to generate the correct plot properties below
fig = plt.figure(figsize=(8,6))
from sklearn.metrics import roc_curve, auc
predicted_values_cnb = get_class_1(y_proba_cnb)
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, predicted_values_cnb)
auc_logistic = auc(logistic_fpr, logistic_tpr)
svm_fpr, svm_tpr, threshold = 
# ROC Curve for the Categorical Naive Bayes
predicted_values_cnb = get_class_1(y_proba_cnb)
cnb_fpr, cnb_tpr, threshold = roc_curve(y_test, predicted_values_cnb)
auc_logistic = auc(cnb_fpr, cnb_tpr)
plt.plot()
# ROC Curve for Logistic Regression
predicted_values_lr = get_class_1(y_proba_lr)
lr_fpr, lr_tpr, threshold = roc_curve(y_test, predicted_values_lr)
auc_logistic = auc(lr_fpr, lr_tpr)
plt.plot()