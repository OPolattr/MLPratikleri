# sklearn : ML Library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

## (1) Veri Seti Incelemesi
cancer = load_breast_cancer()

df = pd.DataFrame(data = cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

# EDA


# (2) KNN Siniflandiricisi

X = cancer.data #features
y = cancer.target #target

# train test split
X_train, X_test, y_train, y_test   = train_test_split(X,y,test_size=0.3, random_state=42)

#olceklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Modelin Train Edilmesi
knn = KNeighborsClassifier(n_neighbors=9) #model oluşturma #k değerinin belirlenmesi
knn.fit(X_train,y_train) # fit fonksiyonu verimizi (samples + target) kullanarak knn algoritmasini eritir.


# (3) Sonuclarin Degerlendirilmesi
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print("Doğruluk : ", accuracy)

conf_matrix = confusion_matrix(y_test,y_pred)
print("confusion_matrix : ", conf_matrix)


# (4) Hiperparametre Ayarlamasi
"""
    KNN : Hyperparameter = K
        K: 1,2,3 .... N
        Accuracy: %A, %B, %C ....
"""

accuracy_values= []
k_values = []
for k in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)

plt.figure()
plt.plot(k_values, accuracy_values, marker="o", linestyle="-")
plt.title("K Degerine Gore Dogruluk")
plt.xlabel("K degeri")
plt.xlabel("Dogruluk")
plt.xticks(k_values)
plt.grid(True)


















