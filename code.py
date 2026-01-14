import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\Admin\Desktop\lgbm\Social_Network_Ads.csv')  

x= dataset.iloc[:,[2,3]].values
y = dataset.iloc[: , -1].values


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from lightgbm import LGBMClassifier

classifier = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=0
)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
y_pred


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
ac

variance = classifier.score(x_test,y_test)
bias = classifier.score(x_train,y_train)

bias 
variance


from sklearn.metrics import roc_auc_score,roc_curve
y_pred_prob = classifier.predict_proba(x_test)[:,1]

auc_score=roc_auc_score(y_test,y_pred_prob)
auc_score

# VISUALISING TRAINING  SET

from matplotlib.colors import ListedColormap

X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(
    np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
)

plt.contourf(
    X1, X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green'))(i),
        label=j
    )

plt.title('LightGBM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# VISUALISING TEST SET

X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(
    np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
)

plt.contourf(
    X1, X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green'))(i),
        label=j
    )

plt.title('LightGBM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# SAVE MODEL

import pickle

# Save model
with open("LGBM_Classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(sc, f)

print("Model and scaler saved successfully")