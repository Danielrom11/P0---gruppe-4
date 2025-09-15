import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Her indlæser vi vores træningsdata
# Datasættet virkede kun ved semikolonspereret csv-fil
df = pd.read_csv(r'C:\Users\danie\Desktop\python_work\P0---gruppe-4\data_KD.csv', sep=';')

# Vi fjerner de tomme felter
df.dropna(subset=['h', 's', 'v', 'target'], inplace=True)

# Vi tildeler X og y vores data værdier
X = df[['h', 's', 'v']].values
y = df['target'].values

# Vi splitter data i trænings- og testdata, vi bruger 20% af data til test og en random state på 10 for reproducerbarhed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# KNN modellen bliver fitter med træningsdata og vi tester den med testdata
# Vi sætter k=11, da vi har 137 træningspunkter og k bør være √137=11.7
k = 11
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of k={k}: {accuracy*100:.2f}%')


# Encode target labels for plotting
y_train_encoded, uniques = pd.factorize(y_train)
y_test_encoded = pd.Categorical(y_test, categories=uniques).codes

# 3D Visualization
fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train_encoded, cmap='Set1', s=20, label='Train')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test_encoded, cmap='Set1', s=50, marker='*', label='Test')
ax.set_xlabel('h')
ax.set_ylabel('s')
ax.set_zlabel('v')

# Annotate train points
for i in range(len(X_train)):
    ax.text(X_train[i, 0], X_train[i, 1], X_train[i, 2], str(y_train[i]), fontsize=7, color='black')

# Annotate test points
for i in range(len(X_test)):
    ax.text(X_test[i, 0], X_test[i, 1], X_test[i, 2], str(y_test[i]), fontsize=9, color='blue')

plt.title(f'3D KNN (k={k})')
plt.legend()
plt.show()