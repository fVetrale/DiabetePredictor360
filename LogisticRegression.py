import matplotlib.pyplot as plt
import pandas as pd
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import MinMaxScaler

dtl = pd.read_csv('diabetes_prediction_dataset.csv')
print(dtl.head())

print('DIMENSIONE DATASET:', dtl.shape)

columnsdataset = dtl.columns

print("Colonne nel file CSV:")
for col in columnsdataset:
    print(dtl[col].value_counts())

print('Tipi di dati: \n', dtl.dtypes)

print('Valori nulli: \n', dtl.isnull().sum())

X = dtl.drop(['diabetes'], axis = 1)
y = dtl['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

#Seleziono le feature categoriche
colonne_categoriche = dtl.select_dtypes(include=['object']).columns.tolist()
# Visualizza le colonne categoriche
print("Feature Categoriche:")
print(colonne_categoriche)

encoder = ce.OrdinalEncoder(cols= colonne_categoriche)

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print(X_train.head())

'''
SMOTETomek: combina oversampling e undersampling utilizzando SMOTE e Tomek Links insieme (Dinamico).
'''
smotetomek = SMOTETomek(sampling_strategy='auto', random_state=42)
X_train, y_train = smotetomek.fit_resample(X_train, y_train)

cols = X_train.columns

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

'''
AUTOML utilizzando GridSearchCV 

1. Il parametro C controlla la forza di regolarizzazione nel modello di regressione logistica.
Valori più grandi di C indicano una regolarizzazione più debole, il che significa che il modello cercherà di adattarsi meglio ai dati di addestramento.

2. La penalità è un termine aggiunto alla funzione di costo del modello per evitare l'overfitting. Scikit-learn supporta due tipi di penalità: "l1" (lasso) e "l2" (ridge).

logistic_regression = LogisticRegression()

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(logistic_regression, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Migliori parametri:", best_params)

model = grid_search.best_estimator_
'''

model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

#Check Overfitting
print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))

confMatrix = confusion_matrix(y_test, prediction)
plt.figure(figsize=(8,8))

sns.heatmap(confMatrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])

plt.title('Matrice di Confusione')
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.show()

TN, FP, FN, TP = confMatrix.ravel()

recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1_score = 2 * (precision * recall) / (precision + recall)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print('Recall: {:.4f}'.format(recall))
print('Precision: {:.4f}'.format(precision))
print('F1-Score: {:.4f}'.format(f1_score))
print('Accuracy: {:.4f}'.format(accuracy))

fpr, tpr, thresholds = roc_curve(y_test, prediction)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
