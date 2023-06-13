import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('datafinlechuza.csv')
#data = data.drop(['Fecha'], axis=1)
#data = data.drop(['decimal'], axis=1)
data['Potencia'] = data['Potencia'].mul(-1)
#print(data)
max_potencia = data['Potencia'].max()
min_potencia = data['Potencia'].min()
#print(max_potencia,min_potencia)

potencia_bins = [min_potencia, 800,1600, 2400, max_potencia]
potencia_labels = [ 'bajo','medio','alto', 'muy alto']
data['categoria_potencia']= pd.cut(data['Potencia'], bins=potencia_bins, labels = potencia_labels)
data = data.fillna('muy alto')


X = data.drop(['Potencia', 'categoria_potencia'], axis=1)
y = data['categoria_potencia']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


model = AdaBoostClassifier(n_estimators=1000, random_state=0)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)



print('Precisión del modelo:', accuracy*100)