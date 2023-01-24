from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV
from  sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
df = pd.read_csv('heart.csv', sep=',', engine='python')
	
X = df.drop(['output'],axis=1).values   
y = df['output'].values
	
#Separa el corpus cargado en el DataFrame en entrenamiento y el pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 0 )

parameters = {'n_neighbors':[1,5,10], 'weights': ('uniform', 'distance')}

#Training
clf = KNeighborsClassifier()

cv = GridSearchCV(clf, parameters, verbose=3)
cv.fit(X_train, y_train)

# ~ print (cv.cv_results_)
df = pd.DataFrame(cv.cv_results_)
print (df)
df.to_csv('cv_results.csv')
print (cv.best_score_)
print (cv.best_params_)

#Testing
y_pred = cv.predict(X_test)

#~ #Model evaluation
print (classification_report(y_test, y_pred))

#################################################
#					PIPELINES                   #
#################################################
scaler_names = ['Escalado estándar', 'Escalado robusto']
scalers = [preprocessing.StandardScaler(), preprocessing.RobustScaler()]
clf = KNeighborsClassifier()
parameters_pipeline = [{'clf__n_neighbors':[1,5,10], 'clf__weights': ('uniform', 'distance')}]


for scaler_name, scaler in zip(scaler_names, scalers):
	print ('Método de escalado: ', scaler_name)
	
	pipeline = Pipeline([
					('scalers', scaler),
					('clf', clf)
					])
					
	cv = GridSearchCV(pipeline, param_grid=parameters_pipeline, verbose=3)
	cv.fit(X_train, y_train)
	print (cv.best_score_)
	print (cv.best_params_)




## 70-30
print('70-30')
x_train_esc= preprocessing.StandardScaler().fit_transform(X_train)
x_test_esc= preprocessing.StandardScaler().fit_transform(X_test)

#Training
clf = KNeighborsClassifier(n_neighbors=10, weights='uniform')
#target_names=clf.classes_
#cv = GridSearchCV(clf, parameters, verbose=3)
clf.fit(x_train_esc, y_train)

#Test
y_pred_test = clf.predict(x_test_esc)
print(classification_report(y_test, y_pred_test))
cm=confusion_matrix(y_test,y_pred_test)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


	
	
	
	
	
	
	
	
			
