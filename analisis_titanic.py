#!/usr/bin/env python
# coding: utf-8

# In[88]:


# presentadoi por ANDRES FELIPE CAICEDO
#Analisis de datos Etap 3 UNAD 
# Importar os para lectura del dataset desde el SO
import os
# Importar pandas y numpy para manipulación de datos
import pandas as pd
import numpy as np
# Importamos sklearn para poder aplicar el modelo arbol de decision
from sklearn import tree
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Importamos matplotlib y seaborn 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
font = {'size': 12}
plt.rc('font', **font)


# In[96]:


# Importamos el dataset y validamos su estructura
os.chdir("C:\\Users\\felipe.caicedo\\Documents\\analis tarea 3\\Dataset_Titanic")
datos = pd.read_csv("train.csv", delimiter=',')
datos.head()


# In[97]:


num_rows, num_cols = datos.shape
print("\033[1mDimensiones del dataset:\033[0m")
print("\033[1mNúmero de filas:\033[0m", num_rows, ", \033[1mNúmero de columnas:\033[0m", num_cols)


# In[98]:


# Exploración de datos

num_rows, num_cols = datos.shape
print("\033[1mDimensiones:\033[0m")
print("\033[1mNúmero de filas:\033[0m", num_rows, ", \033[1mNúmero de columnas:\033[0m", num_rows)


# In[99]:


columns_names = datos.columns
print("\033[1mNombre columnas :\033[0m \n")
print(columns_names)


# In[100]:


print("\033[1mNúmero de datos faltantes por cada campo:\033[0m")
datos.isna().sum()


# In[101]:


print("\033[1mTipos de datos :\033[0m \n")
print(datos.dtypes)


# In[102]:


#  Limpieza e imputación de datos
datos['Age'] = datos['Age'].fillna(round(datos['Age'].mean()))
datos['Cabin'] = datos['Cabin'].fillna("NE")
datos['Embarked'] = datos['Embarked'].fillna("NE")
datos.isna().sum()


# In[109]:


#  Limpieza e imputación de datos

print("\033[1m datos\033[0m ")
# Calcular el rango intercuartil (IQR) para la columna 'Edad'
Q1_edad = datos['Age'].quantile(0.25)
Q3_edad = datos['Age'].quantile(0.75)
IQR_edad = Q3_edad - Q1_edad

# Calcular el rango intercuartil (IQR) para la columna 'Tarifa'
Q1_tarifa = datos['Fare'].quantile(0.25)
Q3_tarifa = datos['Fare'].quantile(0.75)
IQR_tarifa = Q3_tarifa - Q1_tarifa

# Identificar valores atípicos en 'Edad' y 'Tarifa'
outliers_edad = (datos['Age'] < (Q1_edad - 1.5 * IQR_edad)) | (datos['Age'] > (Q3_edad + 1.5 * IQR_edad))
outliers_tarifa = (datos['Fare'] < (Q1_tarifa - 1.5 * IQR_tarifa)) | (datos['Fare'] > (Q3_tarifa + 1.5 * IQR_tarifa))

# Extraer los valores atípicos
outliers_edad_values = datos[outliers_edad]
outliers_tarifa_values = datos[outliers_tarifa]

print('\033[1mDatos anomalos edad:\033[0m ', len(outliers_edad_values))
print('\033[1mDatos anomalos tarifa:\033[0m ', len(outliers_tarifa_values))

# Limpiar el dataset de los valores atípicos
datos.drop(datos[outliers_edad | outliers_tarifa].index, inplace=True)


# In[108]:


print('\033[1mValidamos si hay datos duplicados:\033[0m')
datos.duplicated()


# In[110]:


# Estadística 
print('\033[1mDistribución de la variable Survived en el dataset:\033[0m')
fig, axes = plt.subplots(1, 3, figsize = (15, 5))
# Gráfico de barras de la distribución de 'Survived'
sns.countplot(x='Survived', data=datos, ax = axes[0])
axes[0].set_title('Distribución de Sobrevivientes')
axes[0].set_xlabel('No sobrevivieron (0) / Sobrevivieron (1)')
axes[0].set_ylabel('Cantidad de Pasajeros')

sns.histplot(datos['Age'], kde=True, ax = axes[1])
axes[1].set_title('Histograma de Edades')
axes[1].set_xlabel('Edad')
axes[1].set_ylabel('Frecuencia')

sns.histplot(datos['Fare'], kde=True, ax = axes[2])
axes[2].set_title('Histograma de Tarifa')
axes[2].set_xlabel('Tarifa')
axes[2].set_ylabel('Frecuencia')

plt.show()


# In[55]:


clase_counts = datos['Pclass'].value_counts()

print('\033[1mnumero  pasajeros en cada clase:\033[0m\n')
print(clase_counts)


# In[58]:


# Verificamos tabla de correlación
datos.corr()


# In[60]:


print('\033[1mVerifiquemos la tasa de supervivencia por clase, genero y grupo de edad:\033[0m')
# Crear una figura con tres subgráficos
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Gráfico 1: Tasa de supervivencia por clase de pasajero
sns.barplot(x='Pclass', y='Survived', data=datos, ax=axes[0])
axes[0].set_title('Tasa de Supervivencia por Clase de Pasajero')

# Gráfico 2: Tasa de supervivencia por género
sns.barplot(x='Sex', y='Survived', data=datos, ax=axes[1])
axes[1].set_title('Tasa de Supervivencia por Género')

# Crear intervalos de edad (por ejemplo, 0-9, 10-19, 20-29, ...)
datos['AgeGroup'] = pd.cut(datos['Age'], bins=[5, 15, 25, 35, 45, 55], labels=['5-15', '16-25', '26-35', '36-45', '46-55'])

# Gráfico 3: Tasa de supervivencia por grupo de edad
sns.barplot(x='AgeGroup', y='Survived', data=datos, ax=axes[2])
axes[2].set_title('Tasa de Supervivencia por Grupo de Edad')

plt.show()

print('\033[1mCantidad de personas en clase#1\033[0m: ', datos[datos['Pclass'] == 1]['Pclass'].count())
print('\033[1mCantidad de personas en clase#2\033[0m: ', datos[datos['Pclass'] == 2]['Pclass'].count())
print('\033[1mCantidad de personas en clase#3\033[0m: ', datos[datos['Pclass'] == 3]['Pclass'].count())
print('--------------------------------------------\n')
print('\033[1mCantidad de personas en hombres\033[0m: ', datos[datos['Sex'] == 0]['Sex'].count())
print('\033[1mCantidad de personas en mujeres\033[0m: ', datos[datos['Sex'] == 1]['Sex'].count())
print('--------------------------------------------\n')
print('\033[1mCantidad de personas entre 5 y 15 años\033[0m: ', datos[(datos['Age'] >= 5) & (datos['Age'] <= 15)]['Age'].count())
print('\033[1mCantidad de personas entre 16 y 25 años\033[0m: ', datos[(datos['Age'] >= 16) & (datos['Age'] <= 25)]['Age'].count())
print('\033[1mCantidad de personas entre 26 y 35 años\033[0m: ', datos[(datos['Age'] >= 26) & (datos['Age'] <= 35)]['Age'].count())
print('\033[1mCantidad de personas entre 36 y 45 años\033[0m: ', datos[(datos['Age'] >= 36) & (datos['Age'] <= 45)]['Age'].count())
print('\033[1mCantidad de personas entre 45 y 55 años\033[0m: ', datos[(datos['Age'] >= 46) & (datos['Age'] <= 55)]['Age'].count())


# In[64]:


#Equilibrar el conjunto de datos

survived_counts = datos['Survived'].value_counts()

print('Sobrevivientes (1) y No Sobrevivientes (0):')
print(survived_counts)

# Calcular la proporción
proporciones = survived_counts / len(datos)
print('\nProporción de Sobrevivientes y No Sobrevivientes:\n')
print(proporciones)
print('\n\033[1m se puede  evidenciar un desbalance en las proporciones\033[0m')


# In[118]:


# División del dataset 
# Mapeo de los datos a valores númerico
# evaluar el modelo de arbol de decisión

datos['Survived'].replace(('No', 'Si'), (0, 1), inplace = True)
datos['Sex'].replace(('male', 'female'), (0,1), inplace = True)

# Eliminación de datos categóricos que no ayudaran en el modelo
# Separación en datos dependientes e independiente
X = datos.drop(columns = ['Survived', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Age'])
Y = datos['Survived']

# Separación datos de entrenamiento y prueba con razón de pareto 80-20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,random_state=0)

# Creamos nuestro modelo de regresión lineal
arbol = DecisionTreeClassifier(max_depth=4)

# Aplicamos el modelos al conjunto de entrenamiento
arbol_sobrevivir = arbol.fit(X_train, Y_train)

# Definimos el tamaño de nuestra visualización
fig = plt.figure(figsize=(25,20))

# Construimos nuestro árbol de decisión para ser visualizado
tree.plot_tree(arbol_sobrevivir, feature_names=list(X_train.columns.values),
              class_names=list({"0", "1"}), filled=True)

# Visualizamos nuestro árbol de decisión
plt.show()


# In[120]:


#  validación con el conjunto de prueba
   
Y_pred = arbol_sobrevivir.predict(X_test)
mapeo = {0: "SI", 1: "NO"}
print('\033[1mPredicción de sobrevivientes:\033[0m \n\n', np.vectorize(mapeo.get)(Y_pred))

# Verificamos nuestro modelo a través de una matriz de confusión
print('\n \033[1mConteo de predicciones exitos y no exitosas:\033[0m')
matriz_confusion = confusion_matrix(Y_test, Y_pred)
matriz_confusion


# In[121]:


## Validamos la precisión de nuestro modelo (validaciones correctas)

Presicion_Global = np.sum(matriz_confusion.diagonal())/np.sum(matriz_confusion)
print("\033[1m precisión del modelo en terminos generales :\033[0m", np.round(Presicion_Global*100, 2), "% ")

## Validamos la precisión de nuestro modelo (validación correcta de No Sobrevientes)
Presicion_death = matriz_confusion[1,1]/np.sum(matriz_confusion[1,])
print("\033[1m precisión del modelo en sobrevivientes:\033[0m", np.round(Presicion_death*100, 2), "% ")

## Validamos la precisión de nuestro modelo (validación correcta de Sobrevientes)
Presicion_alive = matriz_confusion[0,0]/np.sum(matriz_confusion[0,])
print("\033[1m precisión del modelo en no sobrevivientes:\033[0m", np.round(Presicion_alive*100, 2), "% ")

print('''\n\033[1mNota:\033[0m la presición está basada  en terminos del os resultados  de sobrevivientes y no sobrevivientes''')

