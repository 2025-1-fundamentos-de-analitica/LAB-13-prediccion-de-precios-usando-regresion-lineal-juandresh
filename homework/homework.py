#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.

import pandas as pd
import gzip
import json
import os
import zipfile
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
def preprocess_data(df):

    df = df.copy()

    df['Age'] = 2021 - df['Year']
    df = df.drop(columns=['Year', 'Car_Name'])
    
    return df

#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#

def split_data(df):

    y = df['Present_Price']
    x = df.drop(columns=['Present_Price'])
    
    return x, y

#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#

def create_pipeline():

    categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
    numerical_features = ['Selling_Price', 'Driven_kms', 'Owner', 'Age']


    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('num', MinMaxScaler(), numerical_features)
        ]
    )


    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_regression)),
        ('regressor', LinearRegression())
    ])
    
    return pipeline

#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#

def optimize_hyperparameters(pipeline, x_train, y_train):

    param_grid = {
        'feature_selection__k': [4, 5, 6, 7, 8, 9, 10, 11],
        'regressor__fit_intercept': [True, False],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    grid_search.fit(x_train, y_train)
    
    return grid_search

#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#

def save_model(model, filename):
    with gzip.open(filename, 'wb') as f:
        pd.to_pickle(model, f)

# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

def calculate_metrics(model, x_train, y_train, x_test, y_test):
    metrics = []

    y_train_pred = model.predict(x_train)
    train_metrics = {
        'type': 'metrics',
        'dataset': 'train',
        'r2': r2_score(y_train, y_train_pred),
        'mse': mean_squared_error(y_train, y_train_pred),
        'mad': median_absolute_error(y_train, y_train_pred)
    }
    metrics.append(train_metrics)

    y_test_pred = model.predict(x_test)
    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mad': median_absolute_error(y_test, y_test_pred)
    }
    metrics.append(test_metrics)

    return metrics


def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')


def main():

    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)
    
    with zipfile.ZipFile('files/input/train_data.csv.zip','r') as comp:
        train_data = comp.namelist()[0]
        with comp.open(train_data) as arch:
            df_train = pd.read_csv(arch)

    with zipfile.ZipFile('files/input/test_data.csv.zip','r') as comp:
        test_data = comp.namelist()[0]
        with comp.open(test_data) as arch:
            df_test = pd.read_csv(arch)

    df_train = preprocess_data(df_train)
    df_test = preprocess_data(df_test)

    # print(df_train.columns)
    # print(df_test.columns)

    x_train, y_train = split_data(df_train)
    x_test, y_test = split_data(df_test)

    pipeline = create_pipeline()
    best_model = optimize_hyperparameters(pipeline, x_train, y_train)

    save_model(best_model, 'files/models/model.pkl.gz')
    metrics = calculate_metrics(best_model, x_train, y_train, x_test, y_test)
    save_metrics(metrics, 'files/output/metrics.json')

if __name__ == "__main__":
    main()