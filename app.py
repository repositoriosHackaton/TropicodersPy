from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, PrecisionRecallDisplay
from scipy.sparse import issparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.calibration import CalibratedClassifierCV
from imblearn.under_sampling import RandomUnderSampler

app = Flask(__name__)

# Funciones de carga y preprocesamiento de datos
def cargar_datos():
    riesgo = pd.read_excel('Cartera de créditos por clasificación de riesgo.xlsx')
    genero = pd.read_excel('Cartera de créditos por genero.xlsx')
    localidad = pd.read_excel('Cartera de créditos por localidad.xlsx')
    sectores = pd.read_excel('Cartera de créditos por sectores económicos.xlsx')
    return riesgo, genero, localidad, sectores

def validar_datos(df):
    df = df.drop_duplicates()
    if df.isnull().sum().sum() > 0:
        print("Datos nulos encontrados, se procederá con la imputación.")
    else:
        print("No se encontraron datos nulos.")
    return df

def preprocesar_datos(riesgo, genero, localidad, sectores, sample_size=5000):
    riesgo = riesgo.sample(n=min(sample_size, len(riesgo)), random_state=42)
    genero = genero.sample(n=min(sample_size, len(genero)), random_state=42)
    localidad = localidad.sample(n=min(sample_size, len(localidad)), random_state=42)
    sectores = sectores.sample(n=min(sample_size, len(sectores)), random_state=42)

    suffixes = ('_riesgo', '_genero', '_localidad', '_sectores')
    df = riesgo.merge(genero, on=['periodo', 'tipoEntidad', 'entidad'], how='inner', suffixes=(suffixes[0], suffixes[1]))
    df = df.merge(localidad, on=['periodo', 'tipoEntidad', 'entidad'], how='inner', suffixes=('', suffixes[2]))
    df = df.merge(sectores, on=['periodo', 'tipoEntidad', 'entidad'], how='inner', suffixes=('', suffixes[3]))

    df['moroso'] = (df['deudaVencida' + suffixes[0]] > 0).astype(int)

    df = validar_datos(df)

    features = [
        'tasaPorDeuda' + suffixes[0],
        'tasaPromedioPonderado' + suffixes[0],
        'deudaCapital' + suffixes[0],
        'valorDesembolso' + suffixes[0],
        'valorGarantia' + suffixes[0],
        'genero',
        'region',
        'sectorEconomico'
    ]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df['moroso']

    return X, y

def preparar_datos(X, y, sampling_strategy=0.5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('normalizer', MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_preprocessed, y_train)
    
    if issparse(X_train_resampled):
        X_train_resampled = X_train_resampled.toarray()
    if issparse(X_test_preprocessed):
        X_test_preprocessed = X_test_preprocessed.toarray()
    
    return X_train_resampled, X_test_preprocessed, y_train_resampled, y_test, preprocessor

# Funciones de modelos
def modelo_red_neuronal(X_train, X_test, y_train, y_test, preprocessor):
    input_dim = X_train.shape[1]
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    y_proba = model.predict(X_test)
    y_pred = (y_proba > 0.5).astype(int)
    
    return y_pred, y_proba, model

def modelo_svm(X_train, X_test, y_train, y_test, preprocessor):
    svm_model = LinearSVC(random_state=42)
    
    param_grid = {
        'C': [0.1, 1, 10],
    }
    
    grid_search = GridSearchCV(svm_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    calibrated_svc = CalibratedClassifierCV(best_model, cv=3)
    calibrated_svc.fit(X_train, y_train)
    
    y_pred = calibrated_svc.predict(X_test)
    y_proba = calibrated_svc.predict_proba(X_test)[:, 1]
    
    return y_pred, y_proba, calibrated_svc

def modelo_knn(X_train, X_test, y_train, y_test, preprocessor):
    knn_model = KNeighborsClassifier()
    
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }
    
    grid_search = GridSearchCV(knn_model, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    return y_pred, y_proba, best_model

def modelo_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba, model

# Funciones de gráficos
def graficar_roc_auc(y_test, y_proba):
    plt.figure(figsize=(10, 6))
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title('Curva ROC')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def graficar_matriz_confusion(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, display_labels=["No Moroso", "Moroso"])
    plt.title("Matriz de Confusión")
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def graficar_metricas(accuracy, precision, recall, f1):
    metrics = ['Exactitud', 'Precisión', 'Sensibilidad', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
    ax.set_ylabel('Puntuación')
    ax.set_title('Métricas de Evaluación')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def graficar_precision_recall(y_test, y_proba):
    plt.figure(figsize=(10, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title('Curva Precisión-Recall')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def graficar_importancia_caracteristicas(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title('Importancia de Características')
    plt.bar(range(len(importances)), importances[indices], color='b', align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Rutas de Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cargar_datos', methods=['POST'])
def cargar_datos_route():
    sample_size = int(request.form['sample_size'])
    riesgo, genero, localidad, sectores = cargar_datos()
    X, y = preprocesar_datos(riesgo, genero, localidad, sectores, sample_size)
    X_train, X_test, y_train, y_test, preprocessor = preparar_datos(X, y)
    
    global g_X_train, g_X_test, g_y_train, g_y_test, g_preprocessor
    g_X_train, g_X_test, g_y_train, g_y_test, g_preprocessor = X_train, X_test, y_train, y_test, preprocessor
    
    return jsonify({'message': 'Datos cargados y preprocesados correctamente'})

@app.route('/entrenar_modelo', methods=['POST'])
def entrenar_modelo():
    modelo = request.form['modelo']
    if modelo == 'nn':
        y_pred, y_proba, model = modelo_red_neuronal(g_X_train, g_X_test, g_y_train, g_y_test, g_preprocessor)
    elif modelo == 'svm':
        y_pred, y_proba, model = modelo_svm(g_X_train, g_X_test, g_y_train, g_y_test, g_preprocessor)
    elif modelo == 'knn':
        y_pred, y_proba, model = modelo_knn(g_X_train, g_X_test, g_y_train, g_y_test, g_preprocessor)
    elif modelo == 'rf':
        y_pred, y_proba, model = modelo_random_forest(g_X_train, g_X_test, g_y_train, g_y_test)
    else:
        return jsonify({'error': 'Modelo no reconocido'})
    
    global g_y_pred, g_y_proba, g_model
    g_y_pred, g_y_proba, g_model = y_pred, y_proba, model
    
    accuracy = accuracy_score(g_y_test, y_pred)
    precision = precision_score(g_y_test, y_pred)
    recall = recall_score(g_y_test, y_pred)
    f1 = f1_score(g_y_test, y_pred)
    
    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

@app.route('/generar_grafico', methods=['POST'])
def generar_grafico():
    tipo_grafico = request.form['tipo_grafico']
    
    if tipo_grafico == 'roc':
        img = graficar_roc_auc(g_y_test, g_y_proba)
    elif tipo_grafico == 'confusion':
        img = graficar_matriz_confusion(g_y_test, g_y_pred)
    elif tipo_grafico == 'metricas':
        accuracy = accuracy_score(g_y_test, g_y_pred)
        precision = precision_score(g_y_test, g_y_pred)
        recall = recall_score(g_y_test, g_y_pred)
        f1 = f1_score(g_y_test, g_y_pred)
        img = graficar_metricas(accuracy, precision, recall, f1)
    elif tipo_grafico == 'precision_recall':
        img = graficar_precision_recall(g_y_test, g_y_proba)
    elif tipo_grafico == 'importancia':
        if isinstance(g_model, RandomForestClassifier):
            feature_names = g_X_train.columns if hasattr(g_X_train, 'columns') else np.arange(g_X_train.shape[1])
            img = graficar_importancia_caracteristicas(g_model, feature_names)
        else:
            return jsonify({'error': 'Importancia de características solo disponible para RandomForest'})
    else:
        return jsonify({'error': 'Tipo de gráfico no reconocido'})
    
    return jsonify({'image': img})

if __name__ == '__main__':
    app.run(debug=True)
