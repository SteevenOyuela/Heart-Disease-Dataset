# Predicción de Enfermedad Cardíaca con Perceptrón

## 📘 Descripción General

Este proyecto corresponde a un trabajo académico de la asignatura **Fundamentos de Inteligencia Artificial**, enfocado en aplicar conceptos de **aprendizaje supervisado** mediante redes neuronales para resolver un problema de clasificación.

Se desarrolló un modelo capaz de analizar variables clínicas de pacientes y predecir si existe presencia o ausencia de enfermedad cardíaca.

El proyecto utiliza el dataset de UCI Machine Learning Repository y construye un perceptrón utilizando TensorFlow y Keras.

---

## 🎯 Objetivos del Proyecto

* Comprender el funcionamiento del aprendizaje supervisado.
* Implementar un modelo de clasificación binaria.
* Diseñar y entrenar un perceptrón.
* Aplicar preprocesamiento y normalización de datos.
* Evaluar el desempeño del modelo con métricas de clasificación.
* Analizar relaciones entre variables clínicas mediante EDA.

---

## 🧠 Fundamento Teórico

### Aprendizaje Supervisado

En aprendizaje supervisado:

* Existen datos de entrada:

$$
X
$$

* Existe una salida esperada:

$$
y
$$

* El modelo aprende una función:

$$
f(X)=y
$$

para realizar predicciones sobre nuevos casos.

---

### Clasificación Binaria

Este proyecto resuelve un problema con dos clases:

* 0 → No enfermedad cardíaca
* 1 → Presencia de enfermedad cardíaca

Esto corresponde a clasificación binaria.

---

### Perceptrón

El modelo implementado es un perceptrón simple.

Calcula:

$$
z=wX+b
$$

y luego aplica función de activación:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

usando **sigmoid** para producir probabilidades.

---

### Función de Pérdida

Se usa entropía cruzada binaria:

$$
L=-(y\log(p)+(1-y)\log(1-p))
$$

para medir error.

---

### Optimización con Descenso de Gradiente

Se utiliza:

* SGD (Stochastic Gradient Descent)

para actualizar:

* Pesos
* Bias

minimizando la pérdida.

---

## ⚙️ Implementación Práctica

### Carga de Datos

```python
df_heart = pd.read_csv("Data/heart.csv")
```

Contiene variables como:

* edad
* colesterol
* presión arterial
* frecuencia cardíaca
* dolor de pecho
* variable objetivo target

---

### Preprocesamiento

Se realizó:

* Revisión de valores nulos
* Separación de variables predictoras y objetivo
* Normalización Min-Max
* División en entrenamiento y prueba

```python
x_train
x_test
```

---

### Análisis Exploratorio (EDA)

Se analizaron patrones mediante:

* Matriz de correlación
* Boxplots
* Histogramas
* Violinplots
* Cross-tabulations

---

### Construcción del Modelo

```python
model = keras.Sequential([
layers.Dense(
units=1,
activation="sigmoid"
)
])
```

---

### Compilación del Modelo

```python
model.compile(
optimizer=SGD(),
loss="binary_crossentropy",
metrics=["accuracy"]
)
```

---

### Entrenamiento

```python
model.fit(
x_train_norm,
y_train,
epochs=100
)
```

Durante esta fase:

* se ajustan pesos
* se minimiza pérdida
* el modelo aprende patrones.

---

### Evaluación

```python
model.evaluate()
```

Se mide:

* Accuracy
* Loss

---

### Matriz de Confusión

Permite analizar:

* Verdaderos positivos
* Verdaderos negativos
* Falsos positivos
* Falsos negativos

---

## ▶️ Ejecución

Para ejecutar:

```bash
jupyter notebook Proyecto.ipynb
```

o

```bash
python main.py
```

---

## 📊 Aprendizajes Obtenidos

A través del proyecto se fortalecieron conocimientos en:

* Aprendizaje supervisado
* Clasificación binaria
* Redes neuronales
* Funciones de activación
* Optimización por gradiente
* Evaluación de modelos
* Análisis exploratorio de datos

---

## 🔧 Mejoras Futuras

* Agregar más capas (red multicapa)
* Comparar con regresión logística
* Probar otros optimizadores
* Ajustar hiperparámetros
* Mejorar precisión con feature engineering
* Implementar validación cruzada

---

## 👨‍💻 Autores
Trabajo realizado por estudiantes de tercer semestre de Ingeniería en Inteligencia Artificial de la Escuela Colombiana de Ingeniería Julio Garavito:

- Andres Steeven Oyuela Mendez, andres.oyuela-m@mail.escuelaing.edu.co
- Juan David Rojas Heredia, juan.rojas-h@mail.escuelaing.edu.co


## 📌 Conclusión

La implementación muestra cómo las redes neuronales pueden utilizarse para resolver problemas reales de predicción médica. La combinación de teoría, análisis de datos y entrenamiento del perceptrón evidencia la aplicación práctica de los fundamentos de inteligencia artificial en clasificación supervisada.
