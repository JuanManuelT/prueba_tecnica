# PRUEBA TÉCNICA
# Juan Manuel Trujillo C
# 27/01/2024

## Contenido

- [Descripción](10)
- [Instalación](11)
- [Uso]()

## Descripción:

Para el desarrollo de la prueba hice uso tanto de Transformer como de modelo de Deep Learning y como PLUS: Hice uso de algoritmos probabilísticos
y 5 de Machine Learning que especificaré a continuación y con la intención de lograr el método adecuado para lograr mejorar la precisión del 
modelo planteado por el Paper.

El reto consiste en lograr una forma eficiente (computacionalmente), sólida, estructurada de lograr un resultado a la altura en los dos días de 
desarrollo límite de la prueba, mi primer hipótesis que lograra cumplir con todos los requisitos fue: Naive Bayes, un algoritmo de Machine Learning
que incorpora lógica probabilística, ¿Y porqué?, Vamos a explorar la lógica detrás del desarrollo del primer modelo desde una visión pragmática, 
haciendo uso del modelo Naive Bayes (Naive Bayes Model).

Antes de empezar cabe aclarar que el funcionamiento, exploración, limpieza y tratamiento de datos tanto para Training como para Testing serán
realizados adecuadamente según buenas prácticas de desarrollo para modelos de Inteligencia Artificial, específicamente Machine Learning & Deep
Learning, el detalle de cada proceso se encuentra denotado en pasos por cada modelo presentado en la entrega al igual que un video explicación
adjunto para el entendimiento de cada uno de estos de manera fácil y concisa, entre los ajustes de los modelos se encuentra un estandar de
separación de datos 80-20 para Training & Testing al igual que métodos de limpieza de datos (punctuación, campos vacíos etc.)

NBM a diferencia del Paper ACL’20 NO es un modelo de Logistic Regression, NBM es un modelo probabilístico, Y ¿porqué entonces abordar un problema
de clasificación binaria (Binary Classification) con un modelo probabilístico?. Primero debemos conocer la forma en la que los datos origen
se distribuyen y en este caso es una Distribución Independiente (Independent Distribution), para validar dicha afirmación se efectúa un test de 
dependencia como Pearsons chi-cuadrado asumiendo H0: Las variables son independientes y calcular el P-value a un nivel de significancia 
estándar de 0.05, crear una tabla binaria con valores en X y Y entre 0 y 1 según si es una mentira o una verdad de esta forma:

table = np.array([[np.sum((X == 0) & (Y == 0)), np.sum((X == 0) & (Y == 1))], [np.sum((X == 1) & (Y == 0)), np.sum((X == 1) & (Y == 1))]])
chi2, p, _, _ = chi2_contingency(table)

Al igual que el Paper ACL’20, se valida la Independencia, cumplimos entonces el primer requisito para aplicar NBM en el modelo 
(la independencia de datos, de ahí el nombre "Naive"), ahora si, NBM se basa en la teoría de Bayes en el que se puede predecir la categoría de la
data según información previa, siendo un método altamente efectivo para la representación de texto como el análisis de sentimientos, en este caso
NBM toma la AUSENCIA o PRESENCIA de términos en cuenta para su clasificación, lo cual nos conviene enormemente al estar tratando con un conjunto de
datos que según el Paper ACL’20 algunos términos (frankly, sincerely, morning, night, past etc.)inciden en el peso para la clasificación.
Cumplimos por el momento dos atributos clave para el uso de NBM, que además por efectos de la prueba necesitamos que sea "ligero" tanto para
entrenamiento como para predicción de resultados, NBM cumple en ser computacionalmente ligero incluso para High-dimensional data. Finalmente sus
resultados de precisión fueron de 0.86 comparémoslo entonces con otro métodos de Machine Learning que apliquen a nuestro contexto como lo son:
SVC, KNN, GaussianNB, Decision Tree, SGDClassifier & Logistic Regression.Entre los cuales sus resultados fueron de: 0.83, 0.56, 0.69, 0.67, 0.83
y 0.86 respectivamente. Nuestra Hipótesis es correcta frente a otros modelos de ML, sin embargo debemos ajustar el modelo implementando un Transformer
y evaluando su desempeño.

La siguiente aproximación consiste del Transformer BERT usando Transfer Learning, ya que al igual que NBM está diseñado para comprender el contexto y 
los matices de las palabras de una frase teniendo en cuenta tanto el contexto izquierdo como el derecho de cada palabra. Esto le ayuda a entender el 
significado de las palabras basándose en cómo encajan en la frase completa lo que lo hace ideal para tareas como entender el significado de una frase al
igual que NBM. Sin embargo existen dos puntos críticos a tener en cuenta en BERT: 1. requiere GPU para su ejecución bajo unicamente 3 Epoch's (De lo contrario
correrlo tardaría al rededor de 16 horas) para lo que usé Google COLAB, de esta forma hacemos uso de su GPU dedicada en servidor para lograr la ejecución usando
el Modelo BERT, por esto también el exploration, training y testing se encuentran dentro del mismo archivo "Bert_Transformer_Training_Testing" para facilidad de 
quien ejecute. 2. Los datos requeridos para entrenamiento deben ser de un tamaño sustancialmente grande, nuestros datos no son objetivamente cuantiosos
por lo que podría suponer un problema de desempeño. Su resultado fué un poco más bajo de lo esperado con una Validation Accuracy de 0.69, que efectivamente
puede deberse al tamaño de training y testing. Es necesario entonces abordar un desarrollo un poco más robusto para alcanzar un modelo ideal, por lo que
hice uso de un modelo de Red Neuronal (Neural Network) de 4 capas que me permitiera ligereza computacional y un análisis profundo.

Por medio de Keras se construye la Neural Network junto con TensorFlow bajo una estructura lineal de capas, una tras otra, y cada una solamente tiene
conexión con aquella anterior o posterior, la primera convierte enteros positivos (índices) en vectores densos de tamaño fijo, Después se añade una capa
Flatten, utilizada para aplanar la salida de la capa anterior, convirtiéndola en una matriz unidimensional. Posteriormente, Se añaden dos hidden layers 
densas (totalmente conectadas) con 128 y 64 neuronas, respectivamente. La función de activación utilizada es la unidad lineal rectificada (ReLU), una 
opción habitual para hidden layers en redes neuronales. Por último, se añade una capa de salida con una sola neurona y una función de activación sigmoidea.
Esto es adecuado para problemas de clasificación binaria, como lo es en este caso. Tras 15 Epochs el modelo presenta un f1score y una accuracy del 0.89.

Logra entonces superar los 5 modelos de ML, BERT y el algoritmo NBM al alcanzar la mayor aproximación ideal ante el reto planteado.

en Conclusión, es factible ampliar los límites de la aproximación, en una hipótesis inicial: Aumentando el conjunto de datos a analizar, aumentando
ligeramente el número de Epochs y desarrollando mayor número de capas en nuestra Neural Network, que pueden ser explorados en versiones posteriores de la
prueba técnica presentada para el equipo de Gestión de la Operación por parte de Juan Manuel Trujillo.


## Instalación

1. Lo primero es tener la librería de JUPYTER en nuestro entorno de ejecución, para ello (si estamos en VSC se instala como una extensión) Ya que el proyecto fué desarrollado en un Notebook de Jupyter.
2. Debemos instalar el entorno virtual y las librerías que dependen de nuestro proyecto para su correcta ejecución
(Opcional: actualizar el pip siguiendo la ruta alojada de nuestro pc):

```
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
3. Verificamos que el Kernel de Jupyter esté instalado correctamente y en caso de necesitar instalarlo separadamente:

```
pip install ipykernel
```
después de instalado debemos habilitarlo bajo el entorno Jupyter:

```
python -m ipykernel install --user
```

4. OPCIONAL: Eliminamos los archivos: Ya que hacen referencia al archivo del modelo entrenado previamente.
    naive_bayes_model.joblib
    count_vectorizer.joblib


## Uso y Ejecución:

1. El proyecto consiste de 3 modelos: 1. Bert Transformer 2. Naive Bayes Model 3. Neural Network
2. Cada modelo es independiente y posee su propia arquitectura. Sin embargo para ejecutar en orden de implementación debemos...
    2.1: Ejecutar el modelo Naive Bayes, primero siguiendo todos los pasos de la instalación
        2.1.1: Abrimos "Naive_Baysian_Training_and_Exploration" y damos click en "Run all" en la parte superior. Esto generará los archivos
                count_vectorizer.joblib & naive_bayes_model.joblib. Al finalizar visualizaremos las métricas, el resultado y la gráfica comparativa
                con otros modelos de ML.
        2.1.2: Abrimos "Naive_Baysian_Model_Testing" y damos click en "RUN ALL". Como PLUS: agregué la posibilidad de comparar un Input propio
                con el modelo entrenado, con la intención de probarlo con alguno de los textos proporcionados por el dataset del PAPER y validar
                la efectividad del modelo. Digitamos el input y visualizamos los resultados.
            
    2.2: Ejecutar el Transformer BERT:
        2.2.1: Abrimos Google Colab "https://colab.google/"
        2.2.2: Click en "Open Colab" o "Abrir Colab".
        2.2.3: En el menú desplegado click en "Subir" o "Upload" y seleccionamos el archivo "Bert_Transformer_Training_Testing" que posee las 3 etapas
                Exploration, Training and Testing en el mismo archivo.
        2.2.4: una vez subido, en la parte izquierda dar click en el ícono de carpeta llamada "Files" o "Archivos" y click al ícono de subir archivo,
                en el que elijiremos el dataset "deceptive-opinion" para subirlo y damos click en "Aceptar". Listo, damos click en "Entorno de ejecucion"
                y "Correr todo", visualizamos los pasos de ejecución del modelo con Transfer Learning usando el Transformer BERT que también incluye las
                métricas de los modelos ML para comparación y la posibilidad de hacer un test de un input propio. La ejecución dejará dos archivos .joblib

    2.3: Ejecutar Neural Network:
        2.3.1: Abrimos "Neural_Network_Training_and_Exploration" y damos click en "Run all" en la parte superior. Esto generará los archivos
                "lie_detection_model.h5", "label_encoder.joblib" & "count_vectorizer_nn.joblib". Al finalizar visualizaremos las métricas, el resultado y
                la gráfica comparativa con otros modelos de ML.
        2.3.2: Abrimos "Neural_Network_testing" y damos click en "RUN ALL". Como PLUS: agregué la posibilidad de comparar un Input propio
                con el modelo entrenado, con la intención de probarlo con alguno de los textos proporcionados por el dataset del PAPER y validar
                la efectividad del modelo. Digitamos el input y visualizamos los resultados.

3. Finalmente hemos ejecutado satisfactoriamente todo el proyecto de la prueba, el video de demostración presenta información para ayudar en la ejecución en
caso que sea requerido. Hemos terminado.
