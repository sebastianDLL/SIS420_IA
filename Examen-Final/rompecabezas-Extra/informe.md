# EXAMEN FINAL SIS-420
## DELGADILLO LLANOS JUAN SEBASTIAN
### Repositorio: [Github](https://github.com/sebastianDLL/SIS420_IA) 

## informe sobre el Entrenamiento de un Agente para Resolver un Rompecabezas

En este informe, se presenta el desarrollo y entrenamiento de un agente de aprendizaje por refuerzo para resolver un rompecabezas en un entorno simulado. El agente está diseñado para moverse en un tablero de tamaño 4x5 con el objetivo de alcanzar una casilla objetivo que cambia aleatoriamente en cada episodio.

### Descripción del Entorno
El entorno del rompecabezas se implementó utilizando la librería Gymnasium, con las siguientes características:

### Espacio de Acción: 
El agente puede realizar cuatro acciones: moverse hacia arriba, abajo, izquierda o derecha en el tablero 4x5.

### Espacio de Observación: 
El estado del entorno se representa como una tupla de dos valores, la posición actual del agente en el tablero.

### Recompensas: 
El agente recibe una recompensa de 1 al alcanzar la casilla objetivo, y una penalización de -0.1 en cada paso no exitoso.

## Estrategia de Entrenamiento
El agente se entrenó utilizando el algoritmo Q-learning con valores iniciales optimistas. Se inicializó la tabla Q con valores de 10 en lugar de ceros para fomentar la exploración al principio del entrenamiento. Se utilizaron los siguientes hiperparámetros:

- Número de episodios: 300
- Tasa de aprendizaje: 0.1
- Factor de descuento: 0.99
- Épsilon inicial: 1.0
- Tasa de decaimiento de épsilon: 0.995

## Valores iniciales optimistas
Los métodos hasta ahora dependían de las estimaciones iniciales Q1(a). 

Esto nos introduce un bias que implica que en la práctica la inicialización se convierte en un hiperparámetro más que el ususario debe escoger. Podemos usar este hecho en nuestra ventaja. Si inicializamos los valores de las acciones por encima de sus valores reales, nuestro agente intentrá llevar a cabo estas acciones, y enseguida las descartará ya que la recompensa obtenida será menor y por lo tanto su valor disminuirá. 

Esto significa que includo un agente greedy probará todas las acciones antes de quedarse con una, habiendo explorado un poco.

Ahora, nuestro agente greedy explora todas las acciones antes de elegir la mejor, la cual explota hasta el final. Esta técnica funciona bien en problemas estacionarios, pero si el problema no es estacionario y las recompensas van cambiando durante el tiempo tendremos el mismo problema que antes con un agente que no explora.

## Implementación
Durante el entrenamiento, el agente realizó acciones basadas en una política epsilon-greedy, donde la exploración se controla mediante el valor de épsilon. La tabla Q se actualizó en cada paso utilizando la ecuación de actualización Q-learning.

## Resultados y Conclusiones
El agente logró aprender una política efectiva para resolver el rompecabezas en el entorno simulado. Se observó un incremento en las recompensas por episodio a lo largo del entrenamiento, lo que indica que el agente mejoró su desempeño con el tiempo. Al finalizar el entrenamiento, el agente fue capaz de resolver el rompecabezas de manera consistente, alcanzando la casilla objetivo en la mayoría de los episodios.