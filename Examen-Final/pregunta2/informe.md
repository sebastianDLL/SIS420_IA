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
El agente recibe una recompensa de 1 positiva cuando el tablero esta ordenado y si no lo esta una negativa

## Estrategia de Entrenamiento


# Parámetros del Q-Learning
- n_episodes = 10000  # Número de episodios de entrenamiento
- max_steps = 1000  # Número máximo de pasos por episodio
- learning_rate = 0.1  # Tasa de aprendizaje
- discount_factor = 0.99  # Factor de descuento
- exploration_rate = 1.0  # Tasa de exploración inicial
- max_exploration_rate = 1.0  # Tasa de exploración máxima
- min_exploration_rate = 0.01  # Tasa de exploración mínima
- exploration_decay_rate = 0.001  # Tasa de decaimiento de la exploración

## Ecuacion de Bellman



## Implementación
Durante el entrenamiento, el agente realizó acciones basadas en una política epsilon-greedy, donde la exploración se controla mediante el valor de épsilon. La tabla Q se actualizó en cada paso utilizando la ecuación de actualización Q-learning.

## Resultados y Conclusiones
El agente logró aprender una política efectiva para resolver el rompecabezas en el entorno simulado. Se observó un incremento en las recompensas por episodio a lo largo del entrenamiento, lo que indica que el agente mejoró su desempeño con el tiempo. Al finalizar el entrenamiento, el agente fue capaz de resolver el rompecabezas de manera consistente, alcanzando la casilla objetivo en la mayoría de los episodios.