# SIS420 - I.A.
## Alumno: Delgadillo Llanos Juan Sebastian 

Implementación de AxR con el entorno [Gymnasium](https://gymnasium.farama.org/) .

Repositorio: [GITHUB](https://github.com/sebastianDLL/SIS420_IA)

# Informe del código: Algoritmo de Aprendizaje Q-Learning para el entorno Taxi-v3 de Gymnasium

## Descripción General
### PRIMERO ("AxR_Taxi.py"): 
El código implementa un algoritmo de aprendizaje Q-learning para entrenar un agente en el entorno Taxi-v3 de Gymnasium. El objetivo del agente es aprender la mejor política para maximizar la recompensa acumulada al completar tareas de recoger y dejar pasajeros en ubicaciones específicas dentro de una cuadrícula, una vez entrenado se mostrara una imagen de las recompensas obtenidas a lo largo del entrenamiento.

### SEGUNDO ("AxR_Taxi_pkl.py"): 
El código implementa un algoritmo de aprendizaje Q-learning para entrenar un agente en el entorno Taxi-v3 de Gymnasium. El objetivo del agente es aprender la mejor política para maximizar la recompensa acumulada al completar tareas de recoger y dejar pasajeros en ubicaciones específicas dentro de una cuadrícula, una vez entrenado se guardara la tabla Q en un archivo "taxi.pkl", el cual dependiendo del modo de ejecucion se explorará o se explotará, tambien se guardara una imagen "taxi.png" de las recompensas obtenidas a lo largo del entrenamiento.


## Algoritmo Utilizado: Q-Learning
El algoritmo Q-learning es un método de aprendizaje por refuerzo que busca aprender la función Q, la cual estima la calidad de una acción en un estado determinado. Este método se basa en la ecuación de Bellman, que actualiza los valores de la función Q iterativamente:

![formula de Bellman](/imagenes/formula.png)


## Estructura del Código
### Importación de Librerías:
```bash
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

```

### Función de Entrenamiento e inicialización del Entorno:
```bash
def train(episodes):

env = gym.make("Taxi-v3")
```


### Inicialización de la Tabla Q y definición de Parámetros:
```bash
q_table = np.zeros((env.observation_space.n, env.action_space.n))

learning_rate = 0.3
discount_factor = 0.9
epsilon = 1.0
epsilon_decay_rate = 0.0003
rng = np.random.default_rng()

```

### Bucle de Entrenamiento y reinicio del entorno:
```bash
for i in range(episodes):


if (i + 1) % 100 == 0:
    env.close()
    env = gym.make("Taxi-v3", render_mode="human")
else:
    env.close()
    env = gym.make("Taxi-v3")
state = env.reset()[0]

```

### Bucle de Pasos dentro de un Episodio:
```bash
while not terminated and not truncated:
    if rng.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state, :])
    new_state, reward, terminated, truncated, _ = env.step(action)
    q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
    state = new_state
```

### Actualización de Epsilon:
```bash
epsilon = max(epsilon - epsilon_decay_rate, 0.01)
rewards_por_episode[i] = reward

```


### Visualización de Resultados:
```bash
suma_rewards = np.zeros(episodes)
for t in range(episodes):
    suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 100) :(t + 1)])
plt.plot(suma_rewards)
plt.xlabel('Episodios')
plt.ylabel('Suma de recompensas acumuladas')
plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
plt.show()

```

## Conclusiones
El código implementa el algoritmo Q-learning para entrenar un agente en el entorno Taxi-v3. Utiliza una tabla Q para almacenar los valores de calidad de las acciones en cada estado, que se actualizan iterativamente mediante la ecuación de Bellman. La estrategia epsilon-greedy permite un balance entre exploración y explotación, ajustándose con el tiempo para mejorar la política del agente. Los resultados se visualizan al final del entrenamiento para evaluar el rendimiento del agente.
