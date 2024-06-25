import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Definición de la clase del entorno del rompecabezas
class PuzzleEnv(gym.Env):
    def __init__(self):
        super(PuzzleEnv, self).__init__()
        
        self.rows = 2  # Número de filas
        self.cols = 3  # Número de columnas
        self.n_tiles = self.rows * self.cols  # Número total de fichas
        
        # Espacio de acciones: 0 (arriba), 1 (derecha), 2 (abajo), 3 (izquierda)
        self.action_space = spaces.Discrete(4)
        # Espacio de observación: representa el estado del tablero como una matriz aplanada
        self.observation_space = spaces.Box(low=0, high=self.n_tiles-1, shape=(self.n_tiles,), dtype=np.int32)
        
        self.reset()
    
    def reset(self):
        # Inicializa el estado del tablero con fichas aleatorias
        self.state = np.arange(self.n_tiles)
        np.random.shuffle(self.state)
        return self.state, {}
    
    def step(self, action):
        # Encuentra la posición del espacio en blanco (representado por 0)
        blank_index = np.where(self.state == 0)[0][0]
        row, col = divmod(blank_index, self.cols)
        
        # Mueve el espacio en blanco según la acción tomada
        if action == 0 and row > 0:  # Mover arriba
            new_blank = blank_index - self.cols
        elif action == 1 and col < self.cols - 1:  # Mover derecha
            new_blank = blank_index + 1
        elif action == 2 and row < self.rows - 1:  # Mover abajo
            new_blank = blank_index + self.cols
        elif action == 3 and col > 0:  # Mover izquierda
            new_blank = blank_index - 1
        else:
            new_blank = blank_index
        
        # Intercambia la posición del espacio en blanco con la nueva posición
        self.state[blank_index], self.state[new_blank] = self.state[new_blank], self.state[blank_index]
        
        # Verifica si el estado actual es el estado objetivo
        done = np.array_equal(self.state, np.arange(self.n_tiles))
        # Recompensa: 1 si está resuelto, -1 en caso contrario
        reward = 1 if done else -1
        
        return self.state, reward, done, False, {}
    
    def render(self):
        # Muestra el estado actual del tablero en la consola
        for i in range(self.rows):
            print(self.state[i*self.cols:(i+1)*self.cols])
        print()

# Función para convertir el estado a una tupla (necesario para usarlo como clave en un diccionario)
def state_to_tuple(state):
    return tuple(state)

# Inicializar el entorno
env = PuzzleEnv()

# Parámetros del Q-Learning
n_episodes = 5000  # Número de episodios de entrenamiento
max_steps = 1000  # Número máximo de pasos por episodio
learning_rate = 0.1  # Tasa de aprendizaje
discount_factor = 0.99  # Factor de descuento
exploration_rate = 1.0  # Tasa de exploración inicial
max_exploration_rate = 1.0  # Tasa de exploración máxima
min_exploration_rate = 0.01  # Tasa de exploración mínima
exploration_decay_rate = 0.001  # Tasa de decaimiento de la exploración

# Inicializar la tabla Q
q_table = {}

# Entrenamiento del agente
for episode in range(n_episodes):
    state, _ = env.reset()
    state_tuple = state_to_tuple(state)
    done = False
    
    for step in range(max_steps):
        # Inicializar la entrada en la tabla Q si no existe
        if state_tuple not in q_table:
            q_table[state_tuple] = np.zeros(env.action_space.n)
        
        # Selección de la acción (exploración vs explotación)
        if np.random.random() < exploration_rate:
            action = env.action_space.sample()  # Acción aleatoria (exploración)
        else:
            action = np.argmax(q_table[state_tuple])  # Mejor acción conocida (explotación)
        
        # Ejecutar la acción y observar el nuevo estado y la recompensa
        new_state, reward, done, _, _ = env.step(action)
        new_state_tuple = state_to_tuple(new_state)
        
        # Inicializar la entrada en la tabla Q para el nuevo estado si no existe
        if new_state_tuple not in q_table:
            q_table[new_state_tuple] = np.zeros(env.action_space.n)
        
        # Actualizar la tabla Q utilizando la ecuación de Bellman
        q_table[state_tuple][action] = q_table[state_tuple][action] + learning_rate * (
            reward + discount_factor * np.max(q_table[new_state_tuple]) - q_table[state_tuple][action]
        )
        
        state = new_state
        state_tuple = new_state_tuple
        
        if done:
            break
    
    # Reducir la tasa de exploración
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    
    # Mostrar progreso cada 1000 episodios
    if episode % 1000 == 0:
        print(f"Episodio {episode}")

# Probar el modelo entrenado
state, _ = env.reset()
env.render()

for _ in range(max_steps):
    state_tuple = state_to_tuple(state)
    if state_tuple in q_table:
        action = np.argmax(q_table[state_tuple])  # Seleccionar la mejor acción según la tabla Q
    else:
        action = env.action_space.sample()  # Acción aleatoria si el estado no está en la tabla Q
    
    state, reward, done, _, _ = env.step(action)
    env.render()
    
    if done:
        print("¡Rompecabezas resuelto!")
        break
