import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

class PuzzleEnv(gym.Env):
    def __init__(self):
        super(PuzzleEnv, self).__init__()
        self.rows = 4
        self.columns = 5
        self.action_space = spaces.Discrete(4)  # 4 acciones: arriba, abajo, izquierda, derecha
        self.observation_space = spaces.MultiDiscrete([self.rows, self.columns])
        self.state = None
        self.goal = (np.random.randint(0, self.rows), np.random.randint(0, self.columns))  # Posición aleatoria del objetivo
        self.reset()

    def reset(self):
        self.state = (np.random.randint(0, self.rows), np.random.randint(0, self.columns))  # Posición inicial del agente
        return self.state

    def step(self, action):
        row, col = self.state
        if action == 0:  # Arriba
            row = max(0, row - 1)
        elif action == 1:  # Abajo
            row = min(self.rows - 1, row + 1)
        elif action == 2:  # Izquierda
            col = max(0, col - 1)
        elif action == 3:  # Derecha
            col = min(self.columns - 1, col + 1)

        self.state = (row, col)
        done = self.state == self.goal
        reward = 1 if done else -0.1  # Recompensa positiva al alcanzar el objetivo, negativa en otro caso
        return self.state, reward, done, {}

    def render(self):
        grid = np.zeros((self.rows, self.columns))
        row, col = self.state
        grid[row, col] = 1
        grid[self.goal[0], self.goal[1]] = 0.5
        plt.imshow(grid, cmap='gray')
        plt.title("Posición del agente en el rompecabezas")
        plt.show()


def train(env, episodes=1000, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995):
# def train(env, episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.03):

    q_table = np.ones((env.rows, env.columns, env.action_space.n)) * 10  # Valores optimistas
    rewards_per_episode = []


    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state[0], state[1], :])

            new_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Actualizar la tabla Q y encontrar la politica optima o max la funcion de valor con Bellman 
            q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + learning_rate * (
                reward + discount_factor * np.max(q_table[new_state[0], new_state[1], :]) - q_table[state[0], state[1], action]
            )

            state = new_state

        epsilon = max(epsilon * epsilon_decay, 0.01)
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 30 == 0:
            print(f'Episodio: {episode + 1}, Recompensa: {total_reward}')

    # Guardar la tabla Q en un archivo pkl
    with open('puzzle_q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)

    plt.plot(rewards_per_episode)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensas')
    plt.title('Recompensas por Episodio durante el Entrenamiento')
    plt.show()

    return q_table

# Crear el entorno y entrenar el modelo
env = PuzzleEnv()
q_table = train(env, episodes=300)


def run_trained_model(env, q_table, episodes=10):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = np.argmax(q_table[state[0], state[1], :])
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.5)

        print(f'Episodio: {episode + 1}, Recompensa Total: {total_reward}')

# Cargar la tabla Q entrenada y ejecutar el modelo
with open('puzzle_q_table.pkl', 'rb') as f:
    trained_q_table = pickle.load(f)

run_trained_model(env, trained_q_table, episodes=5)
