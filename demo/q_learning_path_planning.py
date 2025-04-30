import numpy as np
import matplotlib.pyplot as plt
import time

GRID_SIZE = 7
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
OBSTACLES = [(2, 3), (3, 3), (4, 3), (5, 3)]  
START = (0, 0)
GOAL = (6, 6)

Q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
alpha = 0.3  
gamma = 0.9  
epsilon = 0.5 
epsilon_decay = 0.005  
min_epsilon = 0.05  
episodes = 500 


fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.grid(True)


for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        if (x, y) in OBSTACLES:
            ax.add_patch(plt.Rectangle((y, GRID_SIZE-x-1), 1, 1, color='black'))
        elif (x, y) == GOAL:
            ax.add_patch(plt.Rectangle((y, GRID_SIZE-x-1), 1, 1, color='green'))

agent_marker, = ax.plot([], [], "ro", markersize=10)  

def update_visual(state):
    agent_marker.set_data([state[1] + 0.5], [GRID_SIZE - state[0] - 0.5])
    plt.pause(0.001)


moves_per_episode = []

for episode in range(episodes):
    state = START
    total_reward = 0
    moves = 0
    
    while state != GOAL and moves < 3000:  
        update_visual(state)

        if np.random.rand() < epsilon:
            action = np.random.choice(len(ACTIONS))  
        else:
            action = np.argmax(Q_table[state[0], state[1]])  

        next_state = (state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1])
        
        if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
            next_state = state  
        if next_state in OBSTACLES:
            next_state = state  
        
        reward = 1 if next_state == GOAL else -0.1  

        Q_table[state[0], state[1], action] = (1 - alpha) * Q_table[state[0], state[1], action] + \
            alpha * (reward + gamma * np.max(Q_table[next_state[0], next_state[1]]))
        
        state = next_state
        total_reward += reward
        moves += 1

    moves_per_episode.append(moves)
    epsilon = max(min_epsilon, epsilon - epsilon_decay)  
    print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Moves: {moves}")

plt.show()


plt.figure(figsize=(8, 4))
plt.plot(moves_per_episode, label="Moves per Episode", color="blue")
plt.xlabel("Episode")
plt.ylabel("Number of Moves")
plt.title("Q-Learning Performance Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
