from controller import Supervisor
import numpy as np

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Get the e-puck robot
epuck = robot.getFromDef("Agent")
if epuck is None:
    print("Error: Could not find robot with DEF 'Agent'")
    exit(1)

trans_field = epuck.getField("translation")
rot_field = epuck.getField("rotation")

# Correct ground Z height for e-puck (not 0!)
GROUND_Z = 0.0019  # This keeps it on the floor, not flying

# 5x5 Grid positions on the X-Y plane (Z fixed)
positions = [
    [-0.8, -0.8, GROUND_Z],  # Position 0 (start)
    [-0.4, -0.8, GROUND_Z],  # Position 1
    [ 0.0, -0.8, GROUND_Z],  # Position 2
    [ 0.4, -0.8, GROUND_Z],  # Position 3
    [ 0.8, -0.8, GROUND_Z],  # Position 4
    [-0.8, -0.4, GROUND_Z],  # Position 5
    [-0.4, -0.4, GROUND_Z],  # Position 6
    [ 0.0, -0.4, GROUND_Z],  # Position 7
    [ 0.4, -0.4, GROUND_Z],  # Position 8
    [ 0.8, -0.4, GROUND_Z],  # Position 9
    [-0.8,  0.0, GROUND_Z],  # Position 10
    [-0.4,  0.0, GROUND_Z],  # Position 11
    [ 0.0,  0.0, GROUND_Z],  # Position 12 (center)
    [ 0.4,  0.0, GROUND_Z],  # Position 13
    [ 0.8,  0.0, GROUND_Z],  # Position 14
    [-0.8,  0.4, GROUND_Z],  # Position 15
    [-0.4,  0.4, GROUND_Z],  # Position 16
    [ 0.0,  0.4, GROUND_Z],  # Position 17
    [ 0.4,  0.4, GROUND_Z],  # Position 18
    [ 0.8,  0.4, GROUND_Z],  # Position 19
    [-0.8,  0.8, GROUND_Z],  # Position 20
    [-0.4,  0.8, GROUND_Z],  # Position 21
    [ 0.0,  0.8, GROUND_Z],  # Position 22
    [ 0.4,  0.8, GROUND_Z],  # Position 23
    [ 0.8,  0.8, GROUND_Z],  # Position 24 (goal)
]

# Start position (bottom-left corner)
start_index = 0

# Goal position (top-right corner)
goal_index = 24


obstacle_indices = [2, 7, 10, 15]  


Q = np.zeros((len(positions), 4))  
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  

def index_to_coords(index):
    return index // 5, index % 5

def coords_to_index(row, col):
    if 0 <= row < 5 and 0 <= col < 5:
        return row * 5 + col
    return None


episodes = 300
alpha = 0.3
gamma = 0.9
epsilon = 0.5
min_epsilon = 0.05
epsilon_decay = 0.01

for episode in range(episodes):
    state = start_index 
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 50:
        if robot.step(timestep) == -1:
            break

       
        trans_field.setSFVec3f(positions[state])
        rot_field.setSFRotation([0, 1, 0, 0])  

       
        for _ in range(5):
            if robot.step(timestep) == -1:
                break

        
        if state in obstacle_indices:
            reward = -1  
        elif state == goal_index:
            reward = 10 
            done = True
        else:
            reward = -0.1  

        # Choose action
        if np.random.rand() < epsilon:
            action = np.random.choice(4)
        else:
            action = np.argmax(Q[state])

        row, col = index_to_coords(state)
        d_row, d_col = actions[action]
        new_row, new_col = row + d_row, col + d_col
        next_state = coords_to_index(new_row, new_col)

        if next_state is None or next_state in obstacle_indices:
            next_state = state  

        
        Q[state, action] = (1 - alpha) * Q[state, action] + \
            alpha * (reward + gamma * np.max(Q[next_state]))

        state = next_state
        total_reward += reward
        steps += 1

    epsilon = max(min_epsilon, epsilon - epsilon_decay)
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Steps: {steps}")
