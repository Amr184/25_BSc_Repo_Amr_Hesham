import numpy as np 
import matplotlib.pyplot as plt
import random
import time

GRID_SIZE = 10
OBSTACLES = [(2, 3), (3, 3), (4, 3), (5, 3), (7, 7), (8, 2), (1, 8)]
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

num_agents = 3
num_targets = 5

# PSO Parameters
w, c1, c2 = 0.7, 1.5, 1.5
commitment_time = 5
min_distance = 1.0
target_threshold = 0.5
max_no_progress = 10  

# RL Parameters
alpha = 0.3
gamma = 0.9
epsilon_start, epsilon_decay, min_epsilon = 0.9, 0.995, 0.1

class RLAgent:
    def __init__(self, start_pos, agent_id):
        self.pos = start_pos
        self.id = agent_id
        self.target = None
        self.target_id = None
        self.Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
        self.epsilon = epsilon_start
        self.other_agents_pos = []
        self.commitment = 0
        self.last_distance = float('inf')
        self.no_progress_counter = 0

    def update_other_agents(self, positions):
        self.other_agents_pos = [pos for idx, pos in enumerate(positions) if idx != self.id]

    def choose_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS) - 1)
        return np.argmax(self.Q[self.pos[0], self.pos[1]])

    def update_q(self, action, reward, next_pos):
        max_next_q = np.max(self.Q[next_pos[0], next_pos[1]])
        self.Q[self.pos[0], self.pos[1], action] = (1 - alpha) * self.Q[self.pos[0], self.pos[1], action] + alpha * (reward + gamma * max_next_q)

    def move(self, action):
        dx, dy = ACTIONS[action]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy

        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
            return False  
        if (nx, ny) in OBSTACLES:
            return False  
        if any(np.linalg.norm(np.array((nx, ny)) - np.array(p)) < min_distance for p in self.other_agents_pos):
            return False  

        self.pos = (nx, ny)
        return True

def initialize_positions():
    positions = set()
    agents = []
    for i in range(num_agents):
        while True:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if pos not in OBSTACLES and pos not in positions:
                agents.append(RLAgent(pos, i))
                positions.add(pos)
                break

    targets = []
    for _ in range(num_targets):
        while True:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if pos not in OBSTACLES and pos not in positions:
                targets.append(pos)
                positions.add(pos)
                break
    return agents, targets

def hybrid_pso_rl():
    agents, targets = initialize_positions()
    collected = [False] * num_targets

    agent_distances = [0] * num_agents
    agent_collected_counts = [0] * num_agents
    previous_positions = [agent.pos for agent in agents]

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    for iteration in range(300):
        curr_positions = [agent.pos for agent in agents]
        for agent in agents:
            agent.update_other_agents(curr_positions)

        for agent in agents:
            if agent.target is None or agent.commitment <= 0 or collected[agent.target_id]:
                min_dist = float('inf')
                nearest_target, nearest_id = None, -1
                for j, target in enumerate(targets):
                    if not collected[j]:
                        dist = abs(agent.pos[0] - target[0]) + abs(agent.pos[1] - target[1])
                        if dist < min_dist:
                            min_dist = dist
                            nearest_target = target
                            nearest_id = j
                agent.target = nearest_target
                agent.target_id = nearest_id
                agent.commitment = commitment_time
                agent.last_distance = float('inf')
                agent.no_progress_counter = 0

        for agent in agents:
            if agent.target is None or agent.target_id == -1 or collected[agent.target_id]:
                continue

            if np.linalg.norm(np.array(agent.pos) - np.array(agent.target)) <= target_threshold:
                collected[agent.target_id] = True
                agent_collected_counts[agent.id] += 1  # ✅ Track collection
                agent.update_q(0, 10, agent.pos)
                agent.target = None
                agent.target_id = -1
                continue

            old_dist = abs(agent.pos[0] - agent.target[0]) + abs(agent.pos[1] - agent.target[1])
            action = agent.choose_action()
            moved = agent.move(action)

            if moved:
               
                dx = abs(agent.pos[0] - previous_positions[agent.id][0])
                dy = abs(agent.pos[1] - previous_positions[agent.id][1])
                agent_distances[agent.id] += dx + dy
                previous_positions[agent.id] = agent.pos

                new_dist = abs(agent.pos[0] - agent.target[0]) + abs(agent.pos[1] - agent.target[1])
                if new_dist < agent.last_distance:
                    reward = 1.0
                    agent.no_progress_counter = 0
                    agent.last_distance = new_dist
                else:
                    reward = -0.5
                    agent.no_progress_counter += 1
                agent.update_q(action, reward, agent.pos)
            else:
                reward = -2
                agent.no_progress_counter += 1

            if agent.no_progress_counter >= max_no_progress:
                agent.target = None
                agent.target_id = -1
                agent.no_progress_counter = 0
                agent.epsilon = 1.0  

            agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
            agent.commitment -= 1

        # Visualization
        ax.clear()
        ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
        ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
        ax.grid(True)

        for x, y in OBSTACLES:
            ax.add_patch(plt.Rectangle((y, GRID_SIZE - x - 1), 1, 1, color='black'))

        for i, (x, y) in enumerate(targets):
            if not collected[i]:
                ax.add_patch(plt.Rectangle((y, GRID_SIZE - x - 1), 1, 1, color='green'))

        for i, agent in enumerate(agents):
            x, y = agent.pos
            ax.add_patch(plt.Circle((y + 0.5, GRID_SIZE - x - 0.5), 0.4, color=['red', 'blue', 'purple'][i]))
            if agent.target is not None and agent.target_id != -1 and not collected[agent.target_id]:
                ax.plot([y + 0.5, agent.target[1] + 0.5],
                        [GRID_SIZE - x - 0.5, GRID_SIZE - agent.target[0] - 0.5],
                        linestyle='dashed', color='gray')

        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_title(f"Iteration {iteration}")
        plt.pause(0.01)

        if all(collected):
            print("All targets collected!")
            break

        time.sleep(0.05)

    plt.ioff()
    plt.show()

    # Final stats visualization
    print("\n--- Agent Stats ---")
    for i in range(num_agents):
        print(f"Agent {i}: Distance Traveled = {agent_distances[i]}, Targets Collected = {agent_collected_counts[i]}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(num_agents), agent_distances, color=['red', 'blue', 'purple'])
    plt.xticks(range(num_agents), [f'Agent {i}' for i in range(num_agents)])
    plt.title('Distance Traveled per Agent')
    plt.ylabel('Total Distance')

    plt.subplot(1, 2, 2)
    plt.bar(range(num_agents), agent_collected_counts, color=['red', 'blue', 'purple'])
    plt.xticks(range(num_agents), [f'Agent {i}' for i in range(num_agents)])
    plt.title('Targets Collected per Agent')
    plt.ylabel('Number of Targets')

    plt.tight_layout()
    plt.show()

hybrid_pso_rl()
