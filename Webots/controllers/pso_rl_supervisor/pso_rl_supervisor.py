from controller import Supervisor
import numpy as np
import random
import time
import math

GRID_SIZE = 10
CELL_SIZE = 0.2
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
ACTION_SYMBOLS = ['N', 'E', 'S', 'W']

AGENT_START_POSITIONS = [(0, 0), (0, 1), (1, 0)]

ALPHA = 0.3
GAMMA = 0.9

PRETRAINING_EPISODES = 1000
PRETRAINING_ITERATIONS_PER_EPISODE = 1000
EPSILON_PRETRAIN_START = 0.9
EPSILON_PRETRAIN_END = 0.1

EPSILON_EXECUTION = 0

NUM_AGENTS = 3
NUM_TARGETS = 5
OBSTACLES = [(2, 3), (3, 3), (4, 3), (5, 3), (7, 7), (8, 2), (1, 8)]

AGENT_NAMES = [f"AGENT_{i}" for i in range(NUM_AGENTS)]
TARGET_BOXES_INITIAL = ["box1", "box2", "box3", "box4", "box5"]
ADDITIONAL_BOXES = ["box6", "box7", "box8", "box9", "box10"]
HIDE_POSITION = [3.2, 1.35, 0.026]

MIN_COLLECTION_TIME = 3
MAX_COLLECTION_TIME = 10
target_collection_times = []
target_current_collection_times = []
target_remaining_times = []

MAX_ADDITIONAL_TARGETS = len(ADDITIONAL_BOXES)
SPAWN_INTERVAL_RANGE = (1, 5)

NUM_PARTICLES = 10
MAX_VELOCITY = 2.0
INERTIA_WEIGHT_START = 0.9
INERTIA_WEIGHT_END = 0.4
C1 = 2.0
C2 = 2.0
PSO_ITERATIONS_INITIAL = 50
PSO_ITERATIONS_REALLOC = 25

MIN_STEPS_PER_GRID_MOVE = 10

# Define fixed remaining times for dynamic targets (in seconds)
# You can edit this list to set the expiry time for each dynamic target.
# The length of this list should be equal to MAX_ADDITIONAL_TARGETS.
DYNAMIC_TARGET_EXPIRY_TIMES_SECONDS = [12.0, 8.0, 15.0, 10.0, 9.0] # Example times

targets = []
collected = []
collected_by = []
robots = []
task_allocation = []

NEW_TARGETS_POOL = []
dynamic_targets_spawned_count = 0
next_dynamic_spawn_time = -1
dynamic_spawn_interval_steps = -1

prev_collected_status = []
prev_target_remaining_times = []
expired_target_messages_printed = set() # Set to keep track of printed expiry messages

timestep = -1

# Pre-determined dynamic target spawn intervals (in seconds)
# This sequence will be generated once and used for all training and execution.
# Changed to a fixed 5-second interval for testing
dynamic_spawn_intervals_seconds = [3] * MAX_ADDITIONAL_TARGETS
# Parameters for the New Relearning Phase
RELEARNING_EPISODES = 2000  # Example: Reduced episodes for relearning
RELEARNING_ITERATIONS_PER_EPISODE = 1000 # Example: Reduced iterations
EPSILON_RELEARN_START = 0.5 # Example: Start with higher exploration
EPSILON_RELEARN_END = 0.05 # Example: Decay to a lower value


def get_grid_position_from_coords(coords):
    x = min(max(int(coords[0] / CELL_SIZE), 0), GRID_SIZE - 1)
    y = min(max(int(coords[1] / CELL_SIZE), 0), GRID_SIZE - 1)
    return (x, y)

def get_coords_from_grid_position(grid_pos):
    x, y = grid_pos
    return [(x + 0.5) * CELL_SIZE, (y + 0.5) * CELL_SIZE, 0.0019]

def is_valid_grid_position(grid_pos):
    x, y = grid_pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and (x, y) not in OBSTACLES

def generate_targets(num_targets, existing_taken_positions):
    generated_targets = []
    taken_positions = set(existing_taken_positions)
    while len(generated_targets) < num_targets:
        tx, ty = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if (tx, ty) not in taken_positions and is_valid_grid_position((tx, ty)):
            generated_targets.append((tx, ty))
            taken_positions.add((tx, ty))
    return generated_targets

def spawn_boxes(target_positions, box_names):
    for i, target in enumerate(target_positions):
        if i < len(box_names):
            box_node = supervisor.getFromDef(box_names[i])
            if box_node:
                box_translation = box_node.getField('translation')
                x, y = target
                box_translation.setSFVec3f([((x + 0.5) * CELL_SIZE) + 0.06, ((y + 0.5) * CELL_SIZE) + 0.06, 0.025])

def hide_box(box_name):
    box_node = supervisor.getFromDef(box_name)
    if box_node:
        box_translation = box_node.getField('translation')
        box_translation.setSFVec3f(HIDE_POSITION)

def get_unassigned_targets(agent_id, allocation, all_targets, collected_status):
    unassigned = []
    num_current_targets = len(all_targets)
    current_allocation = list(allocation) + [-1] * max(0, num_current_targets - len(allocation))

    for target_id in range(num_current_targets):
        if not collected_status[target_id] and (target_id >= len(current_allocation) or current_allocation[target_id] != agent_id):
             unassigned.append(all_targets[target_id])
    return unassigned

def estimate_travel_time(start_pos, end_pos):
    return abs(start_pos[0] - end_pos[0]) + abs(start_pos[1] - end_pos[1])

def is_target_feasible(target_id, agents, all_targets, all_target_remaining_times, all_target_collection_times, timestep, min_steps_per_grid_move):
    if target_id >= len(all_targets) or target_id >= len(all_target_remaining_times) or target_id >= len(all_target_collection_times):
        return False

    target_pos = all_targets[target_id]
    target_remaining_time_seconds = all_target_remaining_times[target_id]
    target_collection_time_steps = all_target_collection_times[target_id]

    # A target is not feasible if its remaining time is already zero or less
    if target_remaining_time_seconds <= 0:
        return False

    target_remaining_time_steps = target_remaining_time_seconds * 1000.0 / timestep

    for agent in agents:
        # If an agent is already collecting this target, it's feasible for that agent
        if agent.is_collecting and agent.currently_collecting == target_id:
             return True

        # Otherwise, check if it's feasible to reach and collect before expiry
        travel_time_grid_steps = estimate_travel_time(agent.grid_pos, target_pos)
        travel_time_simulation_steps = travel_time_grid_steps * min_steps_per_grid_move

        estimated_total_time_steps = travel_time_simulation_steps + target_collection_time_steps

        if estimated_total_time_steps <= target_remaining_time_steps:
            return True

    return False

def get_nearest_neighbor_order(start_pos, target_ids, all_targets, all_target_remaining_times, all_target_collection_times, timestep, min_steps_per_grid_move):
    if not target_ids:
        return [], 0

    current_pos = start_pos
    remaining_target_ids = list(target_ids)
    ordered_target_ids = []
    estimated_total_time_steps = 0

    while remaining_target_ids:
        next_target_id = None
        min_priority_metric = (float('inf'), float('inf'))


        for target_id in remaining_target_ids:
            if target_id >= len(all_targets) or target_id >= len(all_target_remaining_times) or target_id >= len(all_target_collection_times):
                continue

            target_pos = all_targets[target_id]
            target_remaining_time_seconds = all_target_remaining_times[target_id]
            target_collection_time_steps = all_target_collection_times[target_id]

            # If the target has already expired, skip it
            if target_remaining_time_seconds <= 0:
                 continue

            travel_time_grid_steps = estimate_travel_time(current_pos, target_pos)
            travel_time_simulation_steps = travel_time_grid_steps * min_steps_per_grid_move

            estimated_completion_time_if_next_steps = estimated_total_time_steps + travel_time_simulation_steps + target_collection_time_steps

            target_remaining_time_steps = target_remaining_time_seconds * 1000.0 / timestep if target_remaining_time_seconds != float('inf') else float('inf')


            current_priority_metric = (target_remaining_time_seconds, estimated_completion_time_if_next_steps)

            if current_priority_metric < min_priority_metric:
                min_priority_metric = current_priority_metric
                next_target_id = target_id


        if next_target_id is not None:
            travel_time_grid_steps = estimate_travel_time(current_pos, all_targets[next_target_id])
            travel_time_simulation_steps = travel_time_grid_steps * min_steps_per_grid_move

            estimated_total_time_steps += travel_time_simulation_steps

            collection_time_steps = all_target_collection_times[next_target_id] if next_target_id < len(all_target_collection_times) else 0
            estimated_total_time_steps += collection_time_steps

            current_pos = all_targets[next_target_id]

            ordered_target_ids.append(next_target_id)
            remaining_target_ids.remove(next_target_id)
        else:
             break


    return ordered_target_ids, estimated_total_time_steps

# Moved Agent class definition here
class Agent:
    # Removed num_phases from initialization
    def __init__(self, name, agent_id):
        self.name = name
        self.id = agent_id
        self.node = supervisor.getFromDef(name)
        self.translation = self.node.getField('translation')
        self.rotation = self.node.getField('rotation')
        self.grid_pos = AGENT_START_POSITIONS[self.id]
        self.assigned_targets = []
        self.collected_targets = []
        # Initialized a single Q-table for each agent
        self.Q_table = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE, len(ACTIONS)))


        self.position_history = []
        self.stuck_counter = 0
        self.is_stuck = False
        self.max_history = 10

        self.all_assigned_targets_collected = False
        self.currently_collecting = None
        self.total_collection_time = 0
        self.is_collecting = False
        self.steps_since_last_move = 0


    def set_webots_position(self, grid_pos):
        self.grid_pos = grid_pos
        self.translation.setSFVec3f(get_coords_from_grid_position(grid_pos))
        self.rotation.setSFRotation(self.rotation.getSFRotation())

    def update_stuck_status(self):
        if self.all_assigned_targets_collected or self.is_collecting:
             self.is_stuck = False
             self.position_history = []
             self.stuck_counter = 0
             return False

        self.position_history.append(self.grid_pos)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)

        self.is_stuck = False
        if len(self.position_history) >= self.max_history:
            unique_positions = set(tuple(pos) for pos in self.position_history)
            if len(unique_positions) <= 2:
                 self.stuck_counter += 1
            else:
                 self.stuck_counter = 0

        if self.stuck_counter >= 3:
             self.is_stuck = True

        return self.is_stuck

    def reset_stuck_status(self):
        self.stuck_counter = 0
        self.is_stuck = False
        self.position_history = []

    def choose_action(self, current_epsilon_value, assigned_target_pos):
        if self.is_collecting:
            return -1

        if assigned_target_pos is None:
             return random.randint(0, len(ACTIONS) - 1)

        current_state = (self.grid_pos[0], self.grid_pos[1], assigned_target_pos[0], assigned_target_pos[1])

        # Use the agent's single Q-table
        current_Q = self.Q_table

        # Epsilon-greedy action selection
        if random.random() < current_epsilon_value:
            return random.randint(0, len(ACTIONS) - 1)
        else:
            q_values = current_Q[current_state]
            q_values = q_values + np.random.uniform(-0.001, 0.001, size=len(ACTIONS))
            return int(np.argmax(q_values))

    def get_next_position(self, action):
        if action == -1: return self.grid_pos
        dx, dy = ACTIONS[action]
        nx, ny = self.grid_pos[0] + dx, self.grid_pos[1] + dy
        return (nx, ny)

    def is_move_valid(self, next_pos, occupied_positions):
        if not (0 <= next_pos[0] < GRID_SIZE and 0 <= next_pos[1] < GRID_SIZE):
            return False
        if next_pos in OBSTACLES:
            return False
        if next_pos in occupied_positions:
            return False
        return True

    def move(self, action, occupied_positions, simulate_move_only=False):
        if self.all_assigned_targets_collected or self.is_collecting or action == -1:
            self.steps_since_last_move = 0
            return False

        if self.steps_since_last_move < MIN_STEPS_PER_GRID_MOVE:
             self.steps_since_last_move += 1
             return False

        next_pos = self.get_next_position(action)

        if self.is_move_valid(next_pos, occupied_positions):
            self.grid_pos = next_pos
            if not simulate_move_only:
                self.set_webots_position(self.grid_pos)
            self.steps_since_last_move = 0
            return True
        else:
            self.steps_since_last_move = 0
            return False

    def update_q_value(self, old_pos, action, reward, new_pos, assigned_target_pos):
        if action == -1: return
        if assigned_target_pos is None: return

        old_state = (old_pos[0], old_pos[1], assigned_target_pos[0], assigned_target_pos[1])
        new_state = (new_pos[0], new_pos[1], assigned_target_pos[0], assigned_target_pos[1])

        if not (0 <= action < len(ACTIONS)): return

        # Update the agent's single Q-table
        current_Q = self.Q_table

        max_next_q = np.max(current_Q[new_state])
        current_q = current_Q[old_state][action]

        current_Q[old_state][action] = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_next_q)

    def get_uncollected_targets(self, all_targets, collected_status):
        uncollected = []
        for tid in self.assigned_targets:
            if tid < len(all_targets) and not collected_status[tid]:
                 uncollected.append(all_targets[tid])
        return uncollected

    def get_next_target_pos(self, all_targets, collected_status):
        for tid in self.assigned_targets:
            if tid < len(all_targets) and not collected[tid]:
                # Check if the target has expired, unless the agent is currently collecting it
                if tid < len(target_remaining_times) and target_remaining_times[tid] <= 0 and not (self.is_collecting and self.currently_collecting == tid):
                     continue
                return all_targets[tid]
        return None

    def update_assigned_targets_collected_status(self, collected_status):
        self.all_assigned_targets_collected = True
        if not self.assigned_targets:
            return True
        for target_id in self.assigned_targets:
            if target_id < len(collected_status) and not collected_status[target_id]:
                 # Check if the target has expired, unless the agent is currently collecting it
                 if target_id < len(target_remaining_times) and target_remaining_times[target_id] > 0 or (self.is_collecting and self.currently_collecting == target_id):
                      self.all_assigned_targets_collected = False
                      return False
                 elif target_id < len(target_remaining_times) and target_remaining_times[target_id] <= 0 and not (self.is_collecting and self.currently_collecting == target_id):
                     pass # Expired targets are considered "processed" in terms of needing assignment
                 else:
                      self.all_assigned_targets_collected = False
                      return False

        return True


    def start_collecting(self, target_id):
        if self.currently_collecting is None and target_id is not None:
             self.is_collecting = True
             self.currently_collecting = target_id
             # Cancel the expiry timer for this target once collection starts
             if target_id < len(target_remaining_times):
                 target_remaining_times[target_id] = float('inf')
                 print(f"DEBUG: Agent {self.id} started collecting target {target_id}. Expiry timer cancelled.")


    def update_collection_progress(self):
        if not self.is_collecting or self.currently_collecting is None:
            return False, None

        target_id = self.currently_collecting
        if target_id >= len(target_current_collection_times):
             print(f"Error: Agent {self.id} collecting invalid target ID {target_id}")
             self.is_collecting = False
             self.currently_collecting = None
             return False, None

        target_current_collection_times[target_id] += 1

        if target_current_collection_times[target_id] >= target_collection_times[target_id]:
            completed_target_id = self.currently_collecting
            self.is_collecting = False
            self.currently_collecting = None

            # --- MODIFICATION START ---
            # Remove collected target from assigned_targets
            if completed_target_id is not None and completed_target_id in self.assigned_targets:
                self.assigned_targets.remove(completed_target_id)
                print(f"DEBUG: Agent {self.id} removed collected target {completed_target_id} from assigned list.")
            # --- MODIFICATION END ---

            return True, completed_target_id
        return False, None
def compute_pretraining_reward(agent, old_pos, moved, potential_next_pos, target_pos_for_episode, simulated_occupied_positions):
    reward = 0

    if not moved and (not is_valid_grid_position(potential_next_pos) or potential_next_pos in simulated_occupied_positions):
        reward = -1.0
    elif not moved:
        pass

    if moved and target_pos_for_episode and agent.grid_pos == target_pos_for_episode:
        reward = 10.0

    if moved and target_pos_for_episode:
        old_dist = np.linalg.norm(np.array(old_pos) - np.array(target_pos_for_episode))
        new_dist = np.linalg.norm(np.array(agent.grid_pos) - np.array(target_pos_for_episode))

        if new_dist < old_dist:
            reward += 0.1
        elif new_dist >= old_dist:
            reward -= 0.1

    return reward

def compute_mission_reward(agent, old_pos, moved, next_assigned_target_pos, unassigned_targets_pos, action, is_collecting_completed):
    reward = 0

    if is_collecting_completed:
        reward += 20.0  # Big reward for collection

    if agent.is_collecting and action != -1:
        reward -= 5.0  # Punish moving while collecting

    if not moved and action != -1:
        reward -= 1.0  # Small punishment for being stuck or invalid move

    if moved and next_assigned_target_pos:
        old_dist = np.linalg.norm(np.array(old_pos) - np.array(next_assigned_target_pos))
        new_dist = np.linalg.norm(np.array(agent.grid_pos) - np.array(next_assigned_target_pos))

        if new_dist < old_dist:
            reward += 1.0  # Reward for moving closer
        elif new_dist > old_dist:
            reward -= 2.0  # Heavy penalty for moving farther

        if agent.grid_pos == next_assigned_target_pos:
            reward += 20.0  # Big reward if reaches the target

    return reward

# Moved PSO class definition here
class PSO:
    def __init__(self, num_agents, num_targets_initial, num_particles, agents, all_targets, collected_status, all_target_remaining_times, all_target_collection_times, current_simulation_time, timestep, min_steps_per_grid_move):
        self.num_agents = num_agents
        self.num_particles = num_particles
        self.agents = agents
        self.all_targets = all_targets
        self.collected_status = collected_status
        self.all_target_remaining_times = all_target_remaining_times
        self.all_target_collection_times = all_target_collection_times
        self.current_simulation_time = current_simulation_time
        self.timestep = timestep
        self.min_steps_per_grid_move = min_steps_per_grid_move

        self.feasible_target_indices = [
            i for i in range(len(self.all_targets))
            if not self.collected_status[i] and
               (i >= len(self.all_target_remaining_times) or all_target_remaining_times[i] > 0) and # Check if remaining time > 0
               is_target_feasible(i, self.agents, self.all_targets, self.all_target_remaining_times, self.all_target_collection_times, self.timestep, self.min_steps_per_grid_move)
        ]
        self.num_feasible_targets = len(self.feasible_target_indices)

        print(f"DEBUG: PSO considering {self.num_feasible_targets} feasible targets out of {len(self.all_targets)} total uncollected targets.")


        self.particles = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_fitness = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')

        self.c1 = C1
        self.c2 = C2
        self.inertia_weight = INERTIA_WEIGHT_START

        if self.num_feasible_targets > 0:
             self.initialize_particles(self.num_feasible_targets)
        else:
             print("DEBUG: No feasible targets to initialize PSO.")
             self.global_best_position = np.full(len(self.all_targets), -1)
             self.global_best_fitness = self.fitness(np.full(self.num_feasible_targets, -1), self.all_targets, self.collected_status, self.all_target_remaining_times, self.all_target_collection_times, self.current_simulation_time)


    def initialize_particles(self, num_feasible_targets):
         self.particles = [self.random_allocation(num_feasible_targets) for _ in range(self.num_particles)]
         self.velocities = [np.zeros(num_feasible_targets) for _ in range(self.num_particles)]
         self.personal_best_positions = [p.copy() for p in self.particles]
         self.personal_best_fitness = [self.fitness(p, self.all_targets, self.collected_status, self.all_target_remaining_times, self.all_target_collection_times, self.current_simulation_time) for p in self.particles]

         best_idx = np.argmin(self.personal_best_fitness)
         self.global_best_position = self.personal_best_positions[best_idx].copy()
         self.global_best_fitness = self.personal_best_fitness[best_idx]
         print(f"DEBUG: Initial Global Best Fitness (Feasible Targets): {self.global_best_fitness:.2f}")

         full_global_best = np.full(len(self.all_targets), -1)
         for i, feasible_idx in enumerate(self.feasible_target_indices):
              full_global_best[feasible_idx] = self.global_best_position[i]
         self.global_best_position = full_global_best


    def random_allocation(self, num_targets_to_allocate):
        allocation = []
        for i in range(num_targets_to_allocate):
             allocation.append(random.randint(0, self.num_agents - 1))
        return allocation

    def fitness(self, particle, all_targets, collected_status, all_target_remaining_times, all_target_collection_times, current_simulation_time):
        fitness_value = 0
        agent_workload = [0] * self.num_agents
        agent_distances = [0] * self.num_agents


        total_uncollected_count = sum(1 for i in range(len(all_targets)) if not collected_status[i] and (i >= len(all_target_remaining_times) or all_target_remaining_times[i] > 0))

        time_penalty = 0
        targets_collected_in_this_allocation = 0

        agent_assigned_feasible_target_ids = [[] for _ in range(self.num_agents)]
        if len(particle) == self.num_feasible_targets:
            for i, agent_id in enumerate(particle):
                 original_target_id = self.feasible_target_indices[i]
                 if agent_id is not None and 0 <= agent_id < self.num_agents:
                      agent_assigned_feasible_target_ids[agent_id].append(original_target_id)
        elif len(particle) == len(all_targets) and np.all(particle == -1):
             pass
        else:
             print(f"Error: Particle size mismatch in fitness function. Expected {self.num_feasible_targets} or {len(all_targets)}, got {len(particle)}")
             if self.num_feasible_targets > 0:
                  random_fallback_allocation = self.random_allocation(self.num_feasible_targets)
                  for i, agent_id in enumerate(random_fallback_allocation):
                       original_target_id = self.feasible_target_indices[i]
                       if agent_id is not None and 0 <= agent_id < self.num_agents:
                            agent_assigned_feasible_target_ids[agent_id].append(original_target_id)
             else:
                  return float('inf')


        agent_projected_completion_times_steps = [0] * self.num_agents
        for agent_id in range(self.num_agents):
             assigned_target_ids_for_agent = agent_assigned_feasible_target_ids[agent_id]

             ordered_target_ids, estimated_time_steps = get_nearest_neighbor_order(
                 self.agents[agent_id].grid_pos,
                 assigned_target_ids_for_agent,
                 all_targets,
                 all_target_remaining_times,
                 all_target_collection_times,
                 self.timestep,
                 self.min_steps_per_grid_move
             )

             agent_projected_completion_times_steps[agent_id] = estimated_time_steps

             time_elapsed_for_agent = 0
             current_pos_for_penalty = self.agents[agent_id].grid_pos

             for target_id in ordered_target_ids:
                  target_remaining_time = all_target_remaining_times[target_id] if target_id < len(all_target_remaining_times) else float('inf')
                  target_collection_time = all_target_collection_times[target_id] if target_id < len(all_target_collection_times) else 0

                  travel_time_grid_steps = estimate_travel_time(current_pos_for_penalty, all_targets[target_id])
                  travel_time_simulation_steps = travel_time_grid_steps * self.min_steps_per_grid_move

                  estimated_task_completion_time_steps = time_elapsed_for_agent + travel_time_simulation_steps + target_collection_time

                  target_remaining_time_steps = target_remaining_time * 1000.0 / self.timestep if target_remaining_time != float('inf') else float('inf')


                  if estimated_task_completion_time_steps > target_remaining_time_steps and not (self.agents[agent_id].is_collecting and self.agents[agent_id].currently_collecting == target_id):
                       # Only penalize if the target expires AND the agent is NOT currently collecting it
                       time_penalty += 200.0 * (estimated_task_completion_time_steps - target_remaining_time_steps) / (1000.0 / self.timestep)
                  else:
                       targets_collected_in_this_allocation += 1

                  time_elapsed_for_agent += travel_time_simulation_steps + target_collection_time
                  current_pos_for_penalty = all_targets[target_id]


        for agent_id in range(self.num_agents):
             agent_workload[agent_id] = len(agent_assigned_feasible_target_ids[agent_id])

        active_workloads = [w for w in agent_workload if w > 0]
        if active_workloads:
            max_load = max(agent_workload)
            min_load = min(active_workloads) if active_workloads else 0
            workload_difference = max_load - min_load
            workload_variance = np.var(active_workloads)

            fitness_value += workload_difference * 30.0
            fitness_value += workload_variance * 25.0

            idle_agents = sum(1 for load in agent_workload if load == 0)
            working_agents = sum(1 for load in agent_workload if load > 0)
            if idle_agents > 0 and working_agents > 0 and total_uncollected_count > 0:
                fitness_value += idle_agents * 50.0 * (working_agents / self.num_agents)

            if working_agents > 0:
                ideal_workload = total_uncollected_count / self.num_agents
                for agent_load in agent_workload:
                    deviation = abs(agent_load - ideal_workload)
                    fitness_value += deviation * 15.0

        active_completion_times_projected_steps = [t for t in agent_projected_completion_times_steps if t > 0]
        if active_completion_times_projected_steps:
             max_time_steps = max(active_completion_times_projected_steps)
             min_time_steps = min(active_completion_times_projected_steps) if active_completion_times_projected_steps else 0
             time_difference_steps = max_time_steps - min_time_steps
             time_variance_steps = np.var(active_completion_times_projected_steps)

             fitness_value += time_difference_steps * 10.0 / (1000.0 / self.timestep)
             fitness_value += time_variance_steps * 8.0 / (1000.0 / self.timestep)**2


             total_collection_time_projected_sum_steps = sum(active_completion_times_projected_steps)
             if self.num_agents > 0:
                 ideal_collection_time_projected_steps = total_collection_time_projected_sum_steps / self.num_agents
                 for agent_time_projected_steps in active_completion_times_projected_steps:
                       time_deviation_steps = abs(agent_time_projected_steps - ideal_collection_time_projected_steps)
                       fitness_value += time_deviation_steps * 5.0 / (1000.0 / self.timestep)

        current_allocation_check = np.full(len(all_targets), -1)
        if len(particle) == self.num_feasible_targets:
             for i, feasible_idx in enumerate(self.feasible_target_indices):
                  current_allocation_check[feasible_idx] = particle[i]
        elif len(particle) == len(all_targets):
             current_allocation_check = particle


        for original_target_id, agent_id in enumerate(current_allocation_check):
             if agent_id is not None and 0 <= agent_id < self.num_agents:
                 agent = self.agents[agent_id]
                 if agent.is_collecting and agent.currently_collecting == original_target_id:
                      fitness_value += 100.0 # Still reward for collecting the assigned target


        targets_missed_in_this_allocation = total_uncollected_count - targets_collected_in_this_allocation
        fitness_value += targets_missed_in_this_allocation * 1000.0


        return fitness_value


    def update_inertia_weight(self, iteration, max_iterations):
        progress = iteration / max_iterations
        self.inertia_weight = INERTIA_WEIGHT_START - progress * (INERTIA_WEIGHT_START - INERTIA_WEIGHT_END)

    def step(self, all_targets, collected_status, all_target_remaining_times, all_target_collection_times, current_simulation_time):
        self.all_targets = all_targets
        self.collected_status = collected_status
        self.all_target_remaining_times = all_target_remaining_times
        self.all_target_collection_times = all_target_collection_times
        self.current_simulation_time = current_simulation_time

        previous_feasible_indices = set(self.feasible_target_indices)
        self.feasible_target_indices = [
            i for i in range(len(self.all_targets))
            if not self.collected_status[i] and
               (i >= len(self.all_target_remaining_times) or all_target_remaining_times[i] > 0) and # Check if remaining time > 0
               is_target_feasible(i, self.agents, self.all_targets, self.all_target_remaining_times, self.all_target_collection_times, self.timestep, self.min_steps_per_grid_move)
        ]
        self.num_feasible_targets = len(self.feasible_target_indices)

        if len(self.feasible_target_indices) != len(previous_feasible_indices):
             print(f"DEBUG: Number of feasible targets changed from {len(previous_feasible_indices)} to {self.num_feasible_targets}. Re-initializing particles.")
             if self.num_feasible_targets > 0:
                  self.initialize_particles(self.num_feasible_targets)
             else:
                  print("DEBUG: No feasible targets left. Global best is all unassigned.")
                  self.particles = []
                  self.velocities = []
                  self.personal_best_positions = []
                  self.personal_best_fitness = []
                  self.global_best_position = np.full(len(self.all_targets), -1)
                  self.global_best_fitness = self.fitness(np.full(self.num_feasible_targets, -1), self.all_targets, self.collected_status, self.all_target_remaining_times, self.all_target_collection_times, self.current_simulation_time)

             return self.global_best_position.copy()


        if self.num_feasible_targets == 0:
             return np.full(len(self.all_targets), -1)

        for i in range(self.num_particles):
            for j in range(self.num_feasible_targets):
                original_target_id = self.feasible_target_indices[j]

                # If the target is collected or expired (and not being collected), it's not feasible for PSO
                if self.collected_status[original_target_id] or (original_target_id < len(self.all_target_remaining_times) and self.all_target_remaining_times[original_target_id] <= 0 and not any(agent.is_collecting and agent.currently_collecting == original_target_id for agent in self.agents)):
                    self.particles[i][j] = -1
                    self.velocities[i][j] = 0
                    if self.personal_best_positions[i][j] != -1 :
                         self.personal_best_positions[i][j] = self.particles[i][j]
                    continue


                inertia = self.inertia_weight * self.velocities[i][j]
                pb_j = self.personal_best_positions[i][j]
                if pb_j == -2: pb_j = self.particles[i][j]
                cognitive = self.c1 * random.random() * (pb_j - self.particles[i][j])

                gb_j = self.global_best_position[original_target_id]
                if gb_j == -1 or gb_j is None: gb_j = self.particles[i][j]
                social = self.c2 * random.random() * (gb_j - self.particles[i][j])


                new_velocity = inertia + cognitive + social

                new_velocity = max(min(new_velocity, MAX_VELOCITY), -MAX_VELOCITY)
                self.velocities[i][j] = new_velocity

                new_pos = self.particles[i][j] + new_velocity
                new_agent = int(round(new_pos))
                new_agent = (new_agent % self.num_agents + self.num_agents) % self.num_agents

                self.particles[i][j] = new_agent

            fitness_val = self.fitness(self.particles[i], self.all_targets, self.collected_status, self.all_target_remaining_times, self.all_target_collection_times, self.current_simulation_time)
            if fitness_val < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness_val
                self.personal_best_positions[i] = self.particles[i].copy()


        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            for i, feasible_idx in enumerate(self.feasible_target_indices):
                 self.global_best_position[feasible_idx] = self.personal_best_positions[best_idx][i]
            print(f"DEBUG: Global Best Fitness Updated: {self.global_best_fitness:.2f}")


        for target_id in range(len(self.all_targets)):
            # If target is collected, expired (and not being collected), or not feasible, unassign it
            if self.collected_status[target_id] or (target_id < len(self.all_target_remaining_times) and self.all_target_remaining_times[target_id] <= 0 and not any(agent.is_collecting and agent.currently_collecting == target_id for agent in self.agents)):
                 self.global_best_position[target_id] = -1
            # If target is feasible and unassigned in global best, assign it randomly (fallback)
            if target_id in self.feasible_target_indices and (self.global_best_position[target_id] == -1 or self.global_best_position[target_id] is None):
                  self.global_best_position[target_id] = random.randint(0, self.num_agents - 1)


        return self.global_best_position.copy()

from controller import Supervisor
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
simulation_time_seconds = supervisor.getTime()


# Generate fixed dynamic spawn intervals at the start
# Changed to a fixed 5-second interval for testing
dynamic_spawn_intervals_seconds = [3] * MAX_ADDITIONAL_TARGETS
dynamic_spawn_times_steps = [interval * 1000 // timestep for interval in dynamic_spawn_intervals_seconds]
cumulative_dynamic_spawn_times_steps = [sum(dynamic_spawn_times_steps[:i+1]) for i in range(MAX_ADDITIONAL_TARGETS)]

print("DEBUG: Pre-determined dynamic spawn intervals (seconds):", dynamic_spawn_intervals_seconds)
print("DEBUG: Cumulative dynamic spawn times (steps):", cumulative_dynamic_spawn_times_steps)


initial_targets_pos = generate_targets(NUM_TARGETS, OBSTACLES + list(AGENT_START_POSITIONS))
targets = list(initial_targets_pos)
collected = [False] * len(targets)
collected_by = [None] * len(targets)
target_collection_times = [random.randint(MIN_COLLECTION_TIME, MAX_COLLECTION_TIME) for _ in range(len(targets))]
# Initial target remaining times remain as they were
target_remaining_times = [14.0, 5.0, 10.0, 7.0, 5.0]
target_current_collection_times = [0] * len(targets)

prev_collected_status = list(collected)
prev_target_remaining_times = list(target_remaining_times)


print("DEBUG: Initial Target Remaining Times:", target_remaining_times)

# Initialize agents - Removed the num_phases parameter
# Moved Agent and PSO class definitions above this line
robots = [Agent(name, i) for i, name in enumerate(AGENT_NAMES)]


# --- Pre-training Phase ---
print("=== STARTING PRE-TRAINING (Episode-based RL Path Planning) ===")
print(f"Running {PRETRAINING_EPISODES} episodes, {PRETRAINING_ITERATIONS_PER_EPISODE} iterations per episode.")


for episode in range(PRETRAINING_EPISODES):
    # Reset agent positions for each episode
    simulated_occupied_positions = []
    for agent in robots:
         #agent.grid_pos = AGENT_START_POSITIONS[agent.id]
         simulated_occupied_positions.append(agent.grid_pos)
         agent.reset_stuck_status()
         agent.steps_since_last_move = 0
         # In pre-training, agents train on initial targets (implicitly phase 0)


    # Reset episode-specific target information for simulation (only initial targets for pre-training)
    episode_targets = list(initial_targets_pos)
    episode_collected = [False] * len(episode_targets)
    episode_target_remaining_times = list(target_remaining_times[:NUM_TARGETS]) # Only initial targets' remaining times
    episode_target_collection_times = list(target_collection_times[:NUM_TARGETS]) # Only initial targets' collection times
    episode_target_current_collection_times = [0] * len(episode_targets)

    # Calculate epsilon for epsilon-greedy strategy with decay
    decay_rate = (EPSILON_PRETRAIN_START - EPSILON_PRETRAIN_END) / (PRETRAINING_EPISODES - 1) if PRETRAINING_EPISODES > 1 else 0
    current_epsilon = max(EPSILON_PRETRAIN_END, EPSILON_PRETRAIN_START - decay_rate * episode)

    if (episode + 1) % 100 == 0 or episode == 0:
         print(f" Episode {episode + 1}/{PRETRAINING_EPISODES}, Epsilon: {current_epsilon:.4f}")

    # Run iterations within the episode
    for iteration in range(PRETRAINING_ITERATIONS_PER_EPISODE):

        # Get simulated occupied positions for collision detection
        simulated_occupied_positions_this_step = [agent.grid_pos for agent in robots]

        for agent_id, agent in enumerate(robots):
            old_pos = agent.grid_pos

            # In pre-training, each agent trains towards a single random initial target
            available_episode_target_indices = [i for i in range(len(episode_targets)) if not episode_collected[i] and (i >= len(episode_target_remaining_times) or episode_target_remaining_times[i] > 0)]

            assigned_target_pos_for_episode = None
            if available_episode_target_indices:
                 # Pick a random target from the currently available initial ones in this episode
                 target_idx_for_episode = random.choice(available_episode_target_indices)
                 assigned_target_pos_for_episode = episode_targets[target_idx_for_episode]


            action = agent.choose_action(current_epsilon, assigned_target_pos_for_episode)
            potential_next_pos = agent.get_next_position(action)

            simulated_occupied_but_self = [pos for i, pos in enumerate(simulated_occupied_positions_this_step) if i != agent_id]
            move_is_valid_simulated = agent.is_move_valid(potential_next_pos, simulated_occupied_but_self)

            if agent.steps_since_last_move < MIN_STEPS_PER_GRID_MOVE:
                 agent.steps_since_last_move += 1
                 moved = False
            else:
                 if move_is_valid_simulated:
                      agent.grid_pos = potential_next_pos
                      moved = True
                 else:
                      moved = False

                 agent.steps_since_last_move = 0


            # Compute reward for the simulated step
            reward = compute_pretraining_reward(agent, old_pos, moved, potential_next_pos, assigned_target_pos_for_episode, simulated_occupied_but_self)

            # Update Q-value using the agent's single Q-table
            if action != -1 and assigned_target_pos_for_episode is not None:
                 if agent.steps_since_last_move == 0 or MIN_STEPS_PER_GRID_MOVE == 1:
                     agent.update_q_value(old_pos, action, reward, agent.grid_pos, assigned_target_pos_for_episode)


            # Simulate target collection if agent reaches the target in pre-training simulation
            if assigned_target_pos_for_episode is not None and agent.grid_pos == assigned_target_pos_for_episode:
                 # Find the index of the target the agent reached in the episode's targets
                 reached_target_indices = [i for i, pos in enumerate(episode_targets) if pos == agent.grid_pos and not episode_collected[i] and (i >= len(episode_target_remaining_times) or episode_target_remaining_times[i] > 0)]
                 if reached_target_indices:
                      reached_target_idx = reached_target_indices[0]
                      episode_collected[reached_target_idx] = True # Mark as collected in simulation
                      # Cancel the expiry timer for this target in pre-training simulation
                      if reached_target_idx < len(episode_target_remaining_times):
                           episode_target_remaining_times[reached_target_idx] = float('inf')


        # Check if all initial targets are collected in this pre-training episode simulation
        all_initial_targets_collected_in_episode = all(episode_collected)
        if all_initial_targets_collected_in_episode:
             break # End episode early if all initial targets are collected


print("=== PRE-TRAINING COMPLETE ===")

# --- Section for Q-table visualization (Removed print statements) ---


print("\n=== STARTING MAIN SIMULATION ===")

# Reset agent positions and status for the main simulation
for i in range(NUM_AGENTS):
    robots[i].set_webots_position(AGENT_START_POSITIONS[i])
    robots[i].reset_stuck_status()
    robots[i].steps_since_last_move = 0
    # Agents start in the implicit phase of having only initial targets


# Spawn the initial target boxes in the Webots world
spawn_boxes(initial_targets_pos, TARGET_BOXES_INITIAL)

# --- Initial PSO Task Allocation (Runs only once at the start) ---
print("=== INITIALIZING AND RUNNING PSO FOR TASK ALLOCATION ===")
pso = PSO(NUM_AGENTS, NUM_TARGETS, NUM_PARTICLES, robots, targets, collected, target_remaining_times, target_collection_times, supervisor.getTime(), timestep, MIN_STEPS_PER_GRID_MOVE)
task_allocation = []
if pso.num_feasible_targets > 0:
    for iteration in range(PSO_ITERATIONS_INITIAL):
        pso.update_inertia_weight(iteration, PSO_ITERATIONS_INITIAL)
        task_allocation = pso.step(targets, collected, target_remaining_times, target_collection_times, supervisor.getTime())
else:
    print("DEBUG: No feasible targets initially. Task allocation is all unassigned.")
    task_allocation = np.full(len(targets), -1)


print("=== ASSIGNING INITIAL TASKS BASED ON PSO ALLOCATION ===")
for agent_id, agent in enumerate(robots):
    # Store the currently collecting target if any
    currently_collecting_target_id = agent.currently_collecting if agent.is_collecting and agent.currently_collecting is not None else None

    agent.assigned_targets.clear()
    agent.all_assigned_targets_collected = False

    # Get the list of other feasible targets assigned by the new PSO allocation
    other_assigned_feasible_target_ids = []
    num_current_targets = len(targets)
    current_task_allocation = list(task_allocation)
    for tid in range(num_current_targets):
        # Only consider targets that are not collected and not expired (unless the agent is currently collecting it)
        if current_task_allocation[tid] == agent_id and not collected[tid] and (tid >= len(target_remaining_times) or target_remaining_times[tid] > 0 or (agent.is_collecting and agent.currently_collecting == tid)):
            if tid != currently_collecting_target_id: # Exclude the target being collected (it's handled separately)
                 other_assigned_feasible_target_ids.append(tid)

    # Order the other assigned targets
    ordered_other_assigned_target_ids, _ = get_nearest_neighbor_order(
        agent.grid_pos,
        other_assigned_feasible_target_ids,
        targets,
        target_remaining_times,
        target_collection_times,
        timestep,
        MIN_STEPS_PER_GRID_MOVE
    )

    # Construct the final assigned targets list, prioritizing the currently collecting target
    final_assigned_targets = []
    # Add the currently collecting target first if it's still valid (not collected)
    if currently_collecting_target_id is not None and currently_collecting_target_id < len(targets) and not collected[currently_collecting_target_id]:
        final_assigned_targets.append(currently_collecting_target_id)

    final_assigned_targets.extend(ordered_other_assigned_target_ids)
    agent.assigned_targets = final_assigned_targets


print("\n=== INITIAL TARGET ASSIGNMENTS (Ordered by Time-Prioritized Nearest Neighbor, Prioritizing Current Collection) ===")
for agent_id, agent in enumerate(robots):
    assigned_target_indices = list(agent.assigned_targets)
    assigned_target_info = []
    for tid in assigned_target_indices:
        if tid < len(targets) and tid < len(target_collection_times) and tid < len(target_remaining_times):
             assigned_target_info.append((tid, targets[tid], f"CollectTime:{target_collection_times[tid]}", f"RemainingTime:{target_remaining_times[tid]:.2f}"))
        elif tid < len(targets):
             assigned_target_info.append((tid, targets[tid], "CollectTime:N/A", "RemainingTime:N/A"))
        else:
             assigned_target_info.append((tid, "N/A", "CollectTime:N/A", "RemainingTime:N/A"))


    print(f"DEBUG:   Agent {agent.id} assigned targets: {assigned_target_info})")


current_epsilon = EPSILON_EXECUTION
iteration_count = 0
final_total_mission_reward = 0

stuck_detection_interval = 20

initial_targets_collected_printed = False
dynamic_targets_collected_printed = False
all_targets_spawned = False
total_targets_collected_count = 0

# Use the pre-determined spawn times for execution
next_dynamic_spawn_time_index = 0 # Index for the next dynamic target to spawn
next_dynamic_spawn_time_step = cumulative_dynamic_spawn_times_steps[0] if cumulative_dynamic_spawn_times_steps else float('inf')


print("\n=== STARTING MAIN SIMULATION LOOP (Execution) ===")

prev_collected_status = list(collected)
prev_target_remaining_times = list(target_remaining_times)


while supervisor.step(timestep) != -1:
    current_step_mission_reward = 0
    simulation_time_seconds = supervisor.getTime()

    # Decrement remaining times only for targets that are NOT collected and NOT currently being collected
    targets_to_hide_on_expiry = []
    for i in range(len(targets)):
        if not collected[i] and not (any(agent.is_collecting and agent.currently_collecting == i for agent in robots)):
             if i < len(target_remaining_times):
                # Check if the target was not expired in the previous step and is now expired
                was_not_expired_prev = i < len(prev_target_remaining_times) and prev_target_remaining_times[i] > 0

                # Decrement the remaining time
                target_remaining_times[i] -= (timestep / 1000.0)

                # Check if it is now expired and the message hasn't been printed yet
                if target_remaining_times[i] <= 0 and not collected[i] and i not in expired_target_messages_printed:
                    target_remaining_times[i] = 0
                    print(f"Target {i} at {targets[i]} EXPIRED at simulation time {simulation_time_seconds:.2f} seconds.")
                    targets_to_hide_on_expiry.append(i)
                    expired_target_messages_printed.add(i) # Mark message as printed


    for expired_target_id in targets_to_hide_on_expiry:
         box_name_to_hide = None
         if expired_target_id < NUM_TARGETS:
             box_name_to_hide = TARGET_BOXES_INITIAL[expired_target_id]
         elif expired_target_id >= NUM_TARGETS and (expired_target_id - NUM_TARGETS) < MAX_ADDITIONAL_TARGETS:
              try:
                   spawn_order_index = expired_target_id - NUM_TARGETS
                   if spawn_order_index < len(ADDITIONAL_BOXES):
                       box_name_to_hide = ADDITIONAL_BOXES[spawn_order_index]
                   else:
                       print(f"Warning: Could not find box name for dynamic target ID {expired_target_id}. Spawn order index {spawn_order_index} out of bounds for ADDITIONAL_BOXES.")
              except Exception as e:
                  print(f"Error mapping dynamic target {expired_target_id} to box name: {e}")
                  box_name_to_hide = None

         if box_name_to_hide:
             hide_box(box_name_to_hide)


    # --- Dynamic Target Spawning (using pre-determined times) ---
    if next_dynamic_spawn_time_index < MAX_ADDITIONAL_TARGETS:
         if iteration_count >= next_dynamic_spawn_time_step:
              dynamic_target_index_to_spawn = next_dynamic_spawn_time_index
              print(f"\n=== Spawning Dynamic Target {dynamic_target_index_to_spawn + NUM_TARGETS} at simulation step {iteration_count} (time {simulation_time_seconds:.2f}s) ===")

              all_taken_positions = set(OBSTACLES)
              all_taken_positions.update(targets)
              all_taken_positions.update([r.grid_pos for r in robots])

              new_target_pos = None
              max_spawn_attempts = 100
              attempt = 0
              while attempt < max_spawn_attempts:
                  tx, ty = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                  potential_pos = (tx, ty)
                  if potential_pos not in all_taken_positions and is_valid_grid_position(potential_pos):
                      new_target_pos = potential_pos
                      break
                  attempt += 1

              if new_target_pos is None:
                  print(f"Warning: Could not find a valid spawn position for dynamic target after {max_spawn_attempts} attempts.")
                  dynamic_targets_spawned_count += 1
                  next_dynamic_spawn_time_index += 1
                  if next_dynamic_spawn_time_index < MAX_ADDITIONAL_TARGETS:
                       next_dynamic_spawn_time_step = cumulative_dynamic_spawn_times_steps[next_dynamic_spawn_time_index]
                       print(f"DEBUG: Next dynamic target will spawn around simulation time {next_dynamic_spawn_time_step * (timestep / 1000.0):.2f} seconds.")
                  else:
                       next_dynamic_spawn_time_step = float('inf') # No more dynamic targets
                  continue

              new_target_global_index = len(targets)

              targets.append(new_target_pos)
              collected.append(False)
              collected_by.append(None)
              collection_time = random.randint(MIN_COLLECTION_TIME, MAX_COLLECTION_TIME)
              target_collection_times.append(collection_time)
              # Assign the fixed remaining time for the dynamic target in execution
              if dynamic_target_index_to_spawn < len(DYNAMIC_TARGET_EXPIRY_TIMES_SECONDS):
                  target_remaining_times.append(DYNAMIC_TARGET_EXPIRY_TIMES_SECONDS[dynamic_target_index_to_spawn])
              else:
                   # Fallback to a default or random if the list is shorter than expected
                   target_remaining_times.append(10.0) # Default expiry time
                   print(f"Warning: DYNAMIC_TARGET_EXPIRY_TIMES_SECONDS list is shorter than MAX_ADDITIONAL_TARGETS. Using default expiry time for dynamic target {dynamic_target_index_to_spawn}.")


              target_current_collection_times.append(0)

              prev_collected_status = list(collected)
              prev_target_remaining_times = list(target_remaining_times)


              if dynamic_target_index_to_spawn < len(ADDITIONAL_BOXES):
                   box_name_to_spawn = ADDITIONAL_BOXES[dynamic_target_index_to_spawn]
                   spawn_boxes([new_target_pos], [box_name_to_spawn])
                   print(f"New target {box_name_to_spawn} spawned at {new_target_pos} (Target ID: {new_target_global_index}, Collection Time: {collection_time}, Remaining Time: {target_remaining_times[new_target_global_index]:.2f})")
              else:
                  print(f"Warning: Not enough dynamic box names configured for target ID {new_target_global_index}")


              dynamic_targets_spawned_count += 1

              # --- Trigger PSO Reallocation due to New Dynamic Target ---
              print(" Reallocating tasks due to new dynamic target...")
              pso = PSO(NUM_AGENTS, len(targets), NUM_PARTICLES, robots, targets, collected, target_remaining_times, target_collection_times, supervisor.getTime(), timestep, MIN_STEPS_PER_GRID_MOVE)

              task_allocation = [0] * len(targets)
              if pso.num_feasible_targets > 0:
                  for pso_iter in range(PSO_ITERATIONS_REALLOC):
                       pso.update_inertia_weight(pso_iter, PSO_ITERATIONS_REALLOC)
                       task_allocation = pso.step(targets, collected, target_remaining_times, target_collection_times, supervisor.getTime())
              else:
                   print("DEBUG: No feasible targets after spawning dynamic target. Task allocation is all unassigned.")
                   task_allocation = np.full(len(targets), -1)


              # --- Assigning Reallocated Tasks (Prioritizing Current Collection) ---
              for agent_id, agent in enumerate(robots):
                  # Store the currently collecting target if any
                  currently_collecting_target_id = agent.currently_collecting if agent.is_collecting and agent.currently_collecting is not None else None

                  agent.assigned_targets.clear()
                  agent.all_assigned_targets_collected = False

                  # Get the list of other feasible targets assigned by the new PSO allocation
                  other_assigned_feasible_target_ids = []
                  num_current_targets = len(targets)
                  current_task_allocation = list(task_allocation)
                  for tid in range(num_current_targets):
                      # Only consider targets that are not collected and not expired (unless the agent is currently collecting it)
                      if current_task_allocation[tid] == agent_id and not collected[tid] and (tid >= len(target_remaining_times) or target_remaining_times[tid] > 0 or (agent.is_collecting and agent.currently_collecting == tid)):
                          if tid != currently_collecting_target_id: # Exclude the target being collected (it's handled separately)
                              other_assigned_feasible_target_ids.append(tid)

                  # Order the other assigned targets
                  ordered_other_assigned_target_ids, _ = get_nearest_neighbor_order(
                      agent.grid_pos,
                      other_assigned_feasible_target_ids,
                      targets,
                      target_remaining_times,
                      target_collection_times,
                      timestep,
                      MIN_STEPS_PER_GRID_MOVE
                  )

                  # Construct the final assigned targets list, prioritizing the currently collecting target
                  final_assigned_targets = []
                  # Add the currently collecting target first if it's still valid (not collected)
                  if currently_collecting_target_id is not None and currently_collecting_target_id < len(targets) and not collected[currently_collecting_target_id]:
                      final_assigned_targets.append(currently_collecting_target_id)

                  final_assigned_targets.extend(ordered_other_assigned_target_ids)
                  agent.assigned_targets = final_assigned_targets


              print("DEBUG: Task reassignment completed.")

              print(f"DEBUG: Assignments after reallocation at iteration {iteration_count} (Ordered by Time-Prioritized Nearest Neighbor):")
              for agent_id, agent in enumerate(robots):
                   assigned_target_indices = list(agent.assigned_targets)
                   assigned_target_info = []
                   for tid in assigned_target_indices:
                       if tid < len(targets) and tid < len(target_collection_times) and tid < len(target_remaining_times):
                           assigned_target_info.append((tid, targets[tid], f"CollectTime:{target_collection_times[tid]}", f"RemainingTime:{target_remaining_times[tid]:.2f}"))
                       elif tid < len(targets):
                            assigned_target_info.append((tid, targets[tid], "CollectTime:N/A", "RemainingTime:N/A"))
                       else:
                           assigned_target_info.append((tid, "N/A", "CollectTime:N/A", "RemainingTime:N/A"))


                   print(f"DEBUG:   Agent {agent.id} assigned targets: {assigned_target_info})")


              # Schedule the next dynamic target spawn if there are more
              next_dynamic_spawn_time_index += 1
              if next_dynamic_spawn_time_index < MAX_ADDITIONAL_TARGETS:
                  next_dynamic_spawn_time_step = cumulative_dynamic_spawn_times_steps[next_dynamic_spawn_time_index]
                  print(f"DEBUG: Next dynamic target will spawn around simulation time {next_dynamic_spawn_time_step * (timestep / 1000.0):.2f} seconds.")
              else:
                   next_dynamic_spawn_time_step = float('inf') # No more dynamic targets

              agent_positions_before_relearning = [(agent.grid_pos) for agent in robots] # <-- Add this line
              # --- Start New RL Learning Phase ---
              print(f"\n=== STARTING NEW RL LEARNING PHASE (Dynamic Targets) ===")

              for relearn_episode in range(RELEARNING_EPISODES):
                  # Reset agent internal state for the relearning episode simulation
                  simulated_occupied_positions = []
                  for agent in robots:
                       # Agents start relearning from their CURRENT simulation position.
                       # Do NOT reset grid_pos to starting positions.
                       # REMOVE the line that resets agent.grid_pos here if it exists
                       # REMOVE the line agent.set_webots_position(agent.grid_pos) if it exists here
                       simulated_occupied_positions.append(agent.grid_pos) # Use current grid_pos for simulation
                       agent.reset_stuck_status()
                       agent.steps_since_last_move = 0


                  # Simulate the current state of targets for relearning (including the new target)
                  episode_targets_relearn = list(targets)
                  episode_collected_relearn = list(collected)
                  episode_target_remaining_times_relearn = list(target_remaining_times)
                  episode_target_collection_times_relearn = list(target_collection_times)
                  episode_target_current_collection_times_relearn = [0] * len(episode_targets_relearn)


                  # Calculate epsilon for relearning
                  decay_rate_relearn = (EPSILON_RELEARN_START - EPSILON_RELEARN_END) / (RELEARNING_EPISODES - 1) if RELEARNING_EPISODES > 1 else 0
                  current_epsilon_relearn = max(EPSILON_RELEARN_END, EPSILON_RELEARN_START - decay_rate_relearn * relearn_episode)

                  if (relearn_episode + 1) % 100 == 0 or relearn_episode == 0:
                       print(f" Relearning Episode {relearn_episode + 1}/{RELEARNING_EPISODES}, Epsilon: {current_epsilon_relearn:.4f}")


                  # Run iterations within the relearning episode
                  for relearn_iteration in range(RELEARNING_ITERATIONS_PER_EPISODE):
                      simulated_occupied_positions_this_step_relearn = [agent.grid_pos for agent in robots]

                      for agent_id, agent in enumerate(robots):
                          old_pos = agent.grid_pos

                          # In relearning, agents should train towards their *currently assigned* targets from the new PSO allocation
                          assigned_target_pos_for_relearn = None
                          # Find the first uncollected assigned target for this agent in the simulated episode data
                          for tid in agent.assigned_targets:
                              if tid < len(episode_targets_relearn) and not episode_collected_relearn[tid]:
                                   # Check if the target is not expired in the relearning simulation
                                   if tid < len(episode_target_remaining_times_relearn) and episode_target_remaining_times_relearn[tid] <= 0:
                                        continue # Skip expired targets in relearning simulation
                                   assigned_target_pos_for_relearn = episode_targets_relearn[tid]
                                   break # Focus on the first assigned uncollected target


                          action = agent.choose_action(current_epsilon_relearn, assigned_target_pos_for_relearn)
                          potential_next_pos = agent.get_next_position(action)

                          simulated_occupied_but_self = [pos for i, pos in enumerate(simulated_occupied_positions_this_step_relearn) if i != agent_id]
                          move_is_valid_simulated = agent.is_move_valid(potential_next_pos, simulated_occupied_but_self)


                          if agent.steps_since_last_move < MIN_STEPS_PER_GRID_MOVE:
                               agent.steps_since_last_move += 1
                               moved = False
                          else:
                               if move_is_valid_simulated:
                                    agent.grid_pos = potential_next_pos
                                    moved = True
                               else:
                                    moved = False

                               agent.steps_since_last_move = 0


                          # Compute reward for relearning
                          reward = compute_pretraining_reward(agent, old_pos, moved, potential_next_pos, assigned_target_pos_for_relearn, simulated_occupied_but_self)


                          # Update Q-value using the agent's single Q-table
                          if action != -1 and assigned_target_pos_for_relearn is not None:
                               if agent.steps_since_last_move == 0 or MIN_STEPS_PER_GRID_MOVE == 1:
                                   agent.update_q_value(old_pos, action, reward, agent.grid_pos, assigned_target_pos_for_relearn)


                          # Simulate target collection in relearning if agent reaches its assigned target
                          if assigned_target_pos_for_relearn is not None and agent.grid_pos == assigned_target_pos_for_relearn:
                              # Find the index of the target the agent reached *among the currently assigned and uncollected targets*
                              reached_target_indices = [i for i in agent.assigned_targets if i < len(episode_targets_relearn) and not episode_collected_relearn[i] and episode_targets_relearn[i] == agent.grid_pos and (i >= len(episode_target_remaining_times_relearn) or episode_target_remaining_times_relearn[i] > 0)]
                              if reached_target_indices:
                                   reached_target_idx = reached_target_indices[0]
                                   episode_collected_relearn[reached_target_idx] = True # Mark as collected in simulation
                                   # Cancel the expiry timer for this target in relearning simulation
                                   if reached_target_idx < len(episode_target_remaining_times_relearn):
                                        episode_target_remaining_times_relearn[reached_target_idx] = float('inf')


                      # Check if all *assigned* targets for all agents are collected in this relearning episode
                      all_assigned_collected_in_relearn_episode = True
                      for agent in robots:
                           uncollected_assigned = [tid for tid in agent.assigned_targets if tid < len(episode_collected_relearn) and not episode_collected_relearn[tid] and (tid >= len(episode_target_remaining_times_relearn) or episode_target_remaining_times_relearn[tid] > 0)]
                           if uncollected_assigned:
                                all_assigned_collected_in_relearn_episode = False
                                break

                      if all_assigned_collected_in_relearn_episode:
                           break # End episode early if all assigned targets are collected

              print(f"=== NEW RL LEARNING PHASE COMPLETE ===")
              for i, agent in enumerate(robots):
                     agent.grid_pos = agent_positions_before_relearning[i]
                     agent.set_webots_position(agent.grid_pos) # <-- Add this loop and call

              print("DEBUG: Agents positions reset to before relearning and Webots updated.") # Update the debug message

              # --- Set Webots position after relearning ---
              # Ensure Webots position matches the internal grid_pos before resuming main simulation
              


    all_targets_spawned = (dynamic_targets_spawned_count == MAX_ADDITIONAL_TARGETS)


    # Check if all targets that were spawned are either collected or expired (and not being collected)
    all_currently_spawned_targets_processed = all(collected[i] or (i < len(target_remaining_times) and target_remaining_times[i] <= 0 and not any(agent.is_collecting and agent.currently_collecting == i for agent in robots)) for i in range(len(targets)))


    if iteration_count % stuck_detection_interval == 0:
         stuck_agents = []
         for agent in robots:
              if not agent.is_collecting and not agent.all_assigned_targets_collected and agent.update_stuck_status():
                   stuck_agents.append(agent.id)


    current_occupied_positions = [r.grid_pos for r in robots]

    for agent_id, agent in enumerate(robots):
        is_collecting_completed_in_step = False

        if agent.is_collecting:
            completed, collected_target_id = agent.update_collection_progress()
            if completed:
                if collected_target_id is not None and collected_target_id < len(collected):
                    collected[collected_target_id] = True
                    collected_by[collected_target_id] = agent_id
                    agent.collected_targets.append(collected_target_id)
                    agent.all_assigned_targets_collected = agent.update_assigned_targets_collected_status(collected)
                    total_targets_collected_count += 1

                    collection_time_completed = target_collection_times[collected_target_id] if collected_target_id < len(target_collection_times) else "N/A"
                    target_pos_collected = targets[collected_target_id] if collected_target_id < len(targets) else "N/A"

                    print(f"Target {collected_target_id} at {target_pos_collected} COLLECTED by Agent {agent_id} after {collection_time_completed} time steps at simulation time {simulation_time_seconds:.2f} seconds.")

                    box_name_to_hide = None
                    if collected_target_id < NUM_TARGETS:
                        box_name_to_hide = TARGET_BOXES_INITIAL[collected_target_id]
                    elif collected_target_id >= NUM_TARGETS and (collected_target_id - NUM_TARGETS) < MAX_ADDITIONAL_TARGETS:
                         try:
                              spawn_order_index = collected_target_id - NUM_TARGETS
                              if spawn_order_index < len(ADDITIONAL_BOXES):
                                   box_name_to_hide = ADDITIONAL_BOXES[spawn_order_index]
                              else:
                                  print(f"Warning: Could not find box name for dynamic target ID {collected_target_id}. Spawn order index {spawn_order_index} out of bounds for ADDITIONAL_BOXES.")
                         except Exception as e:
                             print(f"Error mapping dynamic target {collected_target_id} to box name: {e}")
                             box_name_to_hide = None

                    if box_name_to_hide:
                        hide_box(box_name_to_hide)

                collection_completion_reward = compute_mission_reward(agent, agent.grid_pos, False, None, None, -1, True)
                current_step_mission_reward += collection_completion_reward
                final_total_mission_reward += collection_completion_reward

            continue

        agent.all_assigned_targets_collected = agent.update_assigned_targets_collected_status(collected)
        if agent.all_assigned_targets_collected:
            continue

        next_assigned_target_id = None
        for tid in agent.assigned_targets:
            if tid < len(targets) and not collected[tid]:
                # Only consider targets that are not expired, unless the agent is currently collecting it
                if tid < len(target_remaining_times) and target_remaining_times[tid] <= 0 and not (agent.is_collecting and agent.currently_collecting == tid):
                     continue
                next_assigned_target_id = tid
                break

        next_assigned_target_pos = targets[next_assigned_target_id] if next_assigned_target_id is not None else None

        if not next_assigned_target_pos:
            agent.all_assigned_targets_collected = agent.update_assigned_targets_collected_status(collected)
            if not agent.all_assigned_targets_collected:
                pass
            continue

        if agent.grid_pos == next_assigned_target_pos:
            assigned_target_index = None
            # Check if the target is valid to start collecting (not collected and not expired, unless already collecting)
            if agent.assigned_targets and not collected[agent.assigned_targets[0]]:
                 if agent.assigned_targets[0] < len(target_remaining_times) and (target_remaining_times[agent.assigned_targets[0]] > 0 or (agent.is_collecting and agent.currently_collecting == agent.assigned_targets[0])):
                     if targets[agent.assigned_targets[0]] == agent.grid_pos:
                          assigned_target_index = agent.assigned_targets[0]
                 elif agent.assigned_targets[0] < len(target_remaining_times) and target_remaining_times[agent.assigned_targets[0]] <= 0 and not (agent.is_collecting and agent.currently_collecting == agent.assigned_targets[0]): # Corrected index here
                     pass # Expired and not being collected, so cannot start collecting
                 else:
                      pass # Handle cases where target_remaining_times might not be long enough


            if assigned_target_index is not None:
                agent.start_collecting(assigned_target_index)
                continue # Skip movement and Q-learning if starting collection

            else:
                 pass

        old_pos = agent.grid_pos

        action = -1
        valid_random_action_found = False
        


        if not agent.is_stuck or valid_random_action_found:
             epsilon_for_step = EPSILON_EXECUTION

             if not valid_random_action_found:
                  action = agent.choose_action(epsilon_for_step, next_assigned_target_pos)


        occupied_positions_for_move = [r.grid_pos for r in robots if r.id != agent_id]
        unassigned_targets_pos = get_unassigned_targets(agent.id, task_allocation, targets, collected)
        occupied_positions_for_move.extend(unassigned_targets_pos)

        moved = agent.move(action, occupied_positions_for_move, simulate_move_only=False)

        reward = compute_mission_reward(agent, old_pos, moved, next_assigned_target_pos, unassigned_targets_pos, action, False)

        current_step_mission_reward += reward
        final_total_mission_reward += reward

        if action != -1:
             if agent.steps_since_last_move == 0 or MIN_STEPS_PER_GRID_MOVE == 1:
                 agent.update_q_value(old_pos, action, reward, agent.grid_pos, next_assigned_target_pos)


        if moved or not agent.is_stuck:
             agent.reset_stuck_status()


    iteration_count += 1

    all_targets_spawned = (dynamic_targets_spawned_count == MAX_ADDITIONAL_TARGETS)


    # Check if all targets that were spawned are either collected or expired (and not being collected)
    all_currently_spawned_targets_processed = all(collected[i] or (i < len(target_remaining_times) and target_remaining_times[i] <= 0 and not any(agent.is_collecting and agent.currently_collecting == i for agent in robots)) for i in range(len(targets)))


    if all_currently_spawned_targets_processed and all_targets_spawned:
         print(f"\n=== Simulation Complete ===")
         print(f"=== Simulation ended at iteration {iteration_count} at simulation time {simulation_time_seconds:.2f} seconds. ===")
         print(f"=== Total Targets Initially: {NUM_TARGETS}. Total Dynamic Targets Spawned: {dynamic_targets_spawned_count}. Total Targets in Simulation: {len(targets)}. ===")
         print(f"=== Total Targets Collected: {total_targets_collected_count}. ===")
         expired_count = sum(1 for i in range(len(targets)) if (i < len(target_remaining_times) and target_remaining_times[i] <= 0) and not collected[i])
         print(f"=== Total Targets Expired: {expired_count}. ===")
         print(f"=== Final Total Mission Reward: {final_total_mission_reward:.2f} ===")

         print("\n=== FINAL TARGET COLLECTION SUMMARY ===")
         agent_actual_collection_times = [0] * NUM_AGENTS
         for agent_id, agent in enumerate(robots):
             collected_target_ids = sorted(agent.collected_targets)
             collected_times_list = [target_collection_times[tid] for tid in collected_target_ids if tid < len(target_collection_times)]
             print(f"Agent {agent.id} collected targets (IDs): {collected_target_ids}")
             actual_time_collecting = sum(target_current_collection_times[tid] for tid in collected_target_ids if tid < len(target_current_collection_times))
             print(f" Agent {agent.id} total *actual time* spent collecting: {actual_time_collecting} simulation steps")
             agent_actual_collection_times[agent_id] = actual_time_collecting

         print("\n=== FINAL COLLECTION TIME FAIRNESS ANALYSIS (Actual Time Spent) ===")
         print(f"Agent actual time spent collecting: {agent_actual_collection_times}")
         if agent_actual_collection_times and NUM_AGENTS > 0:
              print(f"Average actual collection time: {sum(agent_actual_collection_times) / NUM_AGENTS:.2f}")
              print(f"Actual collection time variance: {np.var(agent_actual_collection_times):.2f}")
              print(f"Actual collection time standard deviation: {np.std(agent_actual_collection_times):.2f}")
              max_time = max(agent_actual_collection_times) if agent_actual_collection_times else 0
              min_time = min(agent_actual_collection_times) if agent_actual_collection_times else 0
              print(f"Actual collection time max difference: {max_time - min_time}")

         break

if not (all_currently_spawned_targets_processed and all_targets_spawned):
     print(f"\nSimulation ended manually at step {iteration_count} at simulation time {simulation_time_seconds:.2f} seconds.")
     print(f"Total Targets Initially: {NUM_TARGETS}. Total Dynamic Targets Spawned: {dynamic_targets_spawned_count}. Total Targets in Simulation: {len(targets)}. ===")
     print(f"Total Targets Collected: {total_targets_collected_count}.")
     expired_count = sum(1 for i in range(len(targets)) if (i < len(target_remaining_times) and target_remaining_times[i] <= 0) and not collected[i])
     print(f"Expired targets: {expired_count}.")
     uncollected_and_not_expired = sum(1 for i in range(len(targets)) if not collected[i] and (i < len(target_remaining_times) and target_remaining_times[i] > 0))
     print(f"Uncollected and not expired targets remaining: {uncollected_and_not_expired}.")


print("Simulation finished.")