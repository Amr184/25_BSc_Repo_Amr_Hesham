from controller import Robot, Receiver, Motor
import numpy as np

# Constants
TIME_STEP = 32
SPEED = 6.0
MOVE_DURATION = 70
TURN_DURATION = 46
GRID_SIZE = 7
ACTIONS = [0, 1]  # 0: FORWARD, 1: RIGHT (relative)
OBSTACLES = [(3, 4), (3, 3), (3, 2), (3, 1)]
GOAL = (2, 6)

# Q-learning parameters
alpha = 0.3
gamma = 0.9
epsilon = 0.5
epsilon_decay = 0.005
min_epsilon = 0.05
episodes = 500

# Q-table: [x][y][facing][action]
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4, len(ACTIONS)))

# Initialize robot
robot = Robot()
receiver = robot.getDevice("receiver")
receiver.enable(TIME_STEP)

motors = [robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")]
for m in motors:
    m.setPosition(float('inf'))
    m.setVelocity(0)

facing = 0  # 0: right (-y), 1: down (-x), 2: left (+y), 3: up (+x)

# Correct direction deltas for your coordinate system
direction_deltas = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # → ↓ ← ↑

def turn_to(new_facing):
    global facing
    diff = (new_facing - facing) % 4

    if diff == 0:
        return
    if diff == 1:
        motors[0].setVelocity(3 / 2)
        motors[1].setVelocity(-3 / 2)
        steps = TURN_DURATION
    elif diff == 2:
        motors[0].setVelocity(3)
        motors[1].setVelocity(-3)
        steps = TURN_DURATION
    elif diff == 3:
        motors[0].setVelocity(-3 / 2)
        motors[1].setVelocity(3 / 2)
        steps = TURN_DURATION

    for _ in range(steps):
        robot.step(TIME_STEP)

    motors[0].setVelocity(0)
    motors[1].setVelocity(0)
    facing = new_facing

def move_forward():
    motors[0].setVelocity(SPEED)
    motors[1].setVelocity(SPEED)
    for _ in range(MOVE_DURATION):
        robot.step(TIME_STEP)
    motors[0].setVelocity(0)
    motors[1].setVelocity(0)

def execute_action(action):
    global facing
    new_facing = (facing + action) % 4
    turn_to(new_facing)
    move_forward()

def get_valid_actions(x, y, facing):
    valid = []
    for rel_action in ACTIONS:
        global_dir = (facing + rel_action) % 4
        dx, dy = direction_deltas[global_dir]
        new_x = x + dx
        new_y = y + dy

        if (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE):
            if (new_x, new_y) not in OBSTACLES:
                valid.append(rel_action)
            else:
                print(f"Blocked action: {['FWD','RIGHT'][rel_action]} → Obstacle")
        else:
            print(f"Blocked action: {['FWD','RIGHT'][rel_action]} → Out of bounds")
    return valid

# Main loop
current_episode = -1
state = (0, 0)
total_reward = 0

while robot.step(TIME_STEP) != -1:
    if receiver.getQueueLength() > 0:
        msg = receiver.getString()
        receiver.nextPacket()
        episode, gx, gy = map(int, msg.split(","))

        if episode != current_episode:
            current_episode = episode
            facing = 3  # Facing ↑ (+x)
            print(f"\n[EPISODE {episode+1}] Starting new episode.")
            moves = 0
            total_reward = 0
            state = (gx, gy)  # Only update state at beginning of episode

        best_actions = Q_table[state[0], state[1], facing]
        valid = get_valid_actions(state[0], state[1], facing)
        if not valid:
            print("No valid actions available. Skipping step.")
            continue

        explore = np.random.rand() < epsilon
        if explore:
            action = np.random.choice(valid)
        else:
            masked_actions = np.full(len(ACTIONS), -np.inf)
            for a in valid:
                masked_actions[a] = best_actions[a]
            action = np.argmax(masked_actions)

        global_dir = (facing + action) % 4
        dx, dy = direction_deltas[global_dir]
        proposed_next = (state[0] + dx, state[1] + dy)

        if not (0 <= proposed_next[0] < GRID_SIZE and 0 <= proposed_next[1] < GRID_SIZE) or proposed_next in OBSTACLES:
            reward = -1.0
            next_state = state  # Stay in place
        else:
            reward = 1.0 if proposed_next == GOAL else -0.1
            next_state = proposed_next

        Q_table[state[0], state[1], facing, action] = (1 - alpha) * Q_table[state[0], state[1], facing, action] + \
            alpha * (reward + gamma * np.max(Q_table[next_state[0], next_state[1]]))

        total_reward += reward
        print(f"State: {state} | Facing: {['→','↓','←','↑'][facing]} | Action: {['FWD','RIGHT'][action]} | "
              f"Next: {next_state} | Reward: {reward:.2f} | Total: {total_reward:.2f} | Explore: {explore}")

        execute_action(action)
        state = next_state  # ✅ Update state *after* moving
        moves += 1

        if epsilon > min_epsilon:
            epsilon -= epsilon_decay

        # Wait for supervisor only to sync before next episode
        while receiver.getQueueLength() == 0:
            robot.step(TIME_STEP)