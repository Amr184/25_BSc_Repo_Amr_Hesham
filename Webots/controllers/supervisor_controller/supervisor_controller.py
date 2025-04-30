from controller import Supervisor
import math

# Constants
TIME_STEP = 32
TILE_SIZE = 0.2857
X_MIN, Z_MIN = -0.9, -0.83
START_POSITION = [-0.858181, 0.87578, -6.3962e-05]
GOAL_POSITION = (2, 6)
EPISODES = 500
MOVE_LIMIT = 3000

def distance_2d(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

supervisor = Supervisor()
epuck_node = supervisor.getFromDef("e-puck")
if epuck_node is None:
    print("ERROR: e-puck not found")
    exit()

emitter = supervisor.getDevice("emitter")

for episode in range(EPISODES):
    print(f"\n[SUPERVISOR] Starting Episode {episode + 1}")
    epuck_node.getField("translation").setSFVec3f(START_POSITION)
    supervisor.simulationResetPhysics()

    moves = 0
    last_tile = (-1, -1)

    while supervisor.step(TIME_STEP) != -1:
        pos = epuck_node.getPosition()
        x, z = pos[0], pos[1]
        grid_x = round((x - X_MIN) / TILE_SIZE)
        grid_y = round((z - Z_MIN) / TILE_SIZE)
        tile = (grid_x, grid_y)

        if tile != last_tile:
            print(f"[SUPERVISOR] Robot moved to tile {tile} | Real pos: ({x:.3f}, {z:.3f})")
            moves += 1
            last_tile = tile

        emitter.send(f"{episode},{grid_x},{grid_y}".encode())

        if tile == GOAL_POSITION:
            print(f"[SUPERVISOR] Episode {episode+1}: Reached goal in {moves} moves!")
            break

        if moves >= MOVE_LIMIT:
            print(f"[SUPERVISOR] Episode {episode+1}: Move limit reached.")
            break

    supervisor.step(TIME_STEP * 10)