from controller import Supervisor

TIME_STEP = 32  # Time step in milliseconds

# Create a Supervisor instance
supervisor = Supervisor()

# Get the e-puck robot by its DEF name (Make sure your e-puck has DEF "EPUCK" in Webots)
epuck_node = supervisor.getFromDef("EPUCK")

if epuck_node is None:
    print("Error: Could not find EPUCK in the scene. Make sure it has a DEF name 'EPUCK'.")
    exit()

# Get the translation field to track position
position_field = epuck_node.getField("translation")

while supervisor.step(TIME_STEP) != -1:
    # Get the current position of the e-puck
    position = epuck_node.getPosition()
    
    # Print the position (x, y, z)
    print(f"e-puck Position: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f}")
