import numpy as np
import matplotlib.pyplot as plt
import time
import random

num_agents = 3
num_targets = 10
iterations = 100
population_size = 30
w_init = 0.9
w_final = 0.4
c1 = 2.0
c2 = 2.0
speed = 0.5
v_max = 0.1

def initialize_environment():
    agents = np.random.rand(num_agents, 2) * 10
    targets = np.random.rand(num_targets, 2) * 10

    return agents, targets

def initialize_tracking():
    target_collected = [False] * num_targets
    distance_traveled = np.zeros(num_agents)
    targets_reached = np.zeros(num_agents, dtype=int)
    targets_collected_over_time = []
    fitness_over_time = []
    return target_collected, distance_traveled, targets_reached, targets_collected_over_time, fitness_over_time

def initialize_pso():
    particles = np.random.rand(population_size, num_agents, num_targets)


    velocities = np.random.uniform(-v_max, v_max, (population_size, num_agents, num_targets))


    pbest = particles.copy()
    pbest_fitness = np.full(population_size, float('inf'))


    gbest = particles[0].copy()
    gbest_fitness = float('inf')

    return particles, velocities, pbest, pbest_fitness, gbest, gbest_fitness

def decode_solution(particle, target_collected):
    assignments = np.full(num_agents, -1, dtype=int)

    priorities = []
    for i in range(num_agents):
        for j in range(num_targets):
            if not target_collected[j]:
                priorities.append((i, j, particle[i, j]))

    priorities.sort(key=lambda x: x[2], reverse=True)

    assigned_targets = set()
    for agent_idx, target_idx, _ in priorities:
        if assignments[agent_idx] == -1 and target_idx not in assigned_targets:
            assignments[agent_idx] = target_idx
            assigned_targets.add(target_idx)

            if -1 not in assignments or len(assigned_targets) == sum(1 for t in range(num_targets) if not target_collected[t]):
                break

    return assignments

def calculate_fitness(particle, agents, targets, target_collected, distance_traveled=None, targets_reached=None):
    assignments = decode_solution(particle, target_collected)

    agent_distances = np.zeros(num_agents)
    agent_targets = np.zeros(num_agents)

    if distance_traveled is not None:
        agent_distances = distance_traveled.copy()
    if targets_reached is not None:
        agent_targets = targets_reached.copy()

    for i in range(num_agents):
        target_idx = assignments[i]
        if target_idx != -1:
            distance = np.linalg.norm(agents[i] - targets[target_idx])
            agent_distances[i] += distance
            agent_targets[i] += 1

    total_distance = np.sum(agent_distances)

    distance_cv = 0
    if np.mean(agent_distances) > 0:
        distance_cv = np.std(agent_distances) / np.mean(agent_distances)

    targets_cv = 0
    if np.mean(agent_targets) > 0:
        targets_cv = np.std(agent_targets) / np.mean(agent_targets)

    distance_range = np.max(agent_distances) - np.min(agent_distances) if len(agent_distances) > 0 else 0

    targets_range = np.max(agent_targets) - np.min(agent_targets) if len(agent_targets) > 0 else 0

    unassigned_agents = sum(1 for a in assignments if a == -1)
    unassigned_penalty = unassigned_agents * 20

    fitness = (
        0.3 * total_distance +
        2.0 * distance_range +
        3.0 * targets_range +
        1.5 * distance_cv * total_distance +
        1.5 * targets_cv * num_targets +
        unassigned_penalty
    )

    return fitness

def setup_visualization():
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    return fig, ax

def update_plot(ax, assignments, agents, targets, target_collected, iteration, distance_traveled, targets_reached):
    ax.clear()

    uncollected = np.where(np.logical_not(target_collected))[0]
    if len(uncollected) > 0:
        ax.scatter(targets[uncollected, 0], targets[uncollected, 1],
                   c='red', marker='x', s=100, label='Targets')

    collected = np.where(target_collected)[0]
    if len(collected) > 0:
        ax.scatter(targets[collected, 0], targets[collected, 1],
                   c='gray', marker='x', s=50, alpha=0.3, label='Collected')

    if np.max(targets_reached) > 0:
        workload_norm = targets_reached / np.max(targets_reached)
        colors = plt.cm.YlOrRd(workload_norm)
    else:
        colors = ['blue'] * num_agents

    for i in range(num_agents):
        ax.scatter(agents[i, 0], agents[i, 1], color=colors[i], marker='o', s=150)

        ax.annotate(f"A{i}\n({targets_reached[i]} targets, {distance_traveled[i]:.1f} dist)",
                   (agents[i][0], agents[i][1]),
                   fontsize=9, fontweight='bold', ha='center', va='bottom')

    for i in range(num_agents):
        target_idx = assignments[i]
        if target_idx != -1 and not target_collected[target_idx]:
            ax.plot([agents[i][0], targets[target_idx][0]],
                    [agents[i][1], targets[target_idx][1]],
                    'k--', alpha=0.7)

    for i in range(num_targets):
        if not target_collected[i]:
            ax.annotate(f"T{i}", (targets[i][0], targets[i][1]), fontsize=9)

    distance_cv = np.std(distance_traveled) / np.mean(distance_traveled) if np.mean(distance_traveled) > 0 else 0
    targets_cv = np.std(targets_reached) / np.mean(targets_reached) if np.mean(targets_reached) > 0 else 0

    ax.set_title(f"PSO Fair Task Allocation - Iteration {iteration}\n" +
                 f"Distance CV: {distance_cv:.2f}, Targets CV: {targets_cv:.2f}\n" +
                 f"Distance Range: {np.max(distance_traveled) - np.min(distance_traveled):.1f}, " +
                 f"Targets Range: {np.max(targets_reached) - np.min(targets_reached)}",
                 fontsize=10)

    ax.legend(loc='upper right')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.subplots_adjust(bottom=0.2)
    pos = ax.get_position()
    fairness_ax = plt.axes([pos.x0, 0.05, pos.width, 0.1])

    bar_width = 0.35
    x = np.arange(num_agents)

    fairness_ax.bar(x - bar_width/2, targets_reached, bar_width, label='Targets', color='green')

    if np.max(distance_traveled) > 0:
        norm_dist = distance_traveled / np.max(distance_traveled) * np.max(targets_reached)
        fairness_ax.bar(x + bar_width/2, norm_dist, bar_width, label='Dist (norm)', color='orange')

    fairness_ax.set_xticks(x)
    fairness_ax.set_xticklabels([f'Agent {i}' for i in range(num_agents)])
    fairness_ax.legend(loc='upper right', fontsize=8)
    fairness_ax.set_title('Workload Balance', fontsize=9)

    plt.draw()
    plt.pause(0.05)

def run_simulation():
    np.random.seed(int(time.time()))
    random.seed(int(time.time()))

    agents, targets = initialize_environment()

    target_collected, distance_traveled, targets_reached, targets_collected_over_time, fitness_over_time = initialize_tracking()

    particles, velocities, pbest, pbest_fitness, gbest, gbest_fitness = initialize_pso()

    fig, ax = setup_visualization()

    print("PSO Parameters:")
    print(f"Population size: {population_size}")
    print(f"Inertia weight: {w_init} -> {w_final}")
    print(f"Cognitive coefficient (c1): {c1}")
    print(f"Social coefficient (c2): {c2}")
    print(f"Maximum velocity: {v_max}")
    print("\nStarting simulation with fairness-focused fitness...\n")

    distance_cv_history = []
    targets_cv_history = []

    for iteration in range(iterations):
        w = w_init - (w_init - w_final) * (iteration / iterations)

        for p in range(population_size):
            current_fitness = calculate_fitness(particles[p], agents, targets, target_collected,
                                                 distance_traveled, targets_reached)

            if current_fitness < pbest_fitness[p]:
                pbest_fitness[p] = current_fitness
                pbest[p] = particles[p].copy()

            if current_fitness < gbest_fitness:
                gbest_fitness = current_fitness
                gbest = particles[p].copy()

        for p in range(population_size):
            r1 = random.random()
            r2 = random.random()

            velocities[p] = (w * velocities[p] +
                             c1 * r1 * (pbest[p] - particles[p]) +
                             c2 * r2 * (gbest - particles[p]))

            velocities[p] = np.clip(velocities[p], -v_max, v_max)

            particles[p] += velocities[p]

            particles[p] = np.clip(particles[p], 0, 1)

        best_assignments = decode_solution(gbest, target_collected)

        for i in range(num_agents):
            target_idx = best_assignments[i]

            if target_idx != -1 and not target_collected[target_idx]:
                direction = targets[target_idx] - agents[i]
                distance = np.linalg.norm(direction)

                if distance < speed:
                    agents[i] = targets[target_idx].copy()
                    target_collected[target_idx] = True
                    targets_reached[i] += 1
                    distance_traveled[i] += distance
                else:
                    agents[i] += (direction / distance) * speed
                    distance_traveled[i] += speed

        distance_cv = np.std(distance_traveled) / np.mean(distance_traveled) if np.mean(distance_traveled) > 0 else 0
        targets_cv = np.std(targets_reached) / np.mean(targets_reached) if np.mean(targets_reached) > 0 else 0

        distance_cv_history.append(distance_cv)
        targets_cv_history.append(targets_cv)

        update_plot(ax, best_assignments, agents, targets, target_collected,
                    iteration, distance_traveled, targets_reached)

        current_fitness = calculate_fitness(gbest, agents, targets, target_collected,
                                             distance_traveled, targets_reached)
        fitness_over_time.append(current_fitness)
        targets_collected_over_time.append(sum(target_collected))

        remaining_targets = num_targets - sum(target_collected)
        print(f"Iteration {iteration}: Fitness = {current_fitness:.2f}, "
              f"Targets remaining = {remaining_targets}/{num_targets}")
        print(f"  Fairness - Distance CV: {distance_cv:.2f}, Targets CV: {targets_cv:.2f}")
        print(f"  Distance Range: {np.max(distance_traveled) - np.min(distance_traveled):.1f}, "
              f"Targets Range: {np.max(targets_reached) - np.min(targets_reached)}")

        if all(target_collected):
            print(f"\nAll targets collected at iteration {iteration}")
            break

        time.sleep(0.1)

    plt.ioff()

    print("\nFinal Statistics:")
    print(f"Distance traveled by each agent: {distance_traveled}")
    print(f"Targets reached by each agent: {targets_reached}")
    print(f"Distance coefficient of variation: {distance_cv:.2f}")
    print(f"Targets coefficient of variation: {targets_cv:.2f}")
    print(f"Final PSO best fitness: {gbest_fitness:.2f}")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(fitness_over_time, 'b-', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Value")
    plt.title("Fitness Over Time")
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(targets_collected_over_time, 'g-', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Number of Targets")
    plt.title("Targets Collected Over Time")
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(distance_cv_history, 'r-', linewidth=2, label='Distance CV')
    plt.plot(targets_cv_history, 'b-', linewidth=2, label='Targets CV')
    plt.xlabel("Iteration")
    plt.ylabel("Coefficient of Variation")
    plt.title("Fairness Metrics Over Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    bars = plt.bar(range(num_agents), distance_traveled, color='orange')
    plt.axhline(y=np.mean(distance_traveled), color='r', linestyle='-', label='Mean')
    plt.xlabel("Agent")
    plt.ylabel("Total Distance")
    plt.title("Distance Traveled by Each Agent")
    plt.xticks(range(num_agents), [f"Agent {i}" for i in range(num_agents)])
    plt.legend()

    plt.subplot(2, 3, 5)
    bars = plt.bar(range(num_agents), targets_reached, color='green')
    plt.axhline(y=np.mean(targets_reached), color='r', linestyle='-', label='Mean')
    plt.xlabel("Agent")
    plt.ylabel("Count")
    plt.title("Targets Reached by Each Agent")
    plt.xticks(range(num_agents), [f"Agent {i}" for i in range(num_agents)])
    plt.legend()

    plt.subplot(2, 3, 6)
    bar_width = 0.35
    x = np.arange(num_agents)

    norm_distance = distance_traveled / np.max(distance_traveled) if np.max(distance_traveled) > 0 else distance_traveled
    norm_targets = targets_reached / np.max(targets_reached) if np.max(targets_reached) > 0 else targets_reached

    plt.bar(x - bar_width/2, norm_targets, bar_width, label='Normalized Targets', color='green')
    plt.bar(x + bar_width/2, norm_distance, bar_width, label='Normalized Distance', color='orange')

    plt.xlabel("Agent")
    plt.ylabel("Normalized Value")
    plt.title("Workload & Distance Balance")
    plt.xticks(x, [f"Agent {i}" for i in range(num_agents)])
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()