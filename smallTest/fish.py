import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from itertools import product

# Constants for the 3D grid and chunk system
ROWS, COLUMNS, DEPTH = 100, 100, 100
CHUNK_SIZE = 10  # Size of each chunk
FISH_SIZE = 2.0  # Uniform size for all fish
SPEED = 3  # Uniform speed for all fish
NEIGHBOR_RADIUS = 10.0  # Radius to consider other fish as neighbors
SEPARATION_WEIGHT = 2
ALIGNMENT_WEIGHT = 3.5
COHESION_WEIGHT = 2.5
BOUNDARY_AVOIDANCE_WEIGHT = 5.0  # Weight for boundary avoidance

# Calculate the number of chunks along each axis
NUM_CHUNKS_X = ROWS // CHUNK_SIZE
NUM_CHUNKS_Y = COLUMNS // CHUNK_SIZE
NUM_CHUNKS_Z = DEPTH // CHUNK_SIZE


# Fish class implementing Boid behavior
class Fish:
    def __init__(self):
        self.position = np.array([random.uniform(0, ROWS), random.uniform(0, COLUMNS), random.uniform(0, DEPTH)])
        self.velocity = np.random.rand(3) * 2 - 1  # Random initial velocity
        self.velocity = self.velocity / np.linalg.norm(self.velocity) * SPEED  # Normalize and apply speed
        self.chunk = self.get_chunk()

    def get_chunk(self):
        """ Determine the chunk index of the fish based on its position. """
        if np.any(np.isnan(self.position)):
            raise ValueError("Fish position contains NaN values.")
        chunk_x = int(self.position[0] // CHUNK_SIZE)
        chunk_y = int(self.position[1] // CHUNK_SIZE)
        chunk_z = int(self.position[2] // CHUNK_SIZE)
        return (chunk_x, chunk_y, chunk_z)

    def update_chunk(self):
        """ Update the chunk index if the fish moves to a new chunk. """
        new_chunk = self.get_chunk()
        if new_chunk != self.chunk:
            self.chunk = new_chunk

    def apply_boid_rules(self, fish_list):
        separation = np.zeros(3)
        alignment = np.zeros(3)
        cohesion = np.zeros(3)
        boundary_avoidance = np.zeros(3)
        total_neighbors = 0

        # Iterate over the chunk and adjacent chunks
        for dx, dy, dz in product([-1, 0, 1], repeat=3):
            neighbor_chunk = (self.chunk[0] + dx, self.chunk[1] + dy, self.chunk[2] + dz)
            if all(0 <= n < num for n, num in zip(neighbor_chunk, [NUM_CHUNKS_X, NUM_CHUNKS_Y, NUM_CHUNKS_Z])):
                for other_fish in fish_list:
                    if other_fish.chunk == neighbor_chunk:
                        distance = np.linalg.norm(self.position - other_fish.position)
                        if distance < NEIGHBOR_RADIUS:
                            # Separation: steer to avoid crowding neighbors
                            if distance > 0:
                                separation += (self.position - other_fish.position) / distance

                            # Alignment: steer towards the average heading of neighbors
                            alignment += other_fish.velocity

                            # Cohesion: steer towards the average position of neighbors
                            cohesion += other_fish.position

                            total_neighbors += 1

        if total_neighbors > 0:
            # Average the alignment and cohesion vectors
            alignment /= total_neighbors
            cohesion /= total_neighbors

            # Calculate the cohesion vector
            direction_to_cohesion = cohesion - self.position
            norm_direction_to_cohesion = np.linalg.norm(direction_to_cohesion)
            if norm_direction_to_cohesion > 0:
                cohesion = direction_to_cohesion / norm_direction_to_cohesion
            else:
                cohesion = np.zeros(3)  # Avoid division by zero if there's no direction

            # Apply weights to each of the steering behaviors
            separation *= SEPARATION_WEIGHT
            alignment *= ALIGNMENT_WEIGHT
            cohesion *= COHESION_WEIGHT

            # Combine the steering behaviors
            self.velocity += separation + alignment + cohesion

        # Boundary avoidance logic
        boundary_avoidance += self.avoid_boundaries()
        self.velocity += boundary_avoidance
        self.velocity = self.velocity / np.linalg.norm(self.velocity) * SPEED

    def avoid_boundaries(self):
        """ Steers the fish away from the edges of the tank. """
        avoidance = np.zeros(3)
        for i, pos in enumerate(self.position):
            if pos < 5:
                avoidance[i] = 1 / (pos + 0.1)  # Avoid the lower boundary
            elif pos > [ROWS, COLUMNS, DEPTH][i] - 5:
                avoidance[i] = -1 / ([ROWS, COLUMNS, DEPTH][i] - pos + 0.1)  # Avoid the upper boundary

        return avoidance * BOUNDARY_AVOIDANCE_WEIGHT

    def move(self, fish_list):
        self.apply_boid_rules(fish_list)
        new_position = self.position + self.velocity

        # Ensure the fish stays within bounds
        new_position = np.clip(new_position, [0, 0, 0], [ROWS - 1, COLUMNS - 1, DEPTH - 1])
        self.position = new_position
        self.update_chunk()


# Initialize multiple fish with uniform size and speed
fish_list = [Fish() for _ in range(30)]

# Plotting setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define arrow thickness
ARROW_THICKNESS = 5


def plot_fish(fish_list):
    ax.clear()
    ax.set_xlim(0, ROWS - 1)
    ax.set_ylim(0, COLUMNS - 1)
    ax.set_zlim(0, DEPTH - 1)

    for fish in fish_list:
        # Position and direction
        pos = fish.position
        vel = fish.velocity

        # Normalize the velocity for consistent arrow size
        norm_vel = vel / np.linalg.norm(vel)

        # Draw the fish as arrows
        ax.quiver(pos[0], pos[1], pos[2], norm_vel[0], norm_vel[1], norm_vel[2],
                  length=10.0, color='gray', arrow_length_ratio=0.1, linewidth=ARROW_THICKNESS)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Fish Count: {len(fish_list)}')
    plt.draw()
    plt.pause(0.05)  # Pause to update the plot


# Run the simulation
plt.ion()
for _ in range(5000):  # Simulate for 200 steps
    for fish in fish_list:
        fish.move(fish_list)
    plot_fish(fish_list)

plt.ioff()
plt.show()
