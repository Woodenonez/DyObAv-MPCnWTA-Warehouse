import numpy as np

class TimedElasticBand:
    def __init__(self, robot_radius, dt, max_time, max_iterations, max_error):
        self.robot_radius = robot_radius
        self.dt = dt
        self.max_time = max_time
        self.max_iterations = max_iterations
        self.max_error = max_error
        self.path = None
        self.obstacles = None
        self.start = None
        self.goal = None
        self.path_length = None
        self.time = None
        self.velocities = None
        self.accelerations = None
        self.iterations = None
        self.error = None

    def set_path(self, path):
        self.path = path
        self.path_length = len(path)
        self.time = np.zeros(self.path_length)
        self.velocities = np.zeros(self.path_length)
        self.accelerations = np.zeros(self.path_length)
        self.iterations = 0
        self.error = np.inf

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def set_start(self, start):
        self.start = start

    def set_goal(self, goal):
        self.goal = goal

    def optimize(self):
        while self.iterations < self.max_iterations and self.error > self.max_error:
            self.iterations += 1
            self.compute_velocities()
            self.compute_accelerations()
            self.compute_time()
            self.compute_error()

    def compute_velocities(self):
        for i in range(1, self.path_length - 1):
            self.velocities[i] = self.compute_velocity(i)

    def compute_velocity(self, i):
        distance = np.linalg.norm(self.path[i + 1] - self.path[i - 1])
        velocity = distance / (2 * self.dt)
        return min(velocity, self.robot_radius)

    def compute_accelerations(self):
        for i in range(1, self.path_length - 1):
            self.accelerations[i] = self.compute_acceleration(i)

    def compute_acceleration(self, i):
        velocity = self.velocities[i]
        prev_velocity = self.velocities[i - 1]
        next_velocity = self.velocities[i + 1]
        acceleration = (next_velocity - prev_velocity) / (2 * self.dt)
        return min(acceleration, (velocity - prev_velocity) / self.dt, (next_velocity - velocity) / self.dt)

    def compute_time(self):
        for i in range(1, self.path_length):
            self.time[i] = self.time[i - 1] + self.dt

    def compute_error(self):
        self.error = 0
        for i in range(1, self.path_length - 1):
            distance = np.linalg.norm(self.path[i + 1] - self.path[i - 1])
            velocity = self.velocities[i]
            acceleration = self.accelerations[i]
            error = (velocity ** 2 + acceleration ** 2) * distance
            self.error += error

    def get_trajectory(self):
        trajectory = []
        for i in range(self.path_length):
            pose = self.path[i]
            velocity = self.velocities[i]
            acceleration = self.accelerations[i]
            trajectory.append((pose, velocity, acceleration))
        return trajectory
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define the path, obstacles, start, and goal for your mobile robot
    path = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]])
    start = np.array([0, 0])
    goal = np.array([4, 0])

    # Define the parameters for the Timed Elastic Band algorithm
    robot_radius = 0.1
    dt = 0.1
    max_time = 10.0
    max_iterations = 100
    max_error = 0.1

    # Create a TimedElasticBand object and set the path, obstacles, start, and goal
    teb = TimedElasticBand(robot_radius, dt, max_time, max_iterations, max_error)
    teb.set_path(path)
    teb.set_start(start)
    teb.set_goal(goal)

    # Define a function to update the obstacles
    def update_obstacles():
        obstacles = [[1.0, 0.5],
                     [2.0, 1.0],
                     [2.5, 0.5]]
        # while True:
        #     x = input("Enter x-coordinate of obstacle (or 'done' to finish): ")
        #     if x == 'done':
        #         break
        #     y = input("Enter y-coordinate of obstacle: ")
        #     obstacles.append([float(x), float(y)])
        teb.set_obstacles(obstacles)

    # Optimize the trajectory using the Timed Elastic Band algorithm
    update_obstacles()
    teb.optimize()

    # Retrieve the optimized trajectory
    trajectory = teb.get_trajectory()

    # Plot the path and optimized trajectory
    plt.plot(path[:, 0], path[:, 1], 'b--', label='Path')
    plt.plot(start[0], start[1], 'go', label='Start')
    plt.plot(goal[0], goal[1], 'ro', label='Goal')
    for obstacle in teb.obstacles:
        circle = plt.Circle(obstacle, robot_radius, color='k')
        plt.gca().add_artist(circle)
    for pose, _, _ in trajectory:
        plt.plot(pose[0], pose[1], 'g.')
    plt.legend()
    plt.axis('equal')
    plt.show()



