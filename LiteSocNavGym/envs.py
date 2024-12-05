import random
import sys
from typing import Any, Dict, Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register


class LiteSocNavGym(gym.Env):
    """
    Lightweight Social Navigation Gymnasium Environment where a robot navigates toward a goal while avoiding humans and tables.
    Humans navigate toward their own goals while avoiding collisions with other humans and tables.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        resolution=700.0,
        resolution_view=1000.0,
        map_size=8.0,
        margin=0.5,
        max_ticks=250,
        timestep=0.1,
        robot_radius=0.15,
        goal_radius=0.5,
        min_humans=1,
        max_humans=3,
        min_tables=1,
        max_tables=1,
        human_threshold=0.4,
        reach_reward=1.0,
        outofmap_reward=-0.5,
        maxticks_reward=-0.3,
        alive_reward=-0.003,
        collision_reward=-0.5,
        discomfort_distance=0.5,
        discomfort_penalty_factor=0.005,
        max_advance=1.4,
        human_speed=0.3,
        max_rotation=np.pi / 2,
        milliseconds=30,
        debug: int = 0,
        render_mode="rgb_array",
        robot_view=True,
        robot_size=0.15,
        human_size=0.15,
        goal_size=0.25,
        table_size=(1.0, 0.6),
        seed: Optional[int] = None,
        human_debug=False,
    ):
        super(LiteSocNavGym, self).__init__()
        self.resolution = resolution
        self.resolution_view = resolution_view
        self.map_size = map_size
        self.margin = margin
        self.max_ticks = max_ticks
        self.timestep = timestep
        self.robot_radius = robot_radius
        self.goal_radius = goal_radius
        self.min_humans = min_humans
        self.max_humans = max_humans
        self.min_tables = min_tables
        self.max_tables = max_tables
        self.human_threshold = human_threshold
        self.reach_reward = reach_reward
        self.outofmap_reward = outofmap_reward
        self.maxticks_reward = maxticks_reward
        self.alive_reward = alive_reward
        self.collision_reward = collision_reward
        self.discomfort_distance = discomfort_distance
        self.discomfort_penalty_factor = discomfort_penalty_factor
        self.max_advance = max_advance
        self.human_speed = human_speed
        self.max_rotation = max_rotation
        self.milliseconds = milliseconds
        self.debug = debug
        self.render_mode = render_mode
        self.robot_view = robot_view
        self.robot_size = robot_size
        self.human_size = human_size
        self.goal_size = goal_size
        self.table_size = table_size
        self.seed_value = seed
        self.human_debug = human_debug

        self.goal_threshold = self.robot_radius + self.goal_radius
        self.pixel_to_world = resolution / map_size

        self.window_initialised = False

        # Initialize the environment with the provided or default seed
        if self.seed_value is not None:
            self.seed(self.seed_value)
        else:
            self.seed(42)  # Default seed

        self.reset()

    def seed(self, seed: Optional[int] = None):
        """
        Sets the random seed for reproducibility.
        """
        if seed is None:
            seed = 42  # Default seed value
        self.seed_value = seed
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)

    @property
    def observation_space(self):
        """
        Defines the observation space of the environment.
        The observation includes:
            - Robot's orientation or position and orientation
            - Goal's relative or absolute position
            - Humans' relative or absolute positions and orientations
            - Tables' relative or absolute positions and sizes
            - Masks indicating active humans and tables
        """
        num_humans = self.max_humans
        num_tables = self.max_tables

        if self.robot_view:
            # Define the dimensions for each component
            robot_dim = 2  # [cos(theta), sin(theta)]
            goal_dim = 2   # [relative_x, relative_y]
            human_dim = 4  # [relative_x, relative_y, cos(relative_orientation), sin(relative_orientation)]
            table_dim = 4  # [relative_x, relative_y, width, height]
            
            # Masks
            human_mask_dim = num_humans  # 1 per human
            table_mask_dim = num_tables  # 1 per table

            # Calculate low and high bounds
            low = np.array(
                [-1.0, -1.0]  # Robot orientation
                + [-self.map_size, -self.map_size]  # Goal relative position
                + [-self.map_size, -self.map_size, -1.0, -1.0] * num_humans  # Humans
                + [-self.map_size, -self.map_size, 0.0, 0.0] * num_tables  # Tables
                + [0.0] * (human_mask_dim + table_mask_dim)  # Masks
            )
            high = np.array(
                [+1.0, +1.0]  # Robot orientation
                + [+self.map_size, +self.map_size]  # Goal relative position
                + [+self.map_size, +self.map_size, +1.0, +1.0] * num_humans  # Humans
                + [+self.map_size, +self.map_size, self.map_size, self.map_size] * num_tables  # Tables
                + [1.0] * (human_mask_dim + table_mask_dim)  # Masks
            )
        else:
            # Similar structure when not using robot_view
            # Adjust dimensions accordingly
            robot_dim = 4  # [x, y, cos(theta), sin(theta)]
            goal_dim = 2   # [x, y]
            human_dim = 4  # [x, y, cos(theta), sin(theta)]
            table_dim = 4  # [x, y, width, height]
            
            # Masks
            human_mask_dim = num_humans
            table_mask_dim = num_tables

            low = np.array(
                [-self.map_size / 2, -self.map_size / 2, -1.0, -1.0]  # Robot position and orientation
                + [-self.map_size / 2, -self.map_size / 2]  # Goal position
                + [-self.map_size / 2, -self.map_size / 2, -1.0, -1.0] * num_humans  # Humans
                + [-self.map_size / 2, -self.map_size / 2, 0.0, 0.0] * num_tables  # Tables
                + [0.0] * (human_mask_dim + table_mask_dim)  # Masks
            )
            high = np.array(
                [+self.map_size / 2, +self.map_size / 2, +1.0, +1.0]  # Robot position and orientation
                + [+self.map_size / 2, +self.map_size / 2]  # Goal position
                + [+self.map_size / 2, +self.map_size / 2, +1.0, +1.0] * num_humans  # Humans
                + [+self.map_size / 2, +self.map_size / 2, self.map_size, self.map_size] * num_tables  # Tables
                + [1.0] * (human_mask_dim + table_mask_dim)  # Masks
            )

        return spaces.Box(low, high, dtype=np.float32)

    @property
    def action_space(self):
        """
        Defines the action space of the environment.
        Continuous action space: [advance, rotation]
        """
        low = np.array([-1, -1])
        high = np.array([+1, +1])
        return spaces.Box(low, high, dtype=np.float32)

    @property
    def done(self):
        """
        Indicates whether the episode has terminated.
        """
        return self.robot_is_done

    def step(self, action_pre):
        """
        Executes one time step within the environment.
        """
        if self.robot_is_done:
            raise Exception("Step called within a finished episode!")

        action = self.process_action(action_pre)
        self.update_robot(action)
        self.update_humans()
        self.update_tables()

        # Increment ticks
        self.ticks += 1

        # Generate observation, compute reward, and check if done
        observation = self.get_observation()
        reward = self.compute_reward_and_ticks()
        done = self.robot_is_done
        info = {}

        self.cumulative_reward += reward

        if self.debug > 0:
            self.render()

        return observation, reward, done, False, info

    def process_action(self, action_pre):
        """
        Processes the raw action input to scale it appropriately.
        """
        action = np.array(action_pre)
        action[0] = ((action[0] + 1.0) / 2.0) * self.max_advance  # Scale advance
        action[1] = action[1] * self.max_rotation               # Scale rotation
        return action

    def update_robot(self, action):
        """
        Updates the robot's position and orientation based on the action.
        """
        moved = action[0] * self.timestep
        self.robot[0, 0] += np.cos(self.robot[0, 2]) * moved
        self.robot[0, 1] += np.sin(self.robot[0, 2]) * moved
        self.robot[0, 2] += action[1] * self.timestep
        # Normalize orientation between -pi and pi
        self.robot[0, 2] = self._normalize_angle(self.robot[0, 2])

        if self.debug > 0:
            print(f"Robot moved to ({self.robot[0,0]:.2f}, {self.robot[0,1]:.2f}) with orientation {self.robot[0,2]:.2f} radians.")

    def update_humans(self):
        """
        Updates each human's position and orientation, incorporating collision and obstacle avoidance.
        """
        for i in range(self.num_humans):
            human_pos = self.humans[i, :2]
            human_theta = self.humans[i, 2]
            human_goal = self.human_goals[i]
            goal_vector = human_goal - human_pos
            distance_to_goal = np.linalg.norm(goal_vector)

            # Goal Reached: Assign new goal
            if distance_to_goal < self.goal_radius:
                HALF_SIZE = self.map_size / 2.0 - self.margin
                while True:
                    new_goal = np.array([
                        random.uniform(-HALF_SIZE, HALF_SIZE),
                        random.uniform(-HALF_SIZE, HALF_SIZE)
                    ])
                    if not self.is_on_table(new_goal):
                        self.human_goals[i] = new_goal
                        if self.human_debug:
                            print(f"Human {i} reached goal. New goal assigned at {new_goal}.")
                        break

            # Compute movement towards goal
            if distance_to_goal > 0:
                desired_direction = goal_vector / distance_to_goal  # Unit vector toward goal
            else:
                desired_direction = np.array([0.0, 0.0])

            # Initialize avoidance vectors
            avoidance_vector = np.array([0.0, 0.0])

            # Collision Avoidance with Other Humans
            for j in range(self.num_humans):
                if j == i:
                    continue
                other_human_pos = self.humans[j, :2]
                vector_to_other = human_pos - other_human_pos
                distance = np.linalg.norm(vector_to_other)
                if distance < self.human_threshold and distance > 0:
                    # Repulsive force inversely proportional to distance
                    repulsion_strength = (self.human_threshold - distance) / self.human_threshold
                    avoidance_vector += (vector_to_other / distance) * repulsion_strength

            # Obstacle Avoidance with Tables
            for k in range(self.num_tables):
                table = self.tables[k]
                table_pos = table[:2]
                table_w, table_h = table[2], table[3]
                # Compute the closest point on the table rectangle to the human
                closest_x = np.clip(human_pos[0], table_pos[0] - table_w / 2, table_pos[0] + table_w / 2)
                closest_y = np.clip(human_pos[1], table_pos[1] - table_h / 2, table_pos[1] + table_h / 2)
                vector_to_table = human_pos - np.array([closest_x, closest_y])
                distance = np.linalg.norm(vector_to_table)
                min_distance = self.human_size + self.margin
                if distance < min_distance and distance > 0:
                    # Repulsive force inversely proportional to distance
                    repulsion_strength = (min_distance - distance) / min_distance
                    avoidance_vector += (vector_to_table / distance) * repulsion_strength

            # Combine goal direction and avoidance
            combined_vector = desired_direction + avoidance_vector
            combined_distance = np.linalg.norm(combined_vector)

            if combined_distance > 0:
                combined_direction = combined_vector / combined_distance
            else:
                combined_direction = np.array([0.0, 0.0])

            # Compute desired orientation based on combined direction
            desired_theta = np.arctan2(combined_direction[1], combined_direction[0])
            orientation_diff = self._normalize_angle(desired_theta - human_theta)

            # Limit the rotation to max_rotation * timestep
            max_rotation_step = self.max_rotation * self.timestep
            orientation_change = np.clip(orientation_diff, -max_rotation_step, max_rotation_step)
            new_theta = self._normalize_angle(human_theta + orientation_change)

            # Update orientation
            self.humans[i, 2] = new_theta

            # Compute movement step
            movement_step = self.human_speed * self.timestep
            new_x = human_pos[0] + movement_step * np.cos(new_theta)
            new_y = human_pos[1] + movement_step * np.sin(new_theta)

            # Check for potential collisions after movement
            collision = False
            # Collision with other humans
            for j in range(self.num_humans):
                if j == i:
                    continue
                other_human_pos = self.humans[j, :2]
                distance = np.linalg.norm([new_x - other_human_pos[0], new_y - other_human_pos[1]])
                if distance < (self.human_threshold + self.human_size):
                    collision = True
                    if self.human_debug:
                        print(f"Human {i} movement would collide with Human {j}. Movement aborted.")
                    break

            # Collision with tables
            if not collision:
                for k in range(self.num_tables):
                    table = self.tables[k]
                    table_pos = table[:2]
                    table_w, table_h = table[2], table[3]
                    # Compute the closest point on the table rectangle to the new position
                    closest_x = np.clip(new_x, table_pos[0] - table_w / 2, table_pos[0] + table_w / 2)
                    closest_y = np.clip(new_y, table_pos[1] - table_h / 2, table_pos[1] + table_h / 2)
                    vector_to_table = np.array([new_x, new_y]) - np.array([closest_x, closest_y])
                    distance = np.linalg.norm(vector_to_table)
                    min_distance = self.human_size + self.margin
                    if distance < min_distance:
                        collision = True
                        if self.human_debug:
                            print(f"Human {i} movement would collide with Table {k}. Movement aborted.")
                        break

            # Update position if no collision
            if not collision:
                self.humans[i, 0] = new_x
                self.humans[i, 1] = new_y
                if self.human_debug:
                    print(f"Human {i} moved to ({new_x:.2f}, {new_y:.2f}) with orientation {new_theta:.2f} radians.")
            else:
                if self.human_debug:
                    print(f"Human {i} remains at ({human_pos[0]:.2f}, {human_pos[1]:.2f}) due to collision.")

    def update_tables(self):
        """
        Placeholder for updating tables if dynamic behavior is required.
        Currently, tables are static.
        """
        pass  # No dynamic behavior for tables

    def is_on_table(self, position):
        """
        Checks if a given position is on top of any table.
        """
        for table in self.tables[:self.num_tables]:
            table_x, table_y, table_w, table_h = table
            # Include robot's radius in the collision check
            if (abs(position[0] - table_x) < (table_w / 2 + self.robot_radius + self.margin)) and \
               (abs(position[1] - table_y) < (table_h / 2 + self.robot_radius + self.margin)):
                return True
        return False

    def compute_reward_and_ticks(self):
        """
        Computes the reward for the current step and checks termination conditions.
        """
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(self.robot[0, :2] - self.goal[0, :2])

        # Calculate minimum distance to humans
        if self.num_humans > 0:
            min_distance_to_humans = np.min([
                np.linalg.norm(self.robot[0, :2] - human[:2]) - self.robot_radius - self.human_size
                for human in self.humans[:self.num_humans]
            ])
        else:
            min_distance_to_humans = self.discomfort_distance  # No humans, no discomfort

        # Check for collision
        collision = min_distance_to_humans < 0.0
        table_collision = self.is_on_table(self.robot[0, :2])

        reward = 0.0
        self.robot_is_done = False

        if abs(self.robot[0, 0]) > self.map_size / 2 or abs(self.robot[0, 1]) > self.map_size / 2:
            # Robot moved out of map
            self.robot_is_done = True
            reward = self.outofmap_reward  # -0.5
            if self.debug > 0:
                print("Robot moved out of bounds.")
        elif distance_to_goal < self.goal_threshold:
            # Robot reached the goal
            self.robot_is_done = True
            reward = self.reach_reward  # +1.0
            if self.debug > 0:
                print("Robot reached the goal.")
        elif collision or table_collision:
            # Collision with human or table
            self.robot_is_done = True
            reward = self.collision_reward  # -0.5
            if self.debug > 0:
                if collision:
                    print("Robot collided with a human.")
                if table_collision:
                    print("Robot collided with a table.")
        elif self.ticks > self.max_ticks:
            # Exceeded maximum steps
            self.robot_is_done = True
            reward = self.maxticks_reward  # -0.3
            if self.debug > 0:
                print("Exceeded maximum number of ticks.")
        else:
            # Alive penalty
            reward += self.alive_reward  # -0.003
            if self.debug > 0:
                print(f"Alive penalty applied. Current reward: {self.alive_reward}")

            # Discomfort penalty
            if min_distance_to_humans < self.discomfort_distance:
                discomfort = min_distance_to_humans - self.discomfort_distance  # Negative value
                discomfort_penalty = discomfort * self.discomfort_penalty_factor  # 0.005 * negative delta = negative penalty
                reward += discomfort_penalty  # Adding a negative penalty
                if self.debug > 0:
                    print(f"Discomfort penalty applied: {discomfort_penalty:.4f}")

            if self.debug > 0:
                print(f"Total reward for this step: {reward:.4f}")

        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        Resets the environment to an initial state.
        """
        if seed is not None:
            self.seed(seed)

        self.cumulative_reward = 0
        self.ticks = 0

        self.num_humans = random.randint(self.min_humans, self.max_humans)
        self.num_tables = random.randint(self.min_tables, self.max_tables)

        # Initialize humans and human_goals with fixed size (max_humans)
        self.humans = np.zeros((self.max_humans, 3))
        self.human_goals = np.zeros((self.max_humans, 2))
        self.tables = np.zeros((self.max_tables, 4))

        HALF_SIZE = self.map_size / 2.0 - self.margin

        # Initialize tables ensuring they are within the map boundaries
        for i in range(self.num_tables):
            while True:
                table_x = random.uniform(-HALF_SIZE + self.table_size[0] / 2, HALF_SIZE - self.table_size[0] / 2)
                table_y = random.uniform(-HALF_SIZE + self.table_size[1] / 2, HALF_SIZE - self.table_size[1] / 2)
                # Check overlap with other tables
                overlap = False
                for j in range(i):
                    distance_table = np.linalg.norm([table_x - self.tables[j, 0], table_y - self.tables[j, 1]])
                    if distance_table < (self.table_size[0] / 2 + self.margin + self.table_size[0] / 2):
                        overlap = True
                        if self.debug > 0:
                            print(f"Table {i} overlaps with Table {j}. Repositioning.")
                        break
                if overlap:
                    continue
                # If no overlaps, place the table
                self.tables[i] = [
                    table_x,
                    table_y,
                    self.table_size[0],
                    self.table_size[1],
                ]
                if self.debug > 0:
                    print(f"Placed Table {i} at ({table_x:.2f}, {table_y:.2f}) with size ({self.table_size[0]}, {self.table_size[1]})")
                break

        # Initialize robot position and orientation ensuring it's not on a table
        attempts = 0
        while True:
            self.robot = np.zeros((1, 3))
            self.robot[0] = [
                random.uniform(-HALF_SIZE, HALF_SIZE),
                random.uniform(-HALF_SIZE, HALF_SIZE),
                random.uniform(-np.pi, np.pi),
            ]
            if not self.is_on_table(self.robot[0, :2]):
                break
            attempts += 1
            if attempts > 100:
                raise Exception("Unable to place robot without overlapping a table after 100 attempts.")

        if self.debug > 0:
            print(f"Robot placed at ({self.robot[0,0]:.2f}, {self.robot[0,1]:.2f}) with orientation {self.robot[0,2]:.2f} radians.")

        # Initialize goal position ensuring it's not on a table
        attempts = 0
        while True:
            self.goal = np.zeros((1, 2))
            self.goal[0] = [
                random.uniform(-HALF_SIZE, HALF_SIZE),
                random.uniform(-HALF_SIZE, HALF_SIZE)
            ]
            if not self.is_on_table(self.goal[0]):
                break
            attempts += 1
            if attempts > 100:
                raise Exception("Unable to place goal without overlapping a table after 100 attempts.")

        if self.debug > 0:
            print(f"Goal placed at ({self.goal[0,0]:.2f}, {self.goal[0,1]:.2f})")

        # Initialize humans and their goals ensuring they're not on tables
        for i in range(self.num_humans):
            while True:
                human_pos = np.array([
                    random.uniform(-HALF_SIZE, HALF_SIZE),
                    random.uniform(-HALF_SIZE, HALF_SIZE),
                    random.uniform(-np.pi, np.pi),
                ])
                if not self.is_on_table(human_pos[:2]):
                    # Additionally, ensure humans are not overlapping with robot or goal
                    distance_to_robot = np.linalg.norm(human_pos[:2] - self.robot[0, :2])
                    distance_to_goal = np.linalg.norm(human_pos[:2] - self.goal[0, :2])
                    if distance_to_robot < (self.robot_radius + self.human_size + self.margin):
                        if self.debug > 0:
                            print(f"Human {i} too close to robot. Repositioning.")
                        continue
                    if distance_to_goal < (self.goal_radius + self.human_size + self.margin):
                        if self.debug > 0:
                            print(f"Human {i} too close to goal. Repositioning.")
                        continue
                    self.humans[i] = human_pos
                    break
            while True:
                human_goal = np.array([
                    random.uniform(-HALF_SIZE, HALF_SIZE),
                    random.uniform(-HALF_SIZE, HALF_SIZE)
                ])
                if not self.is_on_table(human_goal):
                    # Ensure goal is not overlapping with robot, other humans, or tables
                    distance_to_robot = np.linalg.norm(human_goal - self.robot[0, :2])
                    distance_to_current_human = np.linalg.norm(human_goal - self.humans[i, :2])
                    if distance_to_robot < (self.robot_radius + self.human_size + self.margin):
                        if self.debug > 0:
                            print(f"Human {i}'s goal too close to robot. Reassigning.")
                        continue
                    collision = False
                    for j in range(self.num_humans):
                        if j != i:
                            distance = np.linalg.norm(human_goal - self.humans[j, :2])
                            if distance < (self.human_threshold + self.human_size + self.margin):
                                collision = True
                                if self.debug > 0:
                                    print(f"Human {i}'s goal too close to Human {j}. Reassigning.")
                                break
                    if collision:
                        continue
                    self.human_goals[i] = human_goal
                    break

            if self.debug > 0:
                print(f"Human {i} placed at ({self.humans[i,0]:.2f}, {self.humans[i,1]:.2f}) with goal at ({self.human_goals[i,0]:.2f}, {self.human_goals[i,1]:.2f})")

        # Pad remaining humans (if any) with zeros or default values
        for i in range(self.num_humans, self.max_humans):
            self.humans[i] = [0.0, 0.0, 0.0]      # Example padding
            self.human_goals[i] = [0.0, 0.0]      # Example padding

        # Pad remaining tables (if any) with zeros or default values
        for i in range(self.num_tables, self.max_tables):
            self.tables[i] = [0.0, 0.0, 0.0, 0.0]  # Example padding

        self.robot_is_done = False

        if self.debug > 0:
            print("Environment reset complete.")
            print("Robot Position:", self.robot)
            print("Goal Position:", self.goal)
            print("Humans:", self.humans)
            print("Human Goals:", self.human_goals)
            print("Tables:", self.tables)

        # **Move the following print statement inside the debug block**
        if self.debug > 0:
            print(f"Reset method returning: Observation shape = {observation.shape}, Info = {info}")

        # Return the observation and an empty info dictionary
        observation, info = self.get_observation(), {}
        return observation, info

    def get_observation(self):
        """
        Generates the observation vector for the current state.
        """
        if self.robot_view:
            # Robot's orientation (cos(theta), sin(theta))
            robot_obs = [
                np.cos(self.robot[0, 2]),
                np.sin(self.robot[0, 2])
            ]
        else:
            # Robot's absolute position and orientation
            robot_obs = [
                self.robot[0, 0],
                self.robot[0, 1],
                np.cos(self.robot[0, 2]),
                np.sin(self.robot[0, 2])
            ]

        # Goal observation (relative or absolute)
        if self.robot_view:
            dx = self.goal[0, 0] - self.robot[0, 0]
            dy = self.goal[0, 1] - self.robot[0, 1]
            robot_theta = self.robot[0, 2]
            relative_x = np.cos(-robot_theta) * dx - np.sin(-robot_theta) * dy
            relative_y = np.sin(-robot_theta) * dx + np.cos(-robot_theta) * dy
            goal_obs = [relative_x, relative_y]
        else:
            goal_obs = [self.goal[0, 0], self.goal[0, 1]]

        # Humans observation and mask
        human_obs = []
        human_mask = []
        for i in range(self.max_humans):
            if i < self.num_humans:
                human = self.humans[i]
                human_mask.append(1.0)  # Active human
                if self.robot_view:
                    dx = human[0] - self.robot[0, 0]
                    dy = human[1] - self.robot[0, 1]
                    robot_theta = self.robot[0, 2]
                    relative_x = np.cos(-robot_theta) * dx - np.sin(-robot_theta) * dy
                    relative_y = np.sin(-robot_theta) * dx + np.cos(-robot_theta) * dy
                    relative_orientation = (human[2] - self.robot[0, 2] + np.pi) % (2 * np.pi) - np.pi
                    human_obs.extend([
                        relative_x,
                        relative_y,
                        np.cos(relative_orientation),
                        np.sin(relative_orientation)
                    ])
                else:
                    human_obs.extend([
                        human[0],
                        human[1],
                        np.cos(human[2]),
                        np.sin(human[2])
                    ])
            else:
                # Padding for missing humans
                human_mask.append(0.0)  # Inactive slot
                if self.robot_view:
                    human_obs.extend([0.0, 0.0, 0.0, 0.0])  # Example padding values
                else:
                    human_obs.extend([0.0, 0.0, 0.0, 0.0])  # Example padding values

        # Tables observation and mask
        table_obs = []
        table_mask = []
        for i in range(self.max_tables):
            if i < self.num_tables:
                table = self.tables[i]
                table_mask.append(1.0)  # Active table
                if self.robot_view:
                    dx = table[0] - self.robot[0, 0]
                    dy = table[1] - self.robot[0, 1]
                    robot_theta = self.robot[0, 2]
                    relative_x = np.cos(-robot_theta) * dx - np.sin(-robot_theta) * dy
                    relative_y = np.sin(-robot_theta) * dx + np.cos(-robot_theta) * dy
                    table_obs.extend([
                        relative_x,
                        relative_y,
                        table[2],
                        table[3]
                    ])
                else:
                    table_obs.extend([
                        table[0],
                        table[1],
                        table[2],
                        table[3]
                    ])
            else:
                # Padding for missing tables
                table_mask.append(0.0)  # Inactive slot
                table_obs.extend([0.0, 0.0, 0.0, 0.0])  # Example padding values

        # Combine all components
        observation = np.array(
            robot_obs + goal_obs + human_obs + table_obs + human_mask + table_mask,
            dtype=np.float32
        )

        return observation

    def render(self):
        """
        Renders the environment based on the render_mode.
        - 'human': Displays the environment using cv2.imshow.
        - 'rgb_array': Returns a NumPy array of the current frame.
        """
        def w2px(i): return int(self.pixel_to_world * (i + self.map_size / 2))
        def w2py(i): return int(self.pixel_to_world * (self.map_size / 2 - i))

        def draw_oriented_point(image, data, color, radius=0.15, nose=0.1):
            """
            Draws an oriented point (e.g., robot or human) on the image.
            """
            centre = np.array([data[0], data[1]])
            left = centre + radius * np.array([np.cos(data[2] + np.pi / 2), np.sin(data[2] + np.pi / 2)])
            right = centre + radius * np.array([np.cos(data[2] - np.pi / 2), np.sin(data[2] - np.pi / 2)])
            front = centre + nose * np.array([np.cos(data[2]), np.sin(data[2])])
            cv2.line(image, (w2px(centre[0]), w2py(centre[1])), (w2px(left[0]), w2py(left[1])), color, 3)
            cv2.line(image, (w2px(centre[0]), w2py(centre[1])), (w2px(right[0]), w2py(right[1])), color, 3)
            cv2.line(image, (w2px(centre[0]), w2py(centre[1])), (w2px(front[0]), w2py(front[1])), color, 5)

        # Initialize rendering if not already done
        if not self.window_initialised:
            if self.render_mode == 'human':
                cv2.namedWindow("world", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("world", int(self.resolution_view), int(self.resolution_view))
            self.world_image = (np.ones((int(self.resolution), int(self.resolution), 3)) * 255).astype(np.uint8)
            self.window_initialised = True

        # Fill the world background with white
        self.world_image.fill(255)

        # Draw the black walls (borders)
        thickness = 10
        cv2.rectangle(
            self.world_image,
            (0, 0),
            (self.world_image.shape[1], self.world_image.shape[0]),
            (0, 0, 0),
            thickness
        )

        # Draw the tables
        for i in range(self.num_tables):
            table = self.tables[i]
            top_left = (w2px(table[0] - table[2] / 2), w2py(table[1] + table[3] / 2))
            bottom_right = (w2px(table[0] + table[2] / 2), w2py(table[1] - table[3] / 2))
            cv2.rectangle(self.world_image, top_left, bottom_right, (42, 42, 165), -1)  # Filled brown
            cv2.rectangle(self.world_image, top_left, bottom_right, (0, 0, 0), 2)  # Outline

        # Draw the robot
        draw_oriented_point(self.world_image, self.robot[0], (255, 0, 0), radius=self.robot_size)

        # Draw the robot's goal
        cv2.circle(
            self.world_image,
            (w2px(self.goal[0, 0]), w2py(self.goal[0, 1])),
            int(self.goal_size * self.pixel_to_world),
            (0, 255, 0),
            -1  # Filled circle
        )

        # Draw humans and their goals
        for i in range(self.num_humans):
            draw_oriented_point(self.world_image, self.humans[i], (0, 0, 255), radius=self.human_size)
            cv2.circle(
                self.world_image,
                (w2px(self.human_goals[i, 0]), w2py(self.human_goals[i, 1])),
                int(self.goal_size * self.pixel_to_world),
                (255, 0, 0),
                -1  # Filled circle
            )

        # Handle rendering based on render_mode
        if self.render_mode == 'human':
            cv2.imshow("world", self.world_image)
            if cv2.waitKey(self.milliseconds) & 0xFF == 27:  # Exit on ESC key
                sys.exit(0)
        elif self.render_mode == 'rgb_array':
            # Return the rendered image as an array
            return self.world_image.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def close(self):
        """
        Closes the rendering window if it was initialized.
        """
        if self.window_initialised:
            cv2.destroyAllWindows()

    def compute_reward_and_ticks(self):
        """
        Computes the reward for the current step and checks termination conditions.
        """
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(self.robot[0, :2] - self.goal[0, :2])

        # Calculate minimum distance to humans
        if self.num_humans > 0:
            min_distance_to_humans = np.min([
                np.linalg.norm(self.robot[0, :2] - human[:2]) - self.robot_radius - self.human_size
                for human in self.humans[:self.num_humans]
            ])
        else:
            min_distance_to_humans = self.discomfort_distance  # No humans, no discomfort

        # Check for collision
        collision = min_distance_to_humans < 0.0
        table_collision = self.is_on_table(self.robot[0, :2])

        reward = 0.0
        self.robot_is_done = False

        if abs(self.robot[0, 0]) > self.map_size / 2 or abs(self.robot[0, 1]) > self.map_size / 2:
            # Robot moved out of map
            self.robot_is_done = True
            reward = self.outofmap_reward  # -0.5
            if self.debug > 0:
                print("Robot moved out of bounds.")
        elif distance_to_goal < self.goal_threshold:
            # Robot reached the goal
            self.robot_is_done = True
            reward = self.reach_reward  # +1.0
            if self.debug > 0:
                print("Robot reached the goal.")
        elif collision or table_collision:
            # Collision with human or table
            self.robot_is_done = True
            reward = self.collision_reward  # -0.5
            if self.debug > 0:
                if collision:
                    print("Robot collided with a human.")
                if table_collision:
                    print("Robot collided with a table.")
        elif self.ticks > self.max_ticks:
            # Exceeded maximum steps
            self.robot_is_done = True
            reward = self.maxticks_reward  # -0.3
            if self.debug > 0:
                print("Exceeded maximum number of ticks.")
        else:
            # Alive penalty
            reward += self.alive_reward  # -0.003
            if self.debug > 0:
                print(f"Alive penalty applied. Current reward: {self.alive_reward}")

            # Discomfort penalty
            if min_distance_to_humans < self.discomfort_distance:
                discomfort = min_distance_to_humans - self.discomfort_distance  # Negative value
                discomfort_penalty = discomfort * self.discomfort_penalty_factor  # 0.005 * negative delta = negative penalty
                reward += discomfort_penalty  # Adding a negative penalty
                if self.debug > 0:
                    print(f"Discomfort penalty applied: {discomfort_penalty:.4f}")

            if self.debug > 0:
                print(f"Total reward for this step: {reward:.4f}")

        return reward

    def _normalize_angle(self, angle):
        """
        Normalizes the angle to be within [-pi, pi].
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi


class DiscreteLiteSocNavGym(LiteSocNavGym):
    """
    A subclass of LiteSocNavGym with a discrete action space.
    """
    def __init__(self, advance_split=5, rotation_split=5, **kwargs):
        super().__init__(**kwargs)
        self.advance_split = advance_split
        self.rotation_split = rotation_split
        self.action_map = self.build_action_map(advance_split, rotation_split)

    def build_action_map(self, advance_split, rotation_split):
        """
        Builds a mapping from discrete action indices to continuous action values.
        """
        advance_grid, rotation_grid = np.meshgrid(
            np.linspace(-1, 1, advance_split), np.linspace(-1, 1, rotation_split)
        )
        return np.column_stack((advance_grid.flatten(), rotation_grid.flatten()))

    @property
    def action_space(self):
        """
        Defines the discrete action space based on the splits.
        """
        return spaces.Discrete(self.advance_split * self.rotation_split)

    def step(self, action_idx):
        """
        Executes one time step within the environment using a discrete action.
        """
        if action_idx < 0 or action_idx >= self.advance_split * self.rotation_split:
            raise ValueError(f"Invalid action index: {action_idx}")
        action = self.action_map[action_idx]
        return super().step(action)
