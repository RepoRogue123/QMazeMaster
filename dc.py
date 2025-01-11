import tkinter as tk
import random
import time
import common_functions
from common_classes import Cell
from common_classes import Position

class GridWord(object):
    """
    This class represents the gridworld.
    """

    def __init__(self, name, height, width, r_nt=0):
        self.name = name
        self.episode = 1
        self.step = 1
        self.height = height
        self.width = width
        self.rewards_for_step = []
        self.rewards_for_episode = []
        self.step_for_episode = []
        self.current_position1 = Position(0, 0)  # Player 1
        self.current_position2 = Position(0, 0)  # Player 2, starting at a different position
        self.world = []
        self.killzones = []  # Store killzone positions
        for col in range(width):
            tmp = []
            for row in range(height):
                tmp.append(Cell(reward=r_nt, col=col, row=row))
            self.world.append(tmp)

    def place_killzones(self, num_killzones=3):
        """
        Place a specified number of killzones at random positions on the grid.
        """
        self.killzones.append((3, 7))
        self.killzones.append((6, 5))
        self.killzones.append((4, 2))
        self.world[3][7].killzone = True  # Mark this cell as a killzone
        self.world[6][5].killzone = True  # Mark this cell as a killzone
        self.world[4][2].killzone = True  # Mark this cell as a killzone

    def get_max_q(self, current_state, value_type, player=1):
        """
        Return the maximum value q for the state s, considering killzones.
        Args:
            current_state: actual state in the world
            value_type: VALUE || ACTION. With value on will get the value of q(a).
            Otherwise it will get the Action corresponding to the maximum value of q(a)
        Returns:
        """
        max_value = None
        max_q = None
        potential_actions = []

        for possible_action in [*current_state.q_a]:
            # Check if the action leads to a killzone
            next_position = self.get_next_state(possible_action, player)
            if (next_position[0], next_position[1]) in self.killzones:
                potential_actions.append((possible_action, current_state.q_a[possible_action] - 1))  # Penalize for killzone
            else:
                potential_actions.append((possible_action, current_state.q_a[possible_action]))

        # Determine the best action based on the adjusted Q-values
        for action, value in potential_actions:
            if max_value is None or value > max_value:
                max_value = value
                max_q = action

        if value_type == 'action':
            return max_q
        else:
            return max_value

    def set_terminal_state(self, row: int, col: int, reward: float) -> None:
        """
        This method is used to set terminal states inside the GridWorld.
        Args:
            row: Row of the terminal state
            col: Column of the terminal state
            reward: Reward getting arriving in that terminal state
        """
        self.world[row][col].reward = reward
        self.world[row][col].terminal = True
        self.world[row][col].wall = False

    def get_current_state(self, player=1):
        """
        Get the current state in world considering the current position.
        Returns: Current state
        """
        pos = self.current_position1 if player == 1 else self.current_position2
        return self.world[pos.col][pos.row]

    def set_wall(self, walls: list) -> None:
        """
        Method used to set the walls inside the gridworld.
        Args:
            walls: List containing positions (col,row)
        """
        for wall in walls:
            self.world[wall[0]][wall[1]].wall = True

    def action_e_greedy(self, current_state, epsilon, policy=None, player=1) -> str:
        """
        This method selects the next action following the E-greedy paradigm
        Args:
            current_state: The current state in the grid world
            epsilon: Epsilon to use in the e-greedy function
            policy: List of for integers (up, down, left, right). This parameter has been
        Returns: Action to take
        """
        epsilon = epsilon * 100
        q_current_state = self.world[self.current_position1.col][self.current_position1.row].q_a if player == 1 else self.world[self.current_position2.col][self.current_position2.row].q_a
        possible_action = [*q_current_state]

        
        if policy is not None:
            return random.choices(possible_action, weights=[policy[0], policy[1], policy[2], policy[3]], k=1)[0]

        value = random.choices(['random', 'greedy'], weights=[epsilon, 100 - epsilon], k=1)

        # Choose greedy between the possible actions
        if 'greedy' in value:
            return self.get_max_q(current_state=current_state, value_type='action', player=player)
        else:
            return random.choice(possible_action)

    def get_next_state(self, action, player=1):
        """
        This method returns the next position of the agent given an action to take
        Args:
            action: Action to take
            player: Player identifier (1 or 2)
        Returns: Position of the next state
        """
        col = self.current_position1.col if player == 1 else self.current_position2.col
        row = self.current_position1.row if player == 1 else self.current_position2.row

        if action == Cell.Action.DOWN.value:
            col += 1
        elif action == Cell.Action.UP.value:
            col -= 1
        elif action == Cell.Action.RIGHT.value:
            row += 1
        elif action == Cell.Action.LEFT.value:
            row -= 1

        # Walls or out of the world
        if (col < 0 or col > self.height - 1) or (row < 0 or row > self.width - 1) or self.world[col][row].wall:
            return [self.current_position1.col, self.current_position1.row] if player == 1 else [self.current_position2.col, self.current_position2.row]
        return [col, row]

    def random_position(self):
        """
        This method returns a random position that isn't neither a wall or a terminal state
        Returns: column, row of the random position
        """
        found_position = False
        while not found_position:
            col = random.randint(0, self.height - 1)
            row = random.randint(0, self.width - 1)
            if(col==8 and row==8):
                col = random.randint(0, self.height - 1)
                row = random.randint(0, self.width - 1)
            if not self.world[col][row].wall and not self.world[col][row].terminal and (col,row) not in self.killzones and (col,row) != (8,8):
                found_position = True
        return col, row

    def update_q_value(self, s, s_first, action, action_first, alpha, discount_factor):
        """
        Function to update the value of q(a)
        Args:
            s: State S
            s_first:  State S'
            action: Action A
            alpha: Learning rate
            discount_factor: Discount factor
        """
        if 'Q-Learning' in self.name:
            s.q_a[action] = s.q_a[action] + alpha * (
                    s_first.reward + discount_factor * (self.get_max_q(current_state=s_first, value_type='value')) -
                    s.q_a[action])

        self.rewards_for_step.append(s_first.reward)
        self.step += 1

    def restart_episode(self, random_start):
        """
        This method restarts the episode in position (0,0) and all the counters.
        random_start: True if it needed a random start
        """
        if random_start:
            self.current_position1.col, self.current_position1.row = self.random_position()
            self.current_position2.col, self.current_position2.row = self.random_position()
            while self.current_position1.col==self.current_position2.col and self.current_position1.row==self.current_position2.row:
                self.current_position2.col, self.current_position2.row = self.random_position()

        else:
            self.current_position1.col = 0
            self.current_position1.row = 0
            self.current_position2.col = 8
            self.current_position2.row = 0

        sum_reward = sum(self.rewards_for_step)
        self.rewards_for_episode.append(sum_reward)
        self.step_for_episode.append(self.step)
        self.rewards_for_step = []
        self.step = 0
        print('Episode: ', self.episode, ' Cumulative reward of: ', sum_reward)
        self.episode += 1

    def q_learning_algorithm(self, n_episode, epsilon, alpha, discount_factor, random_start):
        """
        Q-learning algorithm to find the optimal policy
        Args:
            n_episode: Number of episodes
            epsilon: Epsilon to use in e-greedy method
            alpha: Learning rate
            discount_factor: Discount factor gamma
        """
        print('START Q-LEARNING METHOD')
        s1 = self.get_current_state(player=1)
        s2 = self.get_current_state(player=2)
        while self.episode <= n_episode:
            action1 = self.action_e_greedy(current_state=s1, epsilon=epsilon, player=1)
            self.current_position1.col, self.current_position1.row = self.get_next_state(action1, player=1)
            s1_first = self.get_current_state(player=1)
            self.update_q_value(s=s1, s_first=s1_first, action=action1, action_first=None, alpha=alpha,
                                discount_factor=discount_factor)
            s1 = s1_first

            action2 = self.action_e_greedy(current_state=s2, epsilon=epsilon, player=2)
            self.current_position2.col, self.current_position2.row = self.get_next_state(action2, player=2)
            s2_first = self.get_current_state(player=2)
            self.update_q_value(s=s2, s_first=s2_first, action=action2, action_first=None, alpha=alpha,
                                discount_factor=discount_factor)
            s2 = s2_first

            if s1.terminal or s2.terminal:
                self.restart_episode(random_start)
                s1 = self.get_current_state(player=1)
                s2 = self.get_current_state(player=2)

                # Epsilon decrement to find the optimal policy
                epsilon = 1 - (self.episode / n_episode) ** 10

    def run_sample_episodes(self, num_episodes):
        """
        Run sample episodes to see the outcome of learning.
        Args:
            num_episodes: Number of episodes to run
        """
        for episode in range(num_episodes):
            # Randomly initialize the positions
            self.current_position1.col, self.current_position1.row = self.random_position()
            self.current_position2.col, self.current_position2.row = self.random_position()
            while self.current_position1.col==self.current_position2.col and self.current_position1.row==self.current_position2.row:
                self.current_position2.col, self.current_position2.row = self.random_position()
            

            current_state1 = self.get_current_state(player=1)
            current_state2 = self.get_current_state(player=2)
            total_reward1 = 0
            total_reward2 = 0
            path1 = []
            path2 = []



            while not current_state1.terminal and not current_state2.terminal:
                path1.append((self.current_position1.col, self.current_position1.row))
                path2.append((self.current_position2.col, self.current_position2.row))
                action1 = self.get_max_q(current_state=current_state1, value_type='action', player=1)
                action2 = self.get_max_q(current_state=current_state2, value_type='action', player=2)
                self.current_position1.col, self.current_position1.row = self.get_next_state(action1, player=1)
                self.current_position2.col, self.current_position2.row = self.get_next_state(action2, player=2)
                current_state1 = self.get_current_state(player=1)
                current_state2 = self.get_current_state(player=2)
                total_reward1 += current_state1.reward
                total_reward2 += current_state2.reward

                

            # Add the terminal state to the paths
            path1.append((self.current_position1.col, self.current_position1.row))
            path2.append((self.current_position2.col, self.current_position2.row))
            print(f"Episode {episode + 1}: Player 1 Path taken: {path1}, Total reward: {total_reward1}")
            print(f"Episode {episode + 1}: Player 2 Path taken: {path2}, Total reward: {total_reward2}")

    

class GridWorldGUI:
    def __init__(self, gridworld):
        self.gridworld = gridworld
        self.master = tk.Tk()
        self.master.title(f"{gridworld.name} Grid World")
        
        # Create canvas
        self.canvas_width = 500
        self.canvas_height = 500
        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        # Cell size
        self.cell_size = 50

        # Draw initial grid
        self.draw_grid()

        # Add buttons
        self.start_q_learning_button = tk.Button(self.master, text="Start Q-Learning", command=self.start_q_learning)
        self.start_q_learning_button.pack()

        self.run_sample_button = tk.Button(self.master, text="Run Sample Episode", command=self.run_sample_episode_gui)
        self.run_sample_button.pack()

    def draw_grid(self):
        # Clear canvas
        self.canvas.delete("all")

        # Draw grid cells
        for col in range(len(self.gridworld.world)):
            for row in range(len(self.gridworld.world[col])):
                x1 = row * self.cell_size
                y1 = col * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                # Determine cell color
                fill_color = "white"
                if self.gridworld.world[col][row].wall:
                    fill_color = "black"
                elif (col, row) in self.gridworld.killzones:
                    fill_color = "red"
                elif self.gridworld.world[col][row].terminal:
                    fill_color = "green"

                # Draw cell
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="gray")

        # Draw players
        p1_x = self.gridworld.current_position1.row * self.cell_size + self.cell_size // 2
        p1_y = self.gridworld.current_position1.col * self.cell_size + self.cell_size // 2
        p2_x = self.gridworld.current_position2.row * self.cell_size + self.cell_size // 2
        p2_y = self.gridworld.current_position2.col * self.cell_size + self.cell_size // 2

        # Player 1 (blue)
        self.canvas.create_oval(
            p1_x - 15, p1_y - 15, 
            p1_x + 15, p1_y + 15, 
            fill="blue", outline="darkblue"
        )

        # Player 2 (orange)
        self.canvas.create_oval(
            p2_x - 15, p2_y - 15, 
            p2_x + 15, p2_y + 15, 
            fill="orange", outline="darkorange"
        )

    def start_q_learning(self):
        # Start the Q-learning algorithm in a separate thread to avoid blocking the GUI
        import threading
        threading.Thread(target=self.run_q_learning).start()

    def run_q_learning(self):
        # Run the Q-learning algorithm
        self.gridworld.q_learning_algorithm(n_episode=1000, alpha=1, epsilon=1, discount_factor=1, random_start=False)
        print("Q-Learning completed.")

    def run_sample_episode_gui(self):
        # Reset grid
        self.gridworld.current_position1.col, self.gridworld.current_position1.row = self.gridworld.random_position()
        self.gridworld.current_position2.col, self.gridworld.current_position2.row = self.gridworld.random_position()
        while self.gridworld.current_position2.col==self.gridworld.current_position1.col and self.gridworld.current_position2.row==self.gridworld.current_position1.row:
                self.gridworld.current_position2.col, self.gridworld.current_position2.row = self.gridworld.random_position()

        current_state1 = self.gridworld.get_current_state(player=1)
        current_state2 = self.gridworld.get_current_state(player=2)

        def move_players():
            nonlocal current_state1, current_state2
            
            if not current_state1.terminal and not current_state2.terminal:
                # Get and execute actions
                action1 = self.gridworld.get_max_q(current_state=current_state1, value_type='action', player=1)
                action2 = self.gridworld.get_max_q(current_state=current_state2, value_type ='action', player=2)
                
                self.gridworld.current_position1.col, self.gridworld.current_position1.row = self.gridworld.get_next_state(action1, player=1)
                self.gridworld.current_position2.col, self.gridworld.current_position2.row = self.gridworld.get_next_state(action2, player=2)
                
                current_state1 = self.gridworld.get_current_state(player=1)
                current_state2 = self.gridworld.get_current_state(player=2)

                # Redraw grid
                self.draw_grid()

                # Schedule next move
                self.master.after(500, move_players)

        # Start moving players for one episode
        move_players()

def main():
    height = 9
    width = 9
    walls = [
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
        [2, 6], [3, 6], [4, 6], [5, 6], [2, 6],
        [7, 1], [7, 2], [7, 3], [7, 4]]

    # Q-Learning
    q_learning_world = GridWord(name='Q-Learning', height=height, width=width, r_nt=-1)
    q_learning_world.set_terminal_state(row=8, col=8, reward=50)

    # Set walls
    q_learning_world.set_wall(walls=walls)

    # Place killzones
    q_learning_world.place_killzones(num_killzones=3)

    # Create GUI
    gui = GridWorldGUI(q_learning_world)
    gui.master.mainloop()

if __name__ == '__main__':
    main()