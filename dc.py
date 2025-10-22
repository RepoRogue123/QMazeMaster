import tkinter as tk
from tkinter import messagebox
import random
import time
import math
import json
from enum import Enum
from collections import defaultdict
import common_functions
from common_classes import Cell
from common_classes import Position
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# ============================================================================
# Graph Generation and Data Saving Functions
# ============================================================================

def ensure_results_dirs():
    """Ensure results directories exist."""
    os.makedirs('results/graphs', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)

def generate_graphs(metrics, algorithm_name, grid_size, episode_count):
    """Generate and save training graphs."""
    ensure_results_dirs()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_dir = f'results/graphs/{algorithm_name}_{grid_size}x{grid_size}_{timestamp}'
    os.makedirs(graph_dir, exist_ok=True)
    
    # Extract data
    episodes = list(range(len(metrics['rewards'])))
    rewards = metrics['rewards']
    epsilons = metrics['epsilons']
    steps = metrics['steps']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Results - {algorithm_name} on {grid_size}x{grid_size} Grid', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Rewards over episodes
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, color='#1f77b4', linewidth=1, alpha=0.7, label='Episode Reward')
    # Add moving average
    if len(rewards) > 50:
        moving_avg = [sum(rewards[max(0, i-50):i+1])/min(i+1, 51) for i in range(len(rewards))]
        ax1.plot(episodes, moving_avg, color='#ff7f0e', linewidth=2, label='Moving Avg (50 eps)')
    ax1.set_xlabel('Episode', fontweight='bold')
    ax1.set_ylabel('Total Reward', fontweight='bold')
    ax1.set_title('Reward Progression', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Epsilon decay
    ax2 = axes[0, 1]
    ax2.plot(episodes, epsilons, color='#d62728', linewidth=2)
    ax2.set_xlabel('Episode', fontweight='bold')
    ax2.set_ylabel('Epsilon (ε)', fontweight='bold')
    ax2.set_title('Exploration Rate Decay', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(episodes, epsilons, alpha=0.3, color='#d62728')
    
    # Plot 3: Steps per episode
    ax3 = axes[1, 0]
    ax3.bar(episodes, steps, color='#2ca02c', alpha=0.7, width=1)
    ax3.set_xlabel('Episode', fontweight='bold')
    ax3.set_ylabel('Steps', fontweight='bold')
    ax3.set_title('Episode Length', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative reward
    ax4 = axes[1, 1]
    cumulative = [sum(rewards[:i+1]) for i in range(len(rewards))]
    ax4.plot(episodes, cumulative, color='#9467bd', linewidth=2)
    ax4.fill_between(episodes, cumulative, alpha=0.3, color='#9467bd')
    ax4.set_xlabel('Episode', fontweight='bold')
    ax4.set_ylabel('Cumulative Reward', fontweight='bold')
    ax4.set_title('Total Cumulative Reward', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    graph_file = f'{graph_dir}/training_metrics.png'
    plt.savefig(graph_file, dpi=300, bbox_inches='tight')
    print(f'[GRAPHS] Saved training metrics graph: {graph_file}')
    plt.close()
    
    return graph_dir

def save_training_data(metrics, algorithm_name, grid_size, alpha, gamma, epsilon_0, episode_count):
    """Save training data to JSON file."""
    ensure_results_dirs()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = f'results/data/{algorithm_name}_{grid_size}x{grid_size}_{timestamp}.json'
    
    summary = {
        'timestamp': timestamp,
        'algorithm': algorithm_name,
        'grid_size': grid_size,
        'episodes': episode_count,
        'hyperparameters': {
            'learning_rate': alpha,
            'discount_factor': gamma,
            'initial_epsilon': epsilon_0
        },
        'metrics': {
            'total_episodes': len(metrics['rewards']),
            'final_reward': metrics['rewards'][-1] if metrics['rewards'] else 0,
            'average_reward': sum(metrics['rewards']) / len(metrics['rewards']) if metrics['rewards'] else 0,
            'max_reward': max(metrics['rewards']) if metrics['rewards'] else 0,
            'min_reward': min(metrics['rewards']) if metrics['rewards'] else 0,
            'average_episode_length': sum(metrics['steps']) / len(metrics['steps']) if metrics['steps'] else 0,
            'total_steps': sum(metrics['steps'])
        },
        'episode_data': {
            'rewards': metrics['rewards'],
            'epsilons': metrics['epsilons'],
            'steps': metrics['steps']
        }
    }
    
    with open(data_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'[DATA] Saved training data: {data_file}')
    return data_file

class AgentStrategy(Enum):
    """Different learning strategies for agents."""
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"  # On-policy learning
    EXPECTED_SARSA = "expected_sarsa"  # Conservative update
    DOUBLE_Q = "double_q_learning"  # Reduced overestimation

class AgentMemory:
    """Memory system for agents to store and recall experiences."""
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.memory = []
        self.visit_count = defaultdict(int)  # Track state visits
        
    def store_experience(self, state, action, reward, next_state, terminal):
        """Store experience in memory."""
        if len(self.memory) >= self.max_size:
            self.memory.pop(0)
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'terminal': terminal
        })
        
    def get_experiences(self, batch_size=32):
        """Get random batch of experiences (for replay)."""
        if len(self.memory) < batch_size:
            return self.memory
        return random.sample(self.memory, batch_size)
    
    def track_visit(self, state_hash):
        """Track state visitation frequency."""
        self.visit_count[state_hash] += 1
    
    def get_visit_frequency(self, state_hash):
        """Get how many times a state has been visited."""
        return self.visit_count.get(state_hash, 0)

class RewardShaper:
    """Advanced reward shaping for agents."""
    def __init__(self, base_reward=-0.1, killzone_penalty=-10, terminal_reward=100):
        self.base_reward = base_reward
        self.killzone_penalty = killzone_penalty
        self.terminal_reward = terminal_reward
        self.distance_reward_factor = 0.5  # Bonus for getting closer to goal
        
    def shape_reward(self, reward, current_pos, next_pos, goal_pos, is_killzone=False, is_terminal=False):
        """Apply reward shaping to encourage good behavior."""
        shaped_reward = reward
        
        if is_terminal:
            shaped_reward = self.terminal_reward
        elif is_killzone:
            shaped_reward += self.killzone_penalty
        else:
            # Encourage movement toward goal
            curr_dist = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
            next_dist = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])
            
            if next_dist < curr_dist:
                shaped_reward += self.distance_reward_factor
        
        return shaped_reward

class PerformanceMonitor:
    """Monitor and analyze agent performance metrics."""
    def __init__(self):
        self.episode_metrics = []
        self.window_size = 100
        
    def record_episode(self, episode_num, reward, steps, epsilon, exploration_count):
        """Record episode metrics."""
        self.episode_metrics.append({
            'episode': episode_num,
            'reward': reward,
            'steps': steps,
            'epsilon': epsilon,
            'exploration': exploration_count
        })
    
    def get_moving_average(self, metric_name, window=None):
        """Get moving average of a metric."""
        if window is None:
            window = self.window_size
        
        if len(self.episode_metrics) == 0:
            return 0
        
        recent = self.episode_metrics[-window:]
        values = [m[metric_name] for m in recent]
        return sum(values) / len(values) if values else 0
    
    def get_convergence_rate(self):
        """Calculate how quickly the agent is converging."""
        if len(self.episode_metrics) < 2:
            return 0
        
        rewards = [m['reward'] for m in self.episode_metrics[-100:]]
        if len(rewards) < 2:
            return 0
        
        # Compare recent to earlier performance
        first_half = sum(rewards[:len(rewards)//2]) / (len(rewards)//2)
        second_half = sum(rewards[len(rewards)//2:]) / (len(rewards) - len(rewards)//2)
        
        return (second_half - first_half) / abs(first_half) if first_half != 0 else 0
    
    def export_metrics(self, filename):
        """Export metrics to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.episode_metrics, f, indent=2)

class GridWord(object):
    """
    This class represents the gridworld with advanced reinforcement learning features.
    """

    def __init__(self, name, height, width, r_nt=0, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.name = name
        self.episode = 1
        self.step = 1
        self.height = height
        self.width = width
        
        # RL Parameters (configurable)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.initial_epsilon = epsilon
        
        # Advanced RL Features
        self.strategy = AgentStrategy.Q_LEARNING  # Learning strategy
        self.memory1 = AgentMemory(max_size=500)  # Experience replay for P1
        self.memory2 = AgentMemory(max_size=500)  # Experience replay for P2
        self.reward_shaper = RewardShaper()  # Advanced reward shaping
        self.performance_monitor = PerformanceMonitor()  # Performance tracking
        
        # Double Q-Learning support
        self.q_values_primary = {}  # Primary Q-table
        self.q_values_secondary = {}  # Secondary Q-table for Double Q-Learning
        
        self.rewards_for_step = []
        self.rewards_for_episode = []
        self.step_for_episode = []
        self.q_value_history = []  # Track Q-value changes over time
        self.exploration_stats = {'exploration': 0, 'exploitation': 0}  # Track exploration vs exploitation
        
        # Graph metrics tracking
        self.graph_metrics = {
            'rewards': [],
            'epsilons': [],
            'steps': []
        }
        
        self.current_position1 = Position(0, 0)  # Player 1
        self.current_position2 = Position(0, 0)  # Player 2
        self.world = []
        self.killzones = []  # Store killzone positions
        self.walls = []  # Store wall positions
        
        for col in range(width):
            tmp = []
            for row in range(height):
                tmp.append(Cell(reward=r_nt, col=col, row=row))
            self.world.append(tmp)

    def place_killzones(self, num_killzones=None):
        """
        Dynamically place killzones at random positions on the grid.
        If num_killzones is None, it's calculated based on grid size.
        Killzones are placed randomly with no hardcoding.
        """
        # Calculate optimal number of killzones based on grid size if not provided
        if num_killzones is None:
            grid_area = self.height * self.width
            # Dynamic calculation: roughly 10-20% of grid area but at least 1
            num_killzones = max(1, min(5, (grid_area // 20)))
        
        killzone_positions = []
        attempts = 0
        max_attempts = 500  # Prevent infinite loops
        
        # Try to place killzones randomly
        while len(killzone_positions) < num_killzones and attempts < max_attempts:
            col = random.randint(0, self.width - 1)
            row = random.randint(0, self.height - 1)
            
            # Avoid terminal state, starting positions, walls, and duplicates
            if ((col, row) not in killzone_positions and 
                (col, row) != (0, 0) and 
                (col, row) != (self.width - 1, self.height - 1) and
                not self.world[col][row].wall and
                not self.world[col][row].terminal):
                killzone_positions.append((col, row))
            
            attempts += 1
        
        # Add killzones to the world
        for pos in killzone_positions:
            if pos[0] < self.width and pos[1] < self.height:
                self.killzones.append(pos)
                self.world[pos[0]][pos[1]].killzone = True
        
        print(f"[INFO] Placed {len(self.killzones)} killzones dynamically")

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

    def set_wall(self, walls: list = None) -> None:
        """
        Method used to set the walls inside the gridworld dynamically.
        If walls is None, walls are generated randomly based on grid size.
        Args:
            walls: List containing positions (col,row). If None, walls are generated randomly.
        """
        if walls is None:
            # Dynamic wall generation - roughly 10-15% of grid cells
            grid_area = self.height * self.width
            num_walls = max(0, int(grid_area * 0.12))
            
            walls = []
            attempts = 0
            max_attempts = 500
            
            while len(walls) < num_walls and attempts < max_attempts:
                col = random.randint(0, self.width - 1)
                row = random.randint(0, self.height - 1)
                
                # Avoid terminal and starting positions
                if ((col, row) not in walls and
                    (col, row) != (0, 0) and
                    (col, row) != (self.width - 1, self.height - 1)):
                    walls.append((col, row))
                
                attempts += 1
            
            print(f"[INFO] Generated {len(walls)} walls dynamically")
        
        # Set walls in the world
        for wall in walls:
            if wall[0] < self.width and wall[1] < self.height:
                self.world[wall[0]][wall[1]].wall = True
                self.walls.append(wall)

    def action_e_greedy(self, current_state, epsilon, policy=None, player=1) -> str:
        """
        Epsilon-greedy action selection for exploration vs exploitation trade-off.
        This is a fundamental RL technique.
        
        Args:
            current_state: The current state in the grid world
            epsilon: Epsilon value (0-1) for exploration probability
            policy: Optional fixed policy to use
            player: Player identifier
        Returns: 
            Action to take ('up', 'down', 'left', or 'right')
        """
        q_current_state = self.world[self.current_position1.col][self.current_position1.row].q_a if player == 1 else self.world[self.current_position2.col][self.current_position2.row].q_a
        possible_action = [*q_current_state]

        if policy is not None:
            return random.choices(possible_action, weights=[policy[0], policy[1], policy[2], policy[3]], k=1)[0]

        # Epsilon-greedy decision
        epsilon_scaled = epsilon * 100
        value = random.choices(['explore', 'exploit'], weights=[epsilon_scaled, 100 - epsilon_scaled], k=1)

        if 'exploit' in value:
            # Exploitation: choose best known action
            self.exploration_stats['exploitation'] += 1
            return self.get_max_q(current_state=current_state, value_type='action', player=player)
        else:
            # Exploration: choose random action
            self.exploration_stats['exploration'] += 1
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

        # Walls or out of the world (col should be 0 to width-1, row should be 0 to height-1)
        if (col < 0 or col > self.width - 1) or (row < 0 or row > self.height - 1) or self.world[col][row].wall:
            return [self.current_position1.col, self.current_position1.row] if player == 1 else [self.current_position2.col, self.current_position2.row]
        return [col, row]

    def random_position(self):
        """
        This method returns a random position that isn't neither a wall or a terminal state
        Returns: column, row of the random position
        """
        found_position = False
        terminal_col = self.width - 1
        terminal_row = self.height - 1
        
        while not found_position:
            col = random.randint(0, self.width - 1)
            row = random.randint(0, self.height - 1)
            if(col == terminal_col and row == terminal_row):
                col = random.randint(0, self.width - 1)
                row = random.randint(0, self.height - 1)
            if not self.world[col][row].wall and not self.world[col][row].terminal and (col,row) not in self.killzones and (col,row) != (terminal_col, terminal_row):
                found_position = True
        return col, row

    def update_q_value(self, s, s_first, action, player=1):
        """
        Update Q-values using Q-learning formula: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        This is the core reinforcement learning update rule.
        
        Args:
            s: Current state
            s_first: Next state (s')
            action: Action taken
            player: Player identifier (1 or 2)
        """
        # Q-learning formula components
        current_q = s.q_a[action]
        reward = s_first.reward
        max_next_q = self.get_max_q(current_state=s_first, value_type='value', player=player)
        
        # TD error calculation
        td_error = reward + self.gamma * max_next_q - current_q
        
        # Apply Q-learning update
        s.q_a[action] = current_q + self.alpha * td_error
        
        # Store reward for episode statistics
        self.rewards_for_step.append(s_first.reward)
        self.step += 1
    
    def sarsa_update(self, s, s_first, action, action_first, player=1):
        """
        SARSA (State-Action-Reward-State-Action) - On-policy learning.
        Considers the action actually taken in the next state.
        Generally more conservative than Q-Learning.
        """
        current_q = s.q_a[action]
        reward = s_first.reward
        
        # Use the actual next action (on-policy)
        next_q = s_first.q_a[action_first] if action_first else self.get_max_q(s_first, 'value', player)
        
        td_error = reward + self.gamma * next_q - current_q
        s.q_a[action] = current_q + self.alpha * td_error
        
        self.rewards_for_step.append(reward)
        self.step += 1
    
    def expected_sarsa_update(self, s, s_first, action, player=1):
        """
        Expected SARSA - Uses expected value of next action instead of max.
        More conservative and stable than Q-Learning.
        """
        current_q = s.q_a[action]
        reward = s_first.reward
        
        # Calculate expected value using epsilon-greedy probabilities
        possible_actions = list(s_first.q_a.keys())
        num_actions = len(possible_actions)
        max_action = self.get_max_q(s_first, 'action', player)
        
        expected_value = 0
        for act in possible_actions:
            if act == max_action:
                prob = 1 - self.epsilon + (self.epsilon / num_actions)
            else:
                prob = self.epsilon / num_actions
            expected_value += prob * s_first.q_a[act]
        
        td_error = reward + self.gamma * expected_value - current_q
        s.q_a[action] = current_q + self.alpha * td_error
        
        self.rewards_for_step.append(reward)
        self.step += 1
    
    def double_q_learning_update(self, s, s_first, action, player=1):
        """
        Double Q-Learning - Uses two Q-tables to reduce overestimation bias.
        More stable learning with better convergence properties.
        """
        current_q = s.q_a[action]
        reward = s_first.reward
        
        # Random choice of which Q-table to use for update
        use_primary = random.random() < 0.5
        
        if use_primary:
            # Use secondary to select action, primary to evaluate
            best_action_primary = self.get_max_q(s_first, 'action', player)
            next_q = s_first.q_a[best_action_primary]
        else:
            # Use primary to select action, secondary to evaluate
            best_action_secondary = self.get_max_q(s_first, 'action', player)
            next_q = s_first.q_a[best_action_secondary]
        
        td_error = reward + self.gamma * next_q - current_q
        s.q_a[action] = current_q + self.alpha * td_error
        
        self.rewards_for_step.append(reward)
        self.step += 1
    
    def experience_replay(self, player=1):
        """
        Experience replay - Learn from past experiences.
        Breaks correlation in sequential experiences and improves stability.
        """
        memory = self.memory1 if player == 1 else self.memory2
        experiences = memory.get_experiences(batch_size=32)
        
        for exp in experiences:
            state = exp['state']
            action = exp['action']
            reward = exp['reward']
            next_state = exp['next_state']
            
            # Apply Q-learning update from replayed experience
            current_q = state.q_a[action]
            max_next_q = self.get_max_q(next_state, 'value', player)
            
            td_error = reward + self.gamma * max_next_q - current_q
            state.q_a[action] = current_q + self.alpha * td_error

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
            self.current_position2.col = self.height - 1
            self.current_position2.row = 0

        sum_reward = sum(self.rewards_for_step)
        self.rewards_for_episode.append(sum_reward)
        self.step_for_episode.append(self.step)
        
        # Record metrics for graphs
        self.graph_metrics['rewards'].append(sum_reward)
        self.graph_metrics['epsilons'].append(self.epsilon)
        self.graph_metrics['steps'].append(self.step)
        
        # Record episode in performance monitor
        exploration_count = self.exploration_stats.get('exploration', 0)
        self.performance_monitor.record_episode(
            episode_num=self.episode,
            reward=sum_reward,
            steps=self.step,
            epsilon=self.epsilon,
            exploration_count=exploration_count
        )
        
        self.rewards_for_step = []
        self.step = 0
        print('Episode: ', self.episode, ' Cumulative reward of: ', sum_reward)
        self.episode += 1

    def q_learning_algorithm(self, n_episode, random_start=True):
        """
        Advanced Q-learning algorithm with multiple strategies (Q-Learning, SARSA, Expected SARSA, Double Q-Learning).
        Supports experience replay for batch learning.
        Args:
            n_episode: Number of episodes to train
            random_start: Whether to start from random positions
        """
        algo_name = self.strategy.name.replace('_', ' ')
        print(f'[START] {algo_name} with RL Parameters: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}')
        
        s1 = self.get_current_state(player=1)
        s2 = self.get_current_state(player=2)
        
        while self.episode <= n_episode:
            # Adaptive epsilon decay using polynomial decay
            progress = self.episode / n_episode
            self.epsilon = self.initial_epsilon * (1 - progress) ** 2
            
            # Player 1 learning
            action1 = self.action_e_greedy(current_state=s1, epsilon=self.epsilon, player=1)
            self.current_position1.col, self.current_position1.row = self.get_next_state(action1, player=1)
            s1_first = self.get_current_state(player=1)
            
            # Store experience in memory for replay
            self.memory1.store_experience(s1, action1, s1_first.reward, s1_first, s1_first.terminal)
            self.memory1.track_visit(s1)
            
            # Get next action for SARSA (required for on-policy methods)
            action1_next = self.action_e_greedy(current_state=s1_first, epsilon=self.epsilon, player=1) if not s1_first.terminal else None
            
            # Apply selected algorithm for Player 1
            if self.strategy == AgentStrategy.SARSA and action1_next is not None:
                self.sarsa_update(s=s1, s_first=s1_first, action=action1, action_first=action1_next, player=1)
            elif self.strategy == AgentStrategy.EXPECTED_SARSA:
                self.expected_sarsa_update(s=s1, s_first=s1_first, action=action1, player=1)
            elif self.strategy == AgentStrategy.DOUBLE_Q:
                self.double_q_learning_update(s=s1, s_first=s1_first, action=action1, player=1)
            else:  # Q_LEARNING
                self.update_q_value(s=s1, s_first=s1_first, action=action1, player=1)
            
            s1 = s1_first
            
            # Player 2 learning
            action2 = self.action_e_greedy(current_state=s2, epsilon=self.epsilon, player=2)
            self.current_position2.col, self.current_position2.row = self.get_next_state(action2, player=2)
            s2_first = self.get_current_state(player=2)
            
            # Store experience in memory for replay
            self.memory2.store_experience(s2, action2, s2_first.reward, s2_first, s2_first.terminal)
            self.memory2.track_visit(s2)
            
            # Get next action for SARSA (required for on-policy methods)
            action2_next = self.action_e_greedy(current_state=s2_first, epsilon=self.epsilon, player=2) if not s2_first.terminal else None
            
            # Apply selected algorithm for Player 2
            if self.strategy == AgentStrategy.SARSA and action2_next is not None:
                self.sarsa_update(s=s2, s_first=s2_first, action=action2, action_first=action2_next, player=2)
            elif self.strategy == AgentStrategy.EXPECTED_SARSA:
                self.expected_sarsa_update(s=s2, s_first=s2_first, action=action2, player=2)
            elif self.strategy == AgentStrategy.DOUBLE_Q:
                self.double_q_learning_update(s=s2, s_first=s2_first, action=action2, player=2)
            else:  # Q_LEARNING
                self.update_q_value(s=s2, s_first=s2_first, action=action2, player=2)
            
            s2 = s2_first

            # Check for episode termination
            if s1.terminal or s2.terminal:
                # Perform experience replay if enabled
                if hasattr(self, 'experience_replay_enabled') and self.experience_replay_enabled:
                    self.experience_replay(player=1)
                    self.experience_replay(player=2)
                
                # Record episode metrics
                avg_reward = sum(self.rewards_for_episode[-1:]) if self.rewards_for_episode else 0
                self.performance_monitor.record_episode(
                    episode_num=self.episode,
                    reward=avg_reward,
                    steps=sum(self.step_for_episode[-1:]) if self.step_for_episode else 0,
                    epsilon=self.epsilon,
                    exploration_count=self.exploration_stats['exploration']
                )
                
                self.restart_episode(random_start)
                s1 = self.get_current_state(player=1)
                s2 = self.get_current_state(player=2)
            
            # Print progress every 100 episodes
            if self.episode % 100 == 0:
                avg_reward = sum(self.rewards_for_episode[-100:]) / 100 if len(self.rewards_for_episode) >= 100 else 0
                print(f'Episode: {self.episode}/{n_episode} | Avg Reward: {avg_reward:.2f} | Epsilon: {self.epsilon:.4f}')
        
        print(f'[COMPLETE] {algo_name} training finished')

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

    def save_training_results(self, episode_count):
        """Save training graphs and data after training completes."""
        algorithm_name = self.strategy.name.replace('_', ' ')
        
        # Generate and save graphs
        graph_dir = generate_graphs(
            self.graph_metrics,
            algorithm_name,
            self.width,
            episode_count
        )
        
        # Save training data
        data_file = save_training_data(
            self.graph_metrics,
            algorithm_name,
            self.width,
            self.alpha,
            self.gamma,
            self.initial_epsilon,
            episode_count
        )
        
        print(f'[RESULTS] Training results saved:')
        print(f'  - Graphs: {graph_dir}')
        print(f'  - Data: {data_file}')
        
        return graph_dir, data_file

class GridWorldGUI:
    """
    Enhanced GUI for GridWorld Q-Learning Simulator with modern features and real-time statistics.
    """
    def __init__(self, gridworld):
        self.gridworld = gridworld
        self.master = tk.Tk()
        self.master.title(f"🤖 GridWorld Q-Learning Simulator - {gridworld.name}")
        self.master.geometry("1000x750")
        self.master.configure(bg="#f0f0f0")
        
        # Training state
        self.is_training = False
        self.training_thread = None
        
        # Calculate dynamic cell size
        max_canvas_width = 450
        max_canvas_height = 450
        
        cell_size_by_width = max_canvas_width // gridworld.width
        cell_size_by_height = max_canvas_height // gridworld.height
        self.cell_size = min(cell_size_by_width, cell_size_by_height, 50)
        self.cell_size = max(self.cell_size, 15)
        
        self.canvas_width = gridworld.width * self.cell_size
        self.canvas_height = gridworld.height * self.cell_size
        
        # Create main layout with frames
        self.create_layout()
        
    def create_layout(self):
        """Create the enhanced GUI layout with multiple sections."""
        
        # Main container
        main_frame = tk.Frame(self.master, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== LEFT PANEL: Canvas and Grid =====
        left_panel = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        title_label = tk.Label(left_panel, text="🎮 Grid World Visualization", 
                              font=("Arial", 12, "bold"), bg="white")
        title_label.pack(pady=5)
        
        self.canvas = tk.Canvas(left_panel, width=self.canvas_width, height=self.canvas_height,
                               bg="white", relief=tk.SUNKEN, bd=1)
        self.canvas.pack(padx=5, pady=5)
        
        # Grid info
        info_frame = tk.Frame(left_panel, bg="white")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        info_text = f"Grid: {self.gridworld.height}×{self.gridworld.width} | "
        info_text += f"Walls: {len(self.gridworld.walls)} | "
        info_text += f"Killzones: {len(self.gridworld.killzones)}"
        
        info_label = tk.Label(info_frame, text=info_text, font=("Arial", 9),
                             bg="white", fg="#666")
        info_label.pack()
        
        # Draw initial grid
        self.draw_grid()
        
        # ===== RIGHT PANEL: Controls and Statistics =====
        right_panel = tk.Frame(main_frame, bg="#fff9e6", relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control section
        self.create_control_section(right_panel)
        
        # Statistics section
        self.create_statistics_section(right_panel)
        
        # Parameters info section
        self.create_parameters_section(right_panel)
        
    def create_control_section(self, parent):
        """Create the control buttons section with algorithm selection."""
        control_frame = tk.LabelFrame(parent, text="⚙️ Controls", font=("Arial", 11, "bold"),
                                      bg="#fff9e6", fg="#333", padx=10, pady=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Algorithm selection
        algo_frame = tk.Frame(control_frame, bg="#fff9e6")
        algo_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(algo_frame, text="Learning Algorithm:", font=("Arial", 9, "bold"), 
                bg="#fff9e6").pack(side=tk.LEFT, padx=5)
        
        self.algorithm_var = tk.StringVar(value="Q-Learning")
        algo_menu = tk.OptionMenu(algo_frame, self.algorithm_var, 
                                 "Q-Learning", "SARSA", "Expected SARSA", "Double Q-Learning",
                                 command=self.change_algorithm)
        algo_menu.config(font=("Arial", 9), bg="#fff9e6", width=18)
        algo_menu.pack(side=tk.LEFT, padx=5)
        
        # Enable experience replay checkbox
        self.experience_replay_var = tk.BooleanVar(value=False)
        replay_check = tk.Checkbutton(control_frame, text="Enable Experience Replay",
                                     variable=self.experience_replay_var, 
                                     font=("Arial", 9), bg="#fff9e6")
        replay_check.pack(anchor=tk.W, pady=5)
        
        # Start training button
        self.start_button = tk.Button(control_frame, text="▶ Start Training",
                                      command=self.start_q_learning, font=("Arial", 10, "bold"),
                                      bg="#4CAF50", fg="white", padx=15, pady=8,
                                      relief=tk.RAISED, bd=2, cursor="hand2",
                                      activebackground="#45a049", activeforeground="white")
        self.start_button.pack(fill=tk.X, pady=5)
        
        # Run sample episode button
        self.sample_button = tk.Button(control_frame, text="▶ Run Sample Episode",
                                       command=self.run_sample_episode_gui, font=("Arial", 10, "bold"),
                                       bg="#2196F3", fg="white", padx=15, pady=8,
                                       relief=tk.RAISED, bd=2, cursor="hand2",
                                       activebackground="#0b7dda", activeforeground="white")
        self.sample_button.pack(fill=tk.X, pady=5)
        
        # Analyze button
        self.analyze_button = tk.Button(control_frame, text="📈 Analyze Performance",
                                       command=self.analyze_performance, font=("Arial", 10),
                                       bg="#9C27B0", fg="white", padx=15, pady=6,
                                       relief=tk.RAISED, bd=2, cursor="hand2",
                                       activebackground="#7b1fa2", activeforeground="white")
        self.analyze_button.pack(fill=tk.X, pady=5)
        
        # Stop button
        self.stop_button = tk.Button(control_frame, text="⏹ Stop Training",
                                     command=self.stop_training, font=("Arial", 10, "bold"),
                                     bg="#f44336", fg="white", padx=15, pady=8,
                                     relief=tk.RAISED, bd=2, cursor="hand2",
                                     activebackground="#da190b", activeforeground="white",
                                     state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=5)
        
        # Reset button
        self.reset_button = tk.Button(control_frame, text="🔄 Reset Grid",
                                      command=self.reset_grid, font=("Arial", 10),
                                      bg="#FF9800", fg="white", padx=15, pady=6,
                                      relief=tk.RAISED, bd=2, cursor="hand2",
                                      activebackground="#e68900", activeforeground="white")
        self.reset_button.pack(fill=tk.X, pady=5)
    
    def change_algorithm(self, algo):
        """Change the learning algorithm."""
        if algo == "Q-Learning":
            self.gridworld.strategy = AgentStrategy.Q_LEARNING
        elif algo == "SARSA":
            self.gridworld.strategy = AgentStrategy.SARSA
        elif algo == "Expected SARSA":
            self.gridworld.strategy = AgentStrategy.EXPECTED_SARSA
        elif algo == "Double Q-Learning":
            self.gridworld.strategy = AgentStrategy.DOUBLE_Q
        self.status_label.config(text=f"Status: Algorithm changed to {algo}")
    
    def analyze_performance(self):
        """Analyze and display performance metrics."""
        if len(self.gridworld.performance_monitor.episode_metrics) == 0:
            self.status_label.config(text="Status: No training data to analyze")
            tk.messagebox.showwarning("No Data", "No training data available. Start training first!")
            return
        
        # Get metrics
        convergence = self.gridworld.performance_monitor.get_convergence_rate()
        avg_reward = self.gridworld.performance_monitor.get_moving_average('reward')
        avg_steps = self.gridworld.performance_monitor.get_moving_average('steps')
        current_epsilon = self.gridworld.epsilon
        
        # Get total episodes
        total_episodes = len(self.gridworld.performance_monitor.episode_metrics)
        
        # Build message
        msg = f"🎯 Performance Analysis Report\n"
        msg += f"{'='*35}\n\n"
        msg += f"📊 Dataset:\n"
        msg += f"  • Total Episodes: {total_episodes}\n"
        msg += f"  • Algorithm: {self.gridworld.strategy.name.replace('_', ' ')}\n\n"
        msg += f"📈 Metrics:\n"
        msg += f"  • Convergence Rate: {convergence:+.2%}\n"
        msg += f"  • Avg Reward (100 ep): {avg_reward:.2f}\n"
        msg += f"  • Avg Steps (100 ep): {avg_steps:.1f}\n"
        msg += f"  • Current Epsilon: {current_epsilon:.4f}\n\n"
        msg += f"💡 Interpretation:\n"
        
        if convergence > 0.1:
            msg += f"  ✓ Agent is improving rapidly!\n"
        elif convergence > 0:
            msg += f"  ✓ Agent is improving steadily\n"
        elif convergence > -0.1:
            msg += f"  ⚠ Agent performance is plateauing\n"
        else:
            msg += f"  ⚠ Agent is not converging\n"
        
        if avg_reward > 50:
            msg += f"  ✓ Excellent reward accumulation!\n"
        elif avg_reward > 0:
            msg += f"  ✓ Good reward accumulation\n"
        else:
            msg += f"  ⚠ Need more training for better rewards\n"
        
        tk.messagebox.showinfo("Performance Analysis", msg)
        self.status_label.config(text="Status: Performance analyzed ✓")
        
    def create_statistics_section(self, parent):
        """Create the real-time statistics display section."""
        stats_frame = tk.LabelFrame(parent, text="📊 Training Statistics", font=("Arial", 11, "bold"),
                                    bg="#fff9e6", fg="#333", padx=10, pady=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Algorithm display
        self.algorithm_label = tk.Label(stats_frame, text="Algorithm: Q-Learning", 
                                       font=("Arial", 9, "bold"), bg="#fff9e6", fg="#7B1FA2")
        self.algorithm_label.pack(anchor=tk.W, pady=2)
        
        # Episode counter
        self.episode_label = tk.Label(stats_frame, text="Episode: 0", 
                                     font=("Arial", 10, "bold"), bg="#fff9e6", fg="#1976D2")
        self.episode_label.pack(anchor=tk.W, pady=3)
        
        # Total steps
        self.steps_label = tk.Label(stats_frame, text="Total Steps: 0", 
                                   font=("Arial", 10, "bold"), bg="#fff9e6", fg="#388E3C")
        self.steps_label.pack(anchor=tk.W, pady=3)
        
        # Average reward
        self.avg_reward_label = tk.Label(stats_frame, text="Avg Reward: 0.00", 
                                        font=("Arial", 10, "bold"), bg="#fff9e6", fg="#D32F2F")
        self.avg_reward_label.pack(anchor=tk.W, pady=3)
        
        # Current epsilon
        self.epsilon_label = tk.Label(stats_frame, text="Exploration Rate (ε): 1.0000", 
                                     font=("Arial", 10, "bold"), bg="#fff9e6", fg="#F57C00")
        self.epsilon_label.pack(anchor=tk.W, pady=3)
        
        # Convergence rate
        self.convergence_label = tk.Label(stats_frame, text="Convergence Rate: 0.00%", 
                                         font=("Arial", 9, "bold"), bg="#fff9e6", fg="#00796B")
        self.convergence_label.pack(anchor=tk.W, pady=2)
        
        # Exploration ratio
        self.exploration_label = tk.Label(stats_frame, text="Exploration/Exploitation: 0/0", 
                                         font=("Arial", 9), bg="#fff9e6", fg="#555")
        self.exploration_label.pack(anchor=tk.W, pady=3)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.Scale(stats_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                    variable=self.progress_var, state=tk.DISABLED,
                                    bg="#E0E0E0", fg="#1976D2", troughcolor="#BDBDBD")
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Status text
        self.status_label = tk.Label(stats_frame, text="Status: Ready", 
                                    font=("Arial", 9, "italic"), bg="#fff9e6", fg="#666")
        self.status_label.pack(anchor=tk.W, pady=5)
        
    def create_parameters_section(self, parent):
        """Create the RL parameters display section."""
        params_frame = tk.LabelFrame(parent, text="🧠 RL Parameters", font=("Arial", 11, "bold"),
                                     bg="#fff9e6", fg="#333", padx=10, pady=10)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Learning rate
        alpha_text = f"Learning Rate (α): {self.gridworld.alpha:.4f}"
        alpha_label = tk.Label(params_frame, text=alpha_text, 
                              font=("Arial", 9), bg="#fff9e6", fg="#1565C0")
        alpha_label.pack(anchor=tk.W, pady=2)
        
        # Discount factor
        gamma_text = f"Discount Factor (γ): {self.gridworld.gamma:.4f}"
        gamma_label = tk.Label(params_frame, text=gamma_text, 
                              font=("Arial", 9), bg="#fff9e6", fg="#1565C0")
        gamma_label.pack(anchor=tk.W, pady=2)
        
        # Initial epsilon
        eps_text = f"Initial ε: {self.gridworld.initial_epsilon:.4f}"
        eps_label = tk.Label(params_frame, text=eps_text, 
                            font=("Arial", 9), bg="#fff9e6", fg="#1565C0")
        eps_label.pack(anchor=tk.W, pady=2)
        
    def update_statistics(self):
        """Update statistics display in real-time."""
        episode = self.gridworld.episode - 1
        self.episode_label.config(text=f"Episode: {episode}")
        
        total_steps = sum(self.gridworld.step_for_episode)
        self.steps_label.config(text=f"Total Steps: {total_steps}")
        
        if len(self.gridworld.rewards_for_episode) > 0:
            avg_reward = sum(self.gridworld.rewards_for_episode[-100:]) / len(self.gridworld.rewards_for_episode[-100:])
            self.avg_reward_label.config(text=f"Avg Reward: {avg_reward:.2f}")
        
        self.epsilon_label.config(text=f"Exploration Rate (ε): {self.gridworld.epsilon:.4f}")
        
        # Update algorithm display
        algo_name = self.gridworld.strategy.name.replace('_', ' ')
        self.algorithm_label.config(text=f"Algorithm: {algo_name}")
        
        # Display convergence rate
        if hasattr(self.gridworld, 'performance_monitor') and episode > 10:
            convergence = self.gridworld.performance_monitor.get_convergence_rate()
            self.convergence_label.config(text=f"Convergence Rate: {convergence:.2f}%")
        
        exp = self.gridworld.exploration_stats['exploration']
        expl = self.gridworld.exploration_stats['exploitation']
        self.exploration_label.config(text=f"Exploration/Exploitation: {exp}/{expl}")
        
        # Update progress
        if episode > 0:
            progress = min(100, (episode / 500) * 100)  # Assuming 500 episodes max
            self.progress_var.set(progress)
        
        self.status_label.config(text=f"Status: Training... Episode {episode}")

    def draw_grid(self):
        """Draw the grid world with enhanced visuals."""
        self.canvas.delete("all")
        
        # Draw grid background
        self.canvas.create_rectangle(0, 0, self.canvas_width, self.canvas_height, 
                                    fill="white", outline="#ddd")
        
        # Draw cells
        for col in range(len(self.gridworld.world)):
            for row in range(len(self.gridworld.world[col])):
                x1 = row * self.cell_size
                y1 = col * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # Determine cell color with enhanced palette
                fill_color = "#ffffff"
                outline_color = "#cccccc"
                
                if self.gridworld.world[col][row].wall:
                    fill_color = "#2c3e50"  # Dark gray
                    outline_color = "#1a252f"
                elif (col, row) in self.gridworld.killzones:
                    fill_color = "#e74c3c"  # Red
                    outline_color = "#c0392b"
                elif self.gridworld.world[col][row].terminal:
                    fill_color = "#27ae60"  # Green
                    outline_color = "#1e8449"
                
                # Draw cell
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, 
                                           outline=outline_color, width=1)
                
                # Add grid lines
                self.canvas.create_line(x1, y1, x1, y2, fill="#eee", width=0.5)
                self.canvas.create_line(x1, y1, x2, y1, fill="#eee", width=0.5)
        
        # Draw players with enhanced style
        self.draw_players()
        
    def draw_players(self):
        """Draw player positions with enhanced visuals."""
        # Player 1 position
        p1_x = self.gridworld.current_position1.row * self.cell_size + self.cell_size // 2
        p1_y = self.gridworld.current_position1.col * self.cell_size + self.cell_size // 2
        radius = min(self.cell_size // 3, 8)
        
        # Player 1 (Blue) - outer circle
        self.canvas.create_oval(p1_x - radius - 2, p1_y - radius - 2,
                               p1_x + radius + 2, p1_y + radius + 2,
                               fill="", outline="#1e88e5", width=2)
        # Player 1 (Blue) - inner circle
        self.canvas.create_oval(p1_x - radius, p1_y - radius,
                               p1_x + radius, p1_y + radius,
                               fill="#42a5f5", outline="#1565c0", width=1)
        self.canvas.create_text(p1_x, p1_y, text="P1", font=("Arial", 7, "bold"), fill="white")
        
        # Player 2 position
        p2_x = self.gridworld.current_position2.row * self.cell_size + self.cell_size // 2
        p2_y = self.gridworld.current_position2.col * self.cell_size + self.cell_size // 2
        
        # Player 2 (Orange) - outer circle
        self.canvas.create_oval(p2_x - radius - 2, p2_y - radius - 2,
                               p2_x + radius + 2, p2_y + radius + 2,
                               fill="", outline="#ff9800", width=2)
        # Player 2 (Orange) - inner circle
        self.canvas.create_oval(p2_x - radius, p2_y - radius,
                               p2_x + radius, p2_y + radius,
                               fill="#ffb74d", outline="#f57c00", width=1)
        self.canvas.create_text(p2_x, p2_y, text="P2", font=("Arial", 7, "bold"), fill="white")

    def start_q_learning(self):
        """Start Q-learning training in a separate thread."""
        if not self.is_training:
            self.is_training = True
            self.start_button.config(state=tk.DISABLED)
            self.sample_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Training started...")
            
            import threading
            self.training_thread = threading.Thread(target=self.run_q_learning, daemon=True)
            self.training_thread.start()

    def run_q_learning(self):
        """Execute Q-learning with periodic GUI updates."""
        try:
            # Set experience replay flag based on GUI checkbox
            self.gridworld.experience_replay_enabled = self.experience_replay_var.get()
            
            self.gridworld.q_learning_algorithm(n_episode=500, random_start=True)
            
            # Save training results (graphs and data)
            try:
                self.gridworld.save_training_results(500)
                self.status_label.config(text="Status: Training completed! ✓ Graphs saved!")
            except Exception as graph_error:
                print(f"Graph generation error: {graph_error}")
                self.status_label.config(text="Status: Training completed! ✓")
            
            self.update_statistics()
        except Exception as e:
            self.status_label.config(text=f"Status: Error - {str(e)}")
            print(f"Training error: {e}")
        finally:
            self.is_training = False
            self.start_button.config(state=tk.NORMAL)
            self.sample_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            # Update final statistics
            self.master.after(100, self.update_statistics)

    def stop_training(self):
        """Stop training (graceful shutdown)."""
        self.is_training = False
        self.status_label.config(text="Status: Training stopped by user")
        self.start_button.config(state=tk.NORMAL)
        self.sample_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def reset_grid(self):
        """Reset the grid to initial state."""
        if not self.is_training:
            self.gridworld.episode = 1
            self.gridworld.step = 1
            self.gridworld.rewards_for_episode = []
            self.gridworld.step_for_episode = []
            self.gridworld.exploration_stats = {'exploration': 0, 'exploitation': 0}
            self.gridworld.current_position1 = Position(0, 0)
            self.gridworld.current_position2 = Position(0, 0)
            
            self.draw_grid()
            self.episode_label.config(text="Episode: 0")
            self.steps_label.config(text="Total Steps: 0")
            self.avg_reward_label.config(text="Avg Reward: 0.00")
            self.epsilon_label.config(text="Exploration Rate (ε): 1.0000")
            self.exploration_label.config(text="Exploration/Exploitation: 0/0")
            self.progress_var.set(0)
            self.status_label.config(text="Status: Grid reset ✓")

    def run_sample_episode_gui(self):
        """Run a sample episode with visualization."""
        if not self.is_training:
            self.status_label.config(text="Status: Running sample episode...")
            self.start_button.config(state=tk.DISABLED)
            self.sample_button.config(state=tk.DISABLED)
            
            # Reset positions
            self.gridworld.current_position1.col, self.gridworld.current_position1.row = self.gridworld.random_position()
            self.gridworld.current_position2.col, self.gridworld.current_position2.row = self.gridworld.random_position()
            while (self.gridworld.current_position2.col == self.gridworld.current_position1.col and 
                   self.gridworld.current_position2.row == self.gridworld.current_position1.row):
                self.gridworld.current_position2.col, self.gridworld.current_position2.row = self.gridworld.random_position()
            
            current_state1 = self.gridworld.get_current_state(player=1)
            current_state2 = self.gridworld.get_current_state(player=2)
            total_steps = 0
            
            def move_players():
                nonlocal current_state1, current_state2, total_steps
                
                if not current_state1.terminal and not current_state2.terminal and total_steps < 100:
                    action1 = self.gridworld.get_max_q(current_state=current_state1, 
                                                       value_type='action', player=1)
                    action2 = self.gridworld.get_max_q(current_state=current_state2, 
                                                       value_type='action', player=2)
                    
                    self.gridworld.current_position1.col, self.gridworld.current_position1.row = \
                        self.gridworld.get_next_state(action1, player=1)
                    self.gridworld.current_position2.col, self.gridworld.current_position2.row = \
                        self.gridworld.get_next_state(action2, player=2)
                    
                    current_state1 = self.gridworld.get_current_state(player=1)
                    current_state2 = self.gridworld.get_current_state(player=2)
                    total_steps += 1
                    
                    self.draw_grid()
                    self.status_label.config(text=f"Status: Episode running... Steps: {total_steps}")
                    self.master.after(300, move_players)
                else:
                    self.start_button.config(state=tk.NORMAL)
                    self.sample_button.config(state=tk.NORMAL)
                    self.status_label.config(text=f"Status: Episode completed! Total steps: {total_steps} ✓")
            
            move_players()

def get_grid_size_from_user():
    """
    Create a simple dialog window to get grid size from user input.
    Returns: (height, width) tuple
    """
    from tkinter import simpledialog, messagebox
    
    # Create a temporary window for input
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    while True:
        try:
            height_str = simpledialog.askstring(
                "Grid Configuration",
                "Enter grid HEIGHT (default is 9):",
                initialvalue="9"
            )
            
            if height_str is None:  # User clicked Cancel
                messagebox.showinfo("Info", "Using default grid size: 9x9")
                root.destroy()
                return 9, 9
            
            width_str = simpledialog.askstring(
                "Grid Configuration",
                "Enter grid WIDTH (default is 9):",
                initialvalue="9"
            )
            
            if width_str is None:  # User clicked Cancel
                messagebox.showinfo("Info", "Using default grid size: 9x9")
                root.destroy()
                return 9, 9
            
            height = int(height_str)
            width = int(width_str)
            
            # Validate input
            if height < 3 or width < 3:
                messagebox.showerror("Invalid Input", "Grid size must be at least 3x3!")
                continue
            
            if height > 20 or width > 20:
                messagebox.showerror("Invalid Input", "Grid size cannot exceed 20x20!")
                continue
            
            messagebox.showinfo("Success", f"Grid size set to: {height}x{width}")
            root.destroy()
            return height, width
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid integer values!")
            continue

def get_rl_parameters():
    """
    Get RL parameters from user or use defaults.
    Returns: (alpha, gamma, epsilon) tuple
    """
    from tkinter import simpledialog, messagebox
    
    root = tk.Tk()
    root.withdraw()
    
    # Default RL parameters based on reinforcement learning best practices
    alpha_str = simpledialog.askstring(
        "RL Parameters",
        "Enter learning rate α (0.01-1.0, default: 0.1):",
        initialvalue="0.1"
    )
    alpha = float(alpha_str) if alpha_str else 0.1
    alpha = max(0.01, min(1.0, alpha))
    
    gamma_str = simpledialog.askstring(
        "RL Parameters",
        "Enter discount factor γ (0.1-1.0, default: 0.9):",
        initialvalue="0.9"
    )
    gamma = float(gamma_str) if gamma_str else 0.9
    gamma = max(0.1, min(1.0, gamma))
    
    epsilon_str = simpledialog.askstring(
        "RL Parameters",
        "Enter initial exploration rate ε (0.0-1.0, default: 1.0):",
        initialvalue="1.0"
    )
    epsilon = float(epsilon_str) if epsilon_str else 1.0
    epsilon = max(0.0, min(1.0, epsilon))
    
    messagebox.showinfo("RL Parameters", 
                       f"Configuration: α={alpha}, γ={gamma}, ε={epsilon}")
    root.destroy()
    return alpha, gamma, epsilon

def main():
    """
    Main function - Enhanced with fully dynamic RL configuration.
    No hardcoding of grid layouts, walls, or killzones.
    """
    print("[START] GridWorld Q-Learning Simulator")
    
    # Get grid size from user
    height, width = get_grid_size_from_user()
    
    # Get RL parameters from user
    alpha, gamma, epsilon = get_rl_parameters()
    
    # Q-Learning world with RL parameters
    q_learning_world = GridWord(
        name='Q-Learning Adaptive Agent',
        height=height,
        width=width,
        r_nt=-0.1,  # Small negative reward for each step (encourages efficiency)
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )
    
    # Set terminal state at bottom-right corner
    q_learning_world.set_terminal_state(row=width-1, col=height-1, reward=100)
    
    # Dynamically generate walls (no hardcoding)
    q_learning_world.set_wall(walls=None)
    
    # Dynamically place killzones (no hardcoding)
    q_learning_world.place_killzones(num_killzones=None)
    
    print(f"[CONFIG] Grid: {height}x{width} | RL Params: α={alpha}, γ={gamma}, ε={epsilon}")
    
    # Create GUI
    gui = GridWorldGUI(q_learning_world)
    gui.master.mainloop()

if __name__ == '__main__':
    main()