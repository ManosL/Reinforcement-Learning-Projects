from easy21 import Action, State, step
from utils import zeros_2d, zeros_3d

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random 

#Question 2

class MonteCarlo:
    def __init__(self, gamma=1, N_0=100):
        # x_dim = dealer's card
        # y_dim = player's sum
        # z_dim = action
        self.__action_values = zeros_3d(10,21,2)
        self.__times_visited = zeros_2d(10,21)
        self.__act_stt_times = zeros_3d(10,21,2)
        self.__gamma         = gamma
        self.__N_0           = N_0

        return
    
    def __reset(self):
        self.__action_values = zeros_3d(10,21,2)
        self.__times_visited = zeros_2d(10,21)
        self.__act_stt_times = zeros_3d(10,21,2)

        return

    def __epsilon_policy(self, state):
        plr_hand = state.players_sum
        dlr_card = state.dealers_card
        state_times_visited = self.__times_visited[dlr_card - 1][plr_hand - 1]

        epsilon  = self.__N_0 / (self.__N_0 + state_times_visited)

        strategy = random.uniform(0.0, 1.0)

        if strategy >= epsilon:
            state_values_slot = self.__action_values[dlr_card - 1][plr_hand - 1]
            
            if state_values_slot[Action.HIT.value] <= state_values_slot[Action.STICK.value]:
                return Action.STICK
            else:
                return Action.HIT
        else:
            return Action(random.randint(0, 1))
        
        print("Something wrong")
        return None
    
    def __generate_episode(self):
        states = [State()]
        
        # Just check if its better to increase the counters after the episode

        while not states[0].isTerminal():
            action        = self.__epsilon_policy(states[0])
            plr_hand = states[0].players_sum
            dlr_card = states[0].dealers_card

            self.__times_visited[dlr_card - 1][plr_hand - 1] += 1

            self.__act_stt_times[dlr_card - 1][plr_hand - 1][action.value] += 1

            next_state, _ = step(states[0], action)
            
            states[0] = (states[0], action)

            states.insert(0, next_state) 

        states[0] = (states[0], None)

        return states

    def __run(self, episodes):
        for _ in range(episodes):
            curr_episode = self.__generate_episode()

            curr_goal = 0

            for i in range(len(curr_episode)):
                state  = curr_episode[i][0]
                action = curr_episode[i][1]
                
                if action == None:
                    curr_goal = state.getReward()
                else:
                    plr_hnd = state.players_sum
                    dlr_hnd = state.dealers_card

                    learning_rate = 1 / self.__act_stt_times[dlr_hnd - 1][plr_hnd - 1][action.value]

                    curr_goal = self.__gamma * curr_goal # because R_{t+1} does not exist for non terminal states
                    error = curr_goal - self.__action_values[dlr_hnd - 1][plr_hnd - 1][action.value]
                    
                    self.__action_values[dlr_hnd - 1][plr_hnd - 1][action.value] += learning_rate * error

        return

    def __plot(self, episodes):
        x_axis = range(1, 11)
        y_axis = range(1, 22)
        z_axis = []

        for x in x_axis:
            curr_z_ls = []
            for y in y_axis:
                action_vector = self.__action_values[x - 1][y - 1]
                if action_vector[Action.HIT.value] > action_vector[Action.STICK.value]:
                    curr_z_ls.append(action_vector[Action.HIT.value])
                else:
                    curr_z_ls.append(action_vector[Action.STICK.value])

            z_axis.append(curr_z_ls)
        
        fig = plt.figure()
        ax = Axes3D(fig)

        x_axis, y_axis = np.meshgrid(np.array(x_axis), np.array(y_axis))

        ax.plot_surface(x_axis, y_axis, np.transpose(np.array(z_axis)))

        plt.xlabel("Dealer's Card")
        plt.xticks(list(range(1,12)))
        plt.ylabel("Player's sum")
        plt.yticks(list(range(1,22)))
        plt.title("Monte Carlo V*(s) after " + str(episodes) + " episodes.")

        plt.show()
        return z_axis # Returns the optimal values for each state

    def solve(self, episodes=1000):
        self.__run(episodes)
        self.__plot(episodes)
        act_vals = self.__action_values
        self.__reset()
    
        return act_vals
