from easy21 import Action, State, step
from monte_carlo import MonteCarlo
from utils import zeros_2d, zeros_3d, mse_3d

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random 

# I have it as external vars because its easier
mc = MonteCarlo()
mc_sol   = mc.solve(1000000)

# Question 3
class Sarsa:
    def __init__(self, gamma=1, gr_lambda=0.1, N_0=100):
        # x_dim = dealer's card
        # y_dim = player's sum
        # z_dim = action
        self.__action_values = zeros_3d(10,21,2)
        self.__eligibility   = zeros_3d(10,21,2)
        self.__times_visited = zeros_2d(10,21)
        self.__act_stt_times = zeros_3d(10,21,2)
        self.__gamma         = gamma
        self.__lambda        = gr_lambda
        self.__N_0           = N_0

        return
    
    def __reset(self):
        self.__action_values = zeros_3d(10,21,2)
        self.__eligibility   = zeros_3d(10,21,2)
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

    def __run(self, episodes):
        curr_lambda  = self.__lambda
        episodes_mse = []

        for _ in range(episodes):
            self.__eligibility = zeros_3d(10,21,2)
            curr_state         = State()
            curr_action        = self.__epsilon_policy(curr_state)

            while not curr_state.isTerminal():
                # Finding curr reward and next state and action
                curr_reward   = 0 if curr_state.getReward() == None else curr_state.getReward()

                next_state, _ = step(curr_state, curr_action)
                next_action   = self.__epsilon_policy(next_state) if not next_state.isTerminal() else None

                curr_plr_hand = curr_state.players_sum
                curr_dlr_card = curr_state.dealers_card
                
                next_plr_hand = next_state.players_sum
                next_dlr_card = next_state.dealers_card

                # Applying the trace
                if next_action != None:
                    delta = curr_reward + self.__gamma * self.__action_values[next_dlr_card - 1][next_plr_hand - 1][next_action.value]
                else:
                    assert(next_state.getReward() != None)
                    delta = curr_reward + self.__gamma * next_state.getReward()

                delta = delta - self.__action_values[curr_dlr_card - 1][curr_plr_hand - 1][curr_action.value]

                self.__eligibility[curr_dlr_card - 1][curr_plr_hand - 1][curr_action.value] += 1

                self.__times_visited[curr_dlr_card - 1][curr_plr_hand - 1] += 1

                self.__act_stt_times[curr_dlr_card - 1][curr_plr_hand - 1][curr_action.value] += 1

                learning_rate = 1 / self.__act_stt_times[curr_dlr_card - 1][curr_plr_hand - 1][curr_action.value]

                # Its the step for all actions and states
                for i in range(10):
                    for j in range(21):
                        for k in range(2):
                            self.__action_values[i][j][k] += learning_rate * delta * self.__eligibility[i][j][k]
                            self.__eligibility[i][j][k]   *= self.__gamma * curr_lambda

                curr_state  = next_state
                curr_action = next_action
             
            episodes_mse.append(mse_3d(mc_sol, self.__action_values))

        return episodes_mse

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
        plt.title("Sarsa V*(s) after " + str(episodes) + " episodes.")

        plt.show()
        return z_axis # Returns the optimal values for each state

    def solve(self, episodes=1000):
        eps_mse  = self.__run(episodes)
        #self.__plot(episodes)
        act_vals = self.__action_values
        self.__reset()
    
        return act_vals, eps_mse

def main():
    mse_per_lambda = []
    lambdas        = []
    curr_lambda    = 0.0

    # Drawing the nexessary plots
    
    while curr_lambda <= 1.0:
        sars        = Sarsa(gamma=1, gr_lambda=curr_lambda)
        sars_sol, _ = sars.solve(1000)

        curr_mse = mse_3d(mc_sol, sars_sol)

        lambdas.append(curr_lambda)
        mse_per_lambda.append(curr_mse)

        curr_lambda += 0.1

    plt.plot(lambdas, mse_per_lambda)
    plt.title("Sarsa's MSE Error through different lambdas for 1000 episodes")
    plt.xlabel("lambda")
    plt.xticks(lambdas)
    plt.ylabel("MSE")
    plt.show()

    episodes_to_run = 20000
    sars_one  = Sarsa(gamma=1, gr_lambda=1.0)
    sars_half = Sarsa(gamma=1, gr_lambda=0.5)
    sars_zero = Sarsa(gamma=1, gr_lambda=0.0)

    _, ones_mse  = sars_one.solve(episodes_to_run)
    _, halfs_mse = sars_half.solve(episodes_to_run)
    _, zeros_mse = sars_zero.solve(episodes_to_run)

    plt.plot(list(range(episodes_to_run)), ones_mse,  label="lambda=1.0")
    plt.plot(list(range(episodes_to_run)), halfs_mse, label="lambda=0.5")
    plt.plot(list(range(episodes_to_run)), zeros_mse, label="lambda=0.0")
    plt.title("Sarsa's MSE's with different values of lambda against number of episodes")
    plt.xlabel("Episodes")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()