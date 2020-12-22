from easy21 import Action, State, step
from monte_carlo import MonteCarlo
from utils import zeros_2d, zeros_3d, mse_3d, dot_product, random_vec
 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random 

# I have it as external vars because its easier
mc = MonteCarlo()
mc_sol   = mc.solve(1000000)

# Question 4
class ValueFunctionApproximation:
    def __init__(self, gamma=1, gr_lambda=0.1, N_0=100):
        # x_dim = dealer's card
        # y_dim = player's sum
        # z_dim = action
        self.__action_values   = zeros_3d(10,21,2)
        self.__gamma           = gamma
        self.__lambda          = gr_lambda
        self.__N_0             = N_0

        self.__plr_sum_intrvls = [[1,6], [4,9], [7,12], [10,15], [13,18], [16,21]]
        self.__dlr_hnd_intrvls = [[1,4], [4,7], [7,10]]

        # If this does not work I will try randomized initialization
        self.__weights         = random_vec(len(self.__plr_sum_intrvls) * len(self.__dlr_hnd_intrvls) * 2)

        return

    def __reset(self):
        self.__action_values = zeros_3d(10,21,2) # I need it in order to compute MSE
        self.__weights       = random_vec(len(self.__plr_sum_intrvls) * len(self.__dlr_hnd_intrvls) * 2)

        return

    def __stt_act_to_features(self, state, action):
        plr_sum    = state.players_sum
        dlr_hnd    = state.dealers_card
        action_val = action.value

        return_vec = []

        for plr_sum_int in self.__plr_sum_intrvls:
            for dlr_hnd_int in self.__dlr_hnd_intrvls:
                for act in range(2):
                    if act != action_val:
                        return_vec.append(0)
                        continue

                    to_be_one = (plr_sum >= plr_sum_int[0]) and (plr_sum <= plr_sum_int[1])
                    to_be_one = to_be_one and ((dlr_hnd >= dlr_hnd_int[0]) and (dlr_hnd <= dlr_hnd_int[1]))

                    if to_be_one == True:
                        return_vec.append(1)
                    else:
                        return_vec.append(0)

        return return_vec

    def __stt_act_approx_val(self, state, action):
        features = self.__stt_act_to_features(state, action)

        return dot_product(features, self.__weights)

    def __stt_act_active_features(self, state, action):
        active_features = []

        features = self.__stt_act_to_features(state, action)

        for i in range(len(features)):
            if features[i] == 1:
                active_features.append(i)
                
        return active_features

    def __epsilon_policy(self, state):
        epsilon  = 0.05

        strategy = random.uniform(0.0, 1.0)

        if strategy >= epsilon:
            if self.__stt_act_approx_val(state, Action.HIT) <= self.__stt_act_approx_val(state, Action.STICK):
                return Action.STICK
            else:
                return Action.HIT
        else:
            return Action(random.randint(0, 1))
        
        print("Something wrong")
        return None

    def __run(self, episodes):
        # Its the Sarsa lambda algorithm from Sutton and Barto
        # book in page 305

        episodes_mse = []

        for _ in range(episodes):
            curr_state  = State()
            curr_action = self.__epsilon_policy(curr_state)
            z_vec       = [0] * len(self.__weights)

            while not curr_state.isTerminal():
                next_state, _ = step(curr_state, curr_action)

                delta = curr_state.getReward() if curr_state.getReward() != None else 0

                active_features = self.__stt_act_active_features(curr_state, curr_action)

                for active_feature_index in active_features:
                    delta = delta - self.__weights[active_feature_index]
                    z_vec[active_feature_index] += 1
                
                if next_state.isTerminal():
                    # doing w = w + alpha * delta * z_vec
                    for i in range(len(self.__weights)):
                        self.__weights[i] += 0.01 * delta * z_vec[i]
                    
                    break # Going to next episode

                next_action = self.__epsilon_policy(next_state)

                next_active_features = self.__stt_act_active_features(next_state, next_action)

                for nxt_act_index in next_active_features:
                    delta += self.__gamma * self.__weights[nxt_act_index]
                
                # Doing w = w + alpha * delta * z_vec
                for i in range(len(self.__weights)):
                    self.__weights[i] += 0.01 * delta * z_vec[i]

                # Doing z_vec = gamma * lambda * z_vec
                for i in range(len(self.__weights)):
                    z_vec[i] *= self.__gamma * self.__lambda
                
                curr_state  = next_state
                curr_action = next_action
            
            # Just computing the current state-action values 
            for i in range(10):
                for j in range(21):
                    for k in range(2):
                        temp_state = State()
                        temp_state.players_sum  = j + 1
                        temp_state.dealers_card = i + 1
                        self.__action_values[i][j][k] = self.__stt_act_approx_val(temp_state, Action(k))

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
        plt.title("VFA V*(s) after " + str(episodes) + " episodes.")

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
        vfa        = ValueFunctionApproximation(gamma=1, gr_lambda=curr_lambda)
        vfa_sol, _ = vfa.solve(1000)

        curr_mse = mse_3d(mc_sol, vfa_sol)

        lambdas.append(curr_lambda)
        mse_per_lambda.append(curr_mse)

        curr_lambda += 0.1

    plt.plot(lambdas, mse_per_lambda)
    plt.title("VFA's MSE Error through different lambdas for 1000 episodes")
    plt.xlabel("lambda")
    plt.xticks(lambdas)
    plt.ylabel("MSE")
    plt.show()

    episodes_to_run = 30000
    vfa_one  = ValueFunctionApproximation(gamma=1, gr_lambda=1.0)
    vfa_half = ValueFunctionApproximation(gamma=1, gr_lambda=0.5)
    vfa_zero = ValueFunctionApproximation(gamma=1, gr_lambda=0.0)

    _, ones_mse  = vfa_one.solve(episodes_to_run)
    _, halfs_mse = vfa_half.solve(episodes_to_run)
    _, zeros_mse = vfa_zero.solve(episodes_to_run)

    plt.plot(list(range(episodes_to_run)), ones_mse,  label="lambda=1.0")
    plt.plot(list(range(episodes_to_run)), halfs_mse, label="lambda=0.5")
    plt.plot(list(range(episodes_to_run)), zeros_mse, label="lambda=0.0")
    plt.title("VFA's MSE's with different values of lambda against number of episodes")
    plt.xlabel("Episodes")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()