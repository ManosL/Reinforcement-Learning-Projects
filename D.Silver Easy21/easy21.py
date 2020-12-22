import random
from enum import Enum

class Action(Enum):
    HIT   = 0
    STICK = 1

class State:
    def __init__(self):     #Creates an initial state
        self.players_sum   = random.randint(1, 10)
        self.dealers_card  = random.randint(1, 10)
        self.__dealers_sum = self.dealers_card
    
    def copy(self, other):
        self.players_sum   = other.players_sum
        self.dealers_card  = other.dealers_card
        self.__dealers_sum = other.__dealers_sum

    def isTerminal(self):
        if self.players_sum < 1 or self.players_sum > 21:
            return True
        
        if self.__dealers_sum < 1 or self.__dealers_sum >= 17:
            return True
        
        return False
    
    def hit(self):
        if not self.isTerminal():
            float_num = random.uniform(0, 3)

            multiply_factor = 1         # its 1 or -1 through black or red respectively

            if float_num <= (1 / 3):
                multiply_factor = -1
            
            self.players_sum += multiply_factor * random.randint(1, 10)
        else:
            print("Cannot do that, you are already on a terminal state!!!")

    def stick(self):
        if not self.isTerminal():
            while not self.isTerminal():
                float_num = random.uniform(0, 3)

                multiply_factor = 1         # its 1 or -1 through black or red respectively

                if float_num <= (1 / 3):
                    multiply_factor = -1
                
                self.__dealers_sum += multiply_factor * random.randint(1, 10)
        else:
            print("Cannot do that, you are already on a terminal state!!!")
    
    def getReward(self):
        if self.isTerminal():
            if self.players_sum < 1 or self.players_sum > 21: # Player busted
                return -1
            
            if self.__dealers_sum < 1 or self.__dealers_sum > 21: # Dealer busted
                return 1

            if self.players_sum == self.__dealers_sum:   # Tie
                return 0
            
            if self.players_sum > self.__dealers_sum: # Player win
                return 1
            
            if self.players_sum < self.__dealers_sum: # Player lose
                return -1
        else:
            #print("State is not Terminal")
            return None

# Question 1
def step(state, action):
    return_state = State()
    return_state.copy(state)

    if action == Action.STICK:
        return_state.stick()

        assert return_state.isTerminal()

        return return_state, return_state.getReward()
    
    assert action == Action.HIT

    return_state.hit()

    if return_state.isTerminal():
        return return_state, return_state.getReward()
    
    return return_state, None