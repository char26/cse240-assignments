import numpy as np
import helper
import random

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr)
    #   gamma which is another parameter helpful in calculating next move, in other words
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.reset()

        self.epsilon = 0.7

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    #   This is a function you should write.
    #   Function Helper:IT gets the current state, and based on the
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on.
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    def helper_func(self, state: list):
        snake_head_x = state[0]
        snake_head_y = state[1]
        snake_body = state[2]
        food_x = state[3]
        food_y = state[4]
        # print((snake_head_x, snake_head_y))

        # Understanding the current state of the game
        num_adjoining_wall_x_states = 0 # no wall
        if snake_head_x == helper.BOARD_LIMIT_MIN:
            num_adjoining_wall_x_states = 1 # wall to left
        elif snake_head_x == helper.BOARD_LIMIT_MAX:
            num_adjoining_wall_x_states = 2 # wall to right

        num_adjoining_wall_y_states = 0 # no wall
        if snake_head_y == helper.BOARD_LIMIT_MIN:
            num_adjoining_wall_y_states = 1 # wall above
        elif snake_head_y == helper.BOARD_LIMIT_MAX:
            num_adjoining_wall_y_states = 2 # wall below

        num_food_dir_x = 0 # no food
        if snake_head_y == food_y and snake_head_x > food_x:
            num_food_dir_x = 1 # food to left
        elif snake_head_y == food_y and snake_head_x < food_x:
            num_food_dir_x = 2 # food to right

        num_food_dir_y = 0 # no food
        if snake_head_x == food_x and snake_head_y > food_y:
            num_food_dir_y = 1 # food above
        elif snake_head_x == food_x and snake_head_y < food_y:
            num_food_dir_y = 2 # food below

        num_adjoining_body_top_states = 0 # no body above
        if (snake_head_x, snake_head_y - helper.GRID_SIZE) in snake_body:
            num_adjoining_body_top_states = 1

        num_adjoining_body_bottom_states = 0 # no body below
        if (snake_head_x, snake_head_y + helper.GRID_SIZE) in snake_body:
            num_adjoining_body_bottom_states = 1

        num_adjoining_body_left_states = 0 # no body to left
        if (snake_head_x - helper.GRID_SIZE, snake_head_y) in snake_body:
            num_adjoining_body_left_states = 1

        num_adjoining_body_right_states = 0 # no body to right
        if (snake_head_x + helper.GRID_SIZE, snake_head_y) in snake_body:
            num_adjoining_body_right_states = 1

        return [
            num_adjoining_wall_x_states,
            num_adjoining_wall_y_states,
            num_food_dir_x,
            num_food_dir_y,
            num_adjoining_body_top_states,
            num_adjoining_body_bottom_states,
            num_adjoining_body_left_states,
            num_adjoining_body_right_states
        ]


    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1

    #   This is the code you need to write.
    #   This is the reinforcement learning agent
    #   use the helper_func you need to write above to
    #   decide which move is the best move that the snake needs to make
    #   using the compute reward function defined above.
    #   This function also keeps track of the fact that we are in
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make.
    #   the LPC variable can be used to determine the learning rate (lr), but if
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.
    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.
    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.
    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively.
    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.
    def agent_action(self, state: list, points, dead):
        idx = tuple(self.helper_func(state))

        if self._train:
            last_action = self.a
            last_state = self.s

            if last_state:
                last_idx = tuple(self.helper_func(last_state))

                self.gamma = 0.7
                # Update Q table
                self.N[last_idx][last_action] += 1  # Increment the visit count
                lr = 0.7  # Update learning rate based on visit count
                reward = self.compute_reward(points, dead)
                self.points += reward
                sample = reward + self.gamma * max(self.Q[idx])
                self.Q[last_idx][last_action] = (1 - lr) * self.Q[last_idx][last_action] + (lr * sample)

            # Exploration vs. Exploitation
            if random.uniform(0, 40) < self.Ne:
                action = random.choice(self.actions)  # Explore
                self.Ne *= 0.999
            else:
                action = max(self.actions, key=lambda a: self.Q[idx][a])  # Exploit

            self.a = action # last action
            self.s = state # last state

        else:
            # Exploit only
            action = max(self.actions, key=lambda a: self.Q[idx][a])

        return action
