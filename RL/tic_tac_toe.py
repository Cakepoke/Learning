from typing import Dict
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt


class TTT:
    """
    Tic-Tac-Toe, black is the first one.
    1 for black and 0 for white, . for blank.
    """
    def __init__(self, n: int):
        self.size = n
        self.greedy_eps = 0.1
        self.step_size = 0.5
        # self.init_q_value = 0.5
        self.init_q_value = 0
        self.black_Q = self.initial_values('b')
        self.white_Q = self.initial_values('w')

    # @staticmethod
    # def check_terminal(state: np.array):
    #     """
    #     Check whether the current state is a terminal state. Return True if is.
    #     """
    #     # Major Diagonal direction.
    #     n = len(state)
    #     if sum(state.diagonal()) == 0:
    #         return True, "w"
    #     elif sum(state.diagonal()) == n:
    #         return True, "b"
    #     # Minor Diagonal direction.
    #     md = state[np.arange(n), n - 1 - np.arange(n)]
    #     if sum(md) == 0:
    #         return True, "w"
    #     elif sum(md) == n:
    #         return True, "b"
    #     for i in range(n):
    #         # In Row-th direction.
    #         if sum(state[i]) == 0:
    #             return True, "w"
    #         elif sum(state[i]) == n:
    #             return True, "b"
    #         # In Column-th direction.
    #         if sum(state.T[i]) == 0:
    #             return True, "w"
    #         elif sum(state.T[i]) == n:
    #             return True, "b"
    #     return False, 0

    # def initial_values(self):
    #     """
    #     return: Q(state, action), where state is np.array and action is tuple for position.
    #     """
    #     # NOTE that np.array can't be keys.
    #     Q = {}
    #     a_map = set((i,j) for i in range(self.size) for j in range(self.size))
    #     for fill_num in range(self.size**2):
    #         # When stage-th step, we have stage positions occupied.
    #         if fill_num > 0:
    #             for pos in combinations(a_map, fill_num):
    #                 s = np.ones((self.size, self.size)) * np.nan
    #                 for w_pos in combinations(pos, fill_num // 2):
    #                     for ele in pos:
    #                         if ele in w_pos:
    #                             s[ele[0]][ele[1]] = "0"
    #                         else:
    #                             s[ele[0]][ele[1]] = "1"
    #                     Q[s] = {}
    #                     is_end, _ = self.check_terminal(s)
    #                     if is_end:
    #                         score = 0
    #                     else:
    #                         score = self.init_q_value
    #                     for action in set(a_map) - set(pos):
    #                         Q[s][action] = score
    #     return Q

    @staticmethod
    def check_terminal(state: str):
        """
        Check whether the current state is a terminal state.
        Return True and the winner if is.
        """
        n = int(len(state) ** .5)
        if n ** 2 != len(state):
            raise ValueError("The input is NOT a state!")
        a_map = np.array(range(n**2)).reshape((n, n))
        # Major Diagonal direction.
        s = [int(state[i]) for i in a_map.diagonal() if state[i] != "."]
        if len(s) == n:
            if sum(s) == 0:
                return True, "w"
            elif sum(s) == n:
                return True, "b"
        # Minor Diagonal direction.
        md = a_map[np.arange(n), n - 1 - np.arange(n)]
        s = [int(state[i]) for i in md if state[i] != "."]
        if len(s) == n:
            if sum(s) == 0:
                return True, "w"
            elif sum(s) == n:
                return True, "b"
        for i in range(n):
            # In Row-th direction.
            s = [int(state[j]) for j in a_map[i] if state[j] != "."]
            if len(s) == n:
                if sum(s) == 0:
                    return True, "w"
                elif sum(s) == n:
                    return True, "b"
            # In Column-th direction.
            s = [int(state[j]) for j in a_map.T[i] if state[j] != "."]
            if len(s) == n:
                if sum(s) == 0:
                    return True, "w"
                elif sum(s) == n:
                    return True, "b"
        # Check whether is full.
        if sum(1 for ele in state if ele != ".") == n ** 2:
            return True, 0
        else:
            return False, 0

    def initial_values(self, player):
        """
        return: Q(state, action), where state is a string and action is index of position.
        """
        # NOTE that np.array can't be keys.
        if player[0] == "b":
            action_stage = range(0, self.size ** 2, 2)
        elif player[0] == "w":
            action_stage = range(1, self.size ** 2, 2)
        else:
            raise ValueError("Choose the player as 'black' or 'white', where the black takes the first step. ")
        Q = {}
        a_map = range(self.size ** 2)
        for fill_num in action_stage:
            # When stage-th step, we have stage positions occupied.
            if fill_num > 0:
                for pos in combinations(a_map, fill_num):
                    for w_pos in combinations(pos, fill_num // 2):
                        # Create the state as a string.
                        s = ""
                        for ele in a_map:
                            if ele in pos:
                                if ele in w_pos:
                                    s += "0"
                                else:
                                    s += "1"
                            else:
                                s += "."
                        Q[s] = {}
                        # Check whether is terminal if more than n has taken for each one.
                        if (fill_num + 1) // 2 >= self.size:
                            is_end, _ = self.check_terminal(s)
                        else:
                            is_end = False
                        #
                        if is_end:
                            score = 0
                        else:
                            score = self.init_q_value
                        for action in set(a_map) - set(pos):
                            Q[s][action] = score
            else:
                s = "." * self.size ** 2
                Q[s] = {}
                score = self.init_q_value
                for action in set(a_map):
                    Q[s][action] = score
        return Q

    @staticmethod
    def policy_eps_greedy(state, Q, greedy_eps):
        action_set = Q[state]
        if np.random.random() < greedy_eps:
            z = np.random.randint(0, len(action_set))
            return list(action_set.keys())[z]
        else:
            greedy = max(action_set.items(), key=lambda x: x[1])
            return greedy[0]

    def sarsa(self, alpha=0.5, gamma=0.5, max_iter=10**3, trainer="b"):
        """
        Train one player with random policy of the opponent.
        """
        def oppo_round(s):
            oppo_action = self.policy_eps_greedy(s, Q_oppo, 1)
            return s[:oppo_action] + str(1 - me) + s[oppo_action + 1:]

        cnt = 0
        if trainer[0] == 'b':
            Q = self.black_Q
            Q_oppo = self.white_Q
            me = 1
        elif trainer[0] == "w":
            Q = self.white_Q
            Q_oppo = self.black_Q
            me = 0
        else:
            raise ValueError("Choose the player as 'black' or 'white', where the black takes the first step. ")
        while cnt < max_iter:
            start = "." * self.size ** 2
            if trainer[0] == 'b':
                state = start
            else:
                # Opponent's round
                state = oppo_round(start)
            action = self.policy_eps_greedy(state, Q, self.greedy_eps)
            while True:
                # Take action.
                oppo_state = state[:action] + str(me) + state[action + 1:]
                # Get reward
                flag, reward = self.check_terminal(oppo_state)
                if flag:
                    if isinstance(reward, str):
                        reward = 1 if reward == trainer[0] else -1
                    Q[state][action] = alpha * reward + (1 - alpha) * Q[state][action]
                    # print(Q[state][action])
                    break
                # Opponent's round
                new_state = oppo_round(oppo_state)
                # Get reward
                flag, r = self.check_terminal(new_state)
                if flag:
                    if isinstance(r, str):
                        r = 1 if r == trainer[0] else -1
                    Q[state][action] = alpha * r + (1 - alpha) * Q[state][action]
                    # print(Q[state][action])
                    break
                oppo_action = self.policy_eps_greedy(oppo_state, Q_oppo, self.greedy_eps)
                new_state = oppo_state[:oppo_action] + str(1 - me) + oppo_state[oppo_action + 1:]
                # Update my state
                new_action = self.policy_eps_greedy(new_state, Q, self.greedy_eps)
                Q[state][action] = alpha * (reward + gamma * Q[new_state][new_action]) + (1 - alpha) * Q[state][action]
                # print(Q[state][action])
                state = new_state
                action = new_action
            cnt += 1
        print("Training Finished.")

    def sarsa_both(self, learning_rate=0.01, farsight_rate=1, max_iter=10 ** 3):
        """
        Train one player with random policy of the opponent.
        """
        cnt = 0
        b_Q = self.black_Q
        w_Q = self.white_Q
        me = 1
        while cnt < max_iter:
            b_state = "." * self.size ** 2
            b_action = self.policy_eps_greedy(b_state, b_Q, self.greedy_eps)
            w_state = b_state[:b_action] + str(me) + b_state[b_action + 1:]
            b_reward = 0
            w_action = self.policy_eps_greedy(w_state, w_Q, self.greedy_eps)
            # s[:oppo_action] + str(1 - me) + s[oppo_action + 1:]
            while True:
                # White take action
                b_state_new = w_state[:w_action] + str(1 - me) + w_state[w_action + 1:]
                # Check terminal and get reward.
                flag, w_reward = self.check_terminal(b_state_new)
                if flag:
                    if isinstance(w_reward, str):
                        if w_reward == 'w':
                            w_reward, b_reward = 1, -1
                        else:
                            b_reward, w_reward = 1, -1
                    else:
                        w_reward, b_reward = 0, 0
                    w_Q[w_state][w_action] = learning_rate * w_reward + (1 - learning_rate) * w_Q[w_state][w_action]
                    b_Q[b_state][b_action] = learning_rate * b_reward + (1 - learning_rate) * b_Q[b_state][b_action]
                    break
                # Black choose action
                b_action_new = self.policy_eps_greedy(b_state_new, b_Q, self.greedy_eps)
                # Update black Q
                b_Q[b_state][b_action] = learning_rate * (b_reward + farsight_rate * b_Q[b_state_new][b_action_new]) \
                                         + (1 - learning_rate) * b_Q[b_state][b_action]
                # Black take action
                w_state_new = b_state_new[:b_action_new] + str(me) + b_state_new[b_action_new + 1:]
                # Check terminal and get reward.
                flag, b_reward = self.check_terminal(w_state_new)
                if flag:
                    if isinstance(b_reward, str):
                        if b_reward == 'w':
                            w_reward, b_reward = 1, -1
                        else:
                            b_reward, w_reward = 1, -1
                    else:
                        w_reward, b_reward = 0, 0
                    b_Q[b_state_new][b_action_new] = learning_rate * b_reward + (1 - learning_rate) * b_Q[b_state_new][b_action_new]
                    w_Q[w_state][w_action] = learning_rate * w_reward + (1 - learning_rate) * w_Q[w_state][w_action]
                    break
                # White choose action
                w_action_new = self.policy_eps_greedy(w_state_new, w_Q, self.greedy_eps)
                # Update white Q
                w_Q[w_state][w_action] = learning_rate * (w_reward + farsight_rate * w_Q[w_state_new][w_action_new]) \
                                         + (1 - learning_rate) * w_Q[w_state][w_action]
                # Update
                b_state, b_action = b_state_new, b_action_new
                w_state, w_action = w_state_new, w_action_new
            cnt += 1
        print("Training Finished.")

    def play(self, player, play_type="random", plot_on=False):
        """
        type: "random" for random policy on your behave, "ai" for trained policy and "diy" for play by your self.
        """
        def ai_round(s):
            op_action = self.policy_eps_greedy(s, oppo_Q, 0)
            return op_action, s[:op_action] + str(1 - me) + s[op_action + 1:]

        # player = input("Choose a player you'd like to play. b for the first and w for the second.")
        if player[0] == 'b':
            me = 1
            Q, oppo_Q = self.black_Q, self.white_Q
            sym, oppo_sym = "X", "O"
        elif player[0] == 'w':
            me = 0
            Q, oppo_Q = self.white_Q, self.black_Q
            sym, oppo_sym = "O", "X"
        else:
            raise ValueError("Choose b for the first or w for the second!")
        # Store actions
        actions = []
        # Plot
        if plot_on:
            ax = self.plot_initial()
        # Start!
        start = "." * self.size ** 2
        if player[0] == 'b':
            state = start
        else:
            # Opponent's round
            op_action, state = ai_round(start)
            # Store oppo's action
            actions.append(op_action)
            # Plot
            if plot_on:
                self.plot_position(ax, op_action, oppo_sym)
        while True:
            # Choose action.
            if play_type[0] == "r":
                action = self.policy_eps_greedy(state, Q, 1)  # Random
            elif play_type == "ai":
                action = self.policy_eps_greedy(state, Q, 0)  # Greedy
            elif play_type == "diy":
                action = int(input(f"Input your position (1 - {self.size ** 2}): ")) - 1
            else:
                raise ValueError("Choose one playing type as 'random', 'ai' or 'diy'!!!")
            # Take action. & Store
            actions.append(action)
            oppo_state = state[:action] + str(me) + state[action + 1:]
            # Plot
            if plot_on:
                self.plot_position(ax, action, sym)
            # Get terminal
            flag, winner = self.check_terminal(oppo_state)
            if flag:
                if isinstance(winner, str):
                    if winner == player[0]:
                        if plot_on:
                            print("You WIN!! Congratulation!")
                        return winner, actions
                    else:
                        if plot_on:
                            print("You LOSE! Try again.")
                        return winner, actions
                else:
                    if plot_on:
                        print("Draw.")
                    return winner, actions
                # break
            # Opponent's round
            op_action, new_state = ai_round(oppo_state)
            # Store oppo's action
            actions.append(op_action)
            # Plot
            if plot_on:
                self.plot_position(ax, op_action, oppo_sym)
            # Get terminal
            flag, winner = self.check_terminal(new_state)
            if flag:
                if isinstance(winner, str):
                    if winner == player[0]:
                        if plot_on:
                            print("You WIN!! Congratulation!")
                        return winner, actions
                    else:
                        if plot_on:
                            print("You LOSE! Try again.")
                        return winner, actions
                else:
                    if plot_on:
                        print("Draw.")
                    return winner, actions
                # break
            state = new_state

    def plot_initial(self):
        """
        Initialize the canvas.
        """
        fig = plt.figure()
        axe = fig.gca()
        axe.set_xticks(np.arange(0, self.size + 1, 1))
        axe.set_yticks(np.arange(0, self.size + 1, 1))
        axe.set_xticklabels([])
        axe.set_yticklabels([])
        axe.grid()
        # plt.axis('off')
        # plt.xticks([])  # 去掉横坐标值
        # plt.yticks([])  # 去掉纵坐标值
        return axe

    def plot_position(self, axe: plt.axes, pos, symbol):
        """
        Plot symbol in the game. "X" for black player and "O" for while player.
        """
        row = self.size - 1 - pos // self.size
        column = pos % self.size
        axe.text(column, row, symbol, fontsize=120)
        plt.pause(1)


if __name__ == '__main__':
    test = TTT(3)
    # test.sarsa(trainer="w", max_iter=10000)
    test.sarsa_both(max_iter=50000, learning_rate=0.1)
    # test.play("b", "diy")
    # test.play("w", "r", plot_on=True)
    # Calculate the win rate of my ai.
    tt_num = 5 * 10**4
    # For Black Q.
    win_num = 0
    draw_num = 0
    for _ in range(tt_num):
        my_winner, acts = test.play("w", "r")
        if isinstance(my_winner, str):
            if my_winner[0] == 'b':
                win_num += 1
            else:
                print(acts)
        elif my_winner == 0:
            draw_num += 1
    print(f"Win rate for Trained BLACK AI is {win_num/tt_num}")
    # For White Q.
    win_num = 0
    draw_num = 0
    for _ in range(tt_num):
        my_winner, acts = test.play("b", "r")
        if isinstance(my_winner, str):
            if my_winner[0] == 'w':
                win_num += 1
            else:
                print(acts)
        elif my_winner == 0:
            draw_num += 1
    print(f"Win rate for Trained WHITE AI is {win_num/tt_num}")
    # ----------------------------- Test result --------------------------------
    # Best now: n=3, max_iter=100000.
    # learning_rate=0.1, farsight_rate=1, greedy_epsilon=0.1, init_Q=0.5,
    # win-rate={Black: 0.988, White: 0.88848},
    #
    # NOTE:
    # 1. Decrease learning_rate and greedy_epsilon with cnt times.
    # 2. The effect of initial Q-value.
    # 3. Insymmetric of Q value
    # 4. Use other methods including MC, MC_first, Q-learning...
