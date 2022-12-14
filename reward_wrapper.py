from gym import Wrapper
import numpy as np
from constant import GameInfo


# Had to use a wrapper and not obs wrapper because I need access to the step info
class SFRewardWrapper(Wrapper):
    def __init__(self, env, hit_reward_strategy=2, rewards_coef=GameInfo.DEFAULT_REWARDS_COEF.value):
        """
        Street Fighter reward wrapper.
        It return the reward strategy I use to train the model
        :param hit_reward_strategy int: 0 -> health delta by punch/kick, 1 -> when hit, delta of life remaining, 3 -> avg 0 and 1
        :param rewards_coef dict: check REWARD_COEF. Gives more weight to different kind of rewards
        """

        super().__init__(env)
        self.env = env
        self.coef = rewards_coef
        self.hit_reward_strategy = hit_reward_strategy

    def reset(self):
        obs = self.env.reset()
        self.n_steps = self.round_steps = self.fight_won = 0
        # Create a attribute to hold the score delta
        self.previous_matches_won = self.previous_enemy_matches_won = 0
        self.previous_health = self.previous_enemy_health = GameInfo.MAX_HEALTH.value
        self.round_fight = 0
        #TODO: move this elsewhere
        self.info = {}
        return obs

    def step(self, action):
        # Take a step
        # print('ACTION')
        # print(action)
        obs, game_reward, done, info = self.env.step(action)
        # keep track of the game reward just to be able to check
        info['game_reward'] = game_reward

        # if we see the frame match_won == 2 it means we won the fight
        won_fight = int(info['matches_won'] == 2)
        # if won_fight:
        #     print("WON THE FIGHT")
        won_round = info['matches_won'] > self.previous_matches_won
        # if won_round:
        #     print("WON ROUND")
        self.fight_won += won_fight
        info['fight_won'] = self.fight_won
        lost_fight = int(info['enemy_matches_won'] == 2)
        # if lost_fight:
        #     print("LOST THE FIGHT")
        lost_round = int(info['enemy_matches_won'] > self.previous_enemy_matches_won)
        # if lost_round:
        #     print('LOST ROUND')
        # Reshape the reward function
        # score_delta = info['score'] - self.score
        # not using the score as:
        # 1. It increase even between round which makes the network thinks it can help
        # 2. It has a bias toward longer matches as you get point even if you lose round
        matches_won_delta = won_round
        enemy_matches_won_delta = lost_round
        # Was toying around with decreasing reward as the round goes up as well as the fight
        # and increasing as you play longer which mean you're still alive
        # The challenge here is to normalize it correctly to provide the right reward prioritization
        # Additionally because you don't hit or get hit all frames it can provide information for the model hard to understand
        # total amount of steps in the episode
        self.n_steps += 1
        # total amount of steps in the current round
        self.round_steps += 1
        # total amount of steps in the current fight
        self.round_fight += 1

        # just making sure we don't somehow have bad health numbers
        current_health = np.clip(info['health'], 0, GameInfo.MAX_HEALTH.value)
        current_enemy_health = np.clip(info['enemy_health'], 0, GameInfo.MAX_HEALTH.value)

        # Adding all the rewards to the info so it is easier to see what's going on
        # collect the different subreward in the info so it is easier to debut
        # losing health is negative (delta is always <= 0 )
        # % change delta/old_health. So the closer it get to 0 the highest the reward will be
        info['r_my_health'] = self.health_reward_calc(new_health=current_health, old_health=self.previous_health, won_round_or_fight=matches_won_delta + won_fight)
        # enemy losing health is positive
        info['r_enemy_health'] = - self.health_reward_calc(new_health=current_enemy_health, old_health=self.previous_enemy_health, won_round_or_fight=enemy_matches_won_delta + lost_fight)
        # if I win the round I add +1
        info['r_match_won'] = matches_won_delta
        # If I win the fight I add +1
        info['r_won_fight'] = won_fight
        # if I lose the round -1
        info['r_enemy_won'] = - enemy_matches_won_delta
        # if I lose the the fight -1
        info['r_enemy_won_fight'] = - lost_fight
        # Tried to add some rule there but ignoring it for now
        info['r_time_decay'] = - (-((self.n_steps) * 0.00000000001) +
                                  ((self.round_steps) / (GameInfo.MAX_ROUND_SEC.value * GameInfo.FPS.value)) +
                                  ((self.round_fight) / (GameInfo.MAX_ROUND_SEC.value * GameInfo.FPS.value * 3))
                                  ) / 300
        # the more you go toward 0 minute the more you get a penalty. I want my agent to be aggressive
        info['r_timer_decay'] = - (GameInfo.TIMER_START.value - np.clip(info['timer'], 0, GameInfo.TIMER_START.value)) / GameInfo.TIMER_START.value
        reward = self.coef['my_health'] * info['r_my_health'] + \
            self.coef['enemy_health'] * info['r_enemy_health'] + \
            self.coef['round_won'] * info['r_match_won'] + \
            self.coef['round_lost'] * info['r_enemy_won'] + \
            self.coef['game_lost'] * info['r_enemy_won_fight'] + \
            self.coef['fight_won'] * info['r_won_fight'] + \
            self.coef['time_decay'] * info['r_time_decay'] + \
            self.coef['timer_decay'] * info['r_timer_decay']
        # the range should be between -3 and +3 ignoring the time component and the coeficients
        # If I win a fight (+1) I win a round (+1) and I give the hitting blow (+1) -> 3
        # So doing a small normalization to have the total reward beeing between -1 and 1
        reward = reward / (
            np.max((self.coef['my_health'], self.coef['enemy_health'])) +
            np.max((self.coef['round_won'], self.coef['round_lost'])) +
            np.max((self.coef['fight_won'], self.coef['game_lost'])) +
            self.coef['timer_decay']
        )
        # keep track of reward in the info. I find it easier to check
        info['reward'] = reward

        # info for current round could be used if we wanted different kind of rewards added
        self.previous_health = current_health
        self.previous_enemy_health = current_enemy_health
        self.previous_enemy_matches_won = info['enemy_matches_won']
        self.previous_matches_won = info['matches_won']
        # updating info for next round
        if won_round == 1 or lost_round == 1:
            self.round_steps = 0
        if won_fight == 1:
            self.round_fight = 0
        #TODO: move it
        self.info = info
        return obs, reward, done, info

    def health_reward_calc(self, new_health, old_health, won_round_or_fight):
        # doing some cleaning just in case..
        new_health = np.clip(new_health, 0, GameInfo.MAX_HEALTH.value)
        old_health = np.clip(old_health, 0, GameInfo.MAX_HEALTH.value)
        # if nothing happen, no reward
        # also if reset no reward
        # if I somehow have more health now, it's a bug so no reward
        if new_health == old_health or old_health == 0 or new_health > old_health or new_health == GameInfo.MAX_HEALTH.value:
            return 0
        # you can't lose health but still win the fight or round
        # I've seen happening at reset time eg. lose game
        if new_health < old_health and won_round_or_fight:
            return 0
        out = []
        # 3 strategies
        # 0. do % difference -> new vs old -> This should help to reward more at the end and bigger hit
        out.append((new_health - old_health) / old_health)
        # 1. do inverse % health remaining -> This should help to reward more the closer you get to the end
        out.append(-1 + new_health / GameInfo.MAX_HEALTH.value)
        # 2. do mean of 1 and 2
        out.append((out[0] + out[1]) / 2)
        return out[self.hit_reward_strategy]
