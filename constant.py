from enum import Enum


class GameInfo(Enum):
    DEFAULT_REWARDS_COEF = {'my_health': 1,
                            'enemy_health': 2,  # Gives more weight to attacking the enemy vs losing health -> I want to be aggressive
                            'round_won': 3,
                            'round_lost': 3,
                            'game_lost': 4,
                            'fight_won': 4,
                            'time_decay': 0,  # Ignore this, need to think about it a bit more, the formula ain't good
                            'timer_decay': 0.001  # I don't want to be penalizing too much here. Still If I don't do it he like to just defend lol
                            }
    COMBOS = [[],
              ['UP'],
              ['DOWN'],
              ['LEFT'],
              ['UP', 'LEFT'],
              ['DOWN', 'LEFT'],
              ['RIGHT'],
              ['UP', 'RIGHT'],
              ['DOWN', 'RIGHT'],
              ['B'],
              ['B', 'DOWN'],
              ['B', 'LEFT'],
              ['B', 'RIGHT'],
              ['A'],
              ['A', 'DOWN'],
              ['A', 'LEFT'],
              ['A', 'RIGHT'],
              ['C'],
              ['DOWN', 'C'],
              ['LEFT', 'C'],
              ['RIGHT', 'C'],
              ['Y'],
              ['DOWN', 'Y'],
              ['LEFT', 'Y'],
              ['DOWN', 'LEFT', 'Y'],
              ['RIGHT', 'Y'],
              ['X'],
              ['DOWN', 'X'],
              ['LEFT', 'X'],
              ['DOWN', 'LEFT', 'X'],
              ['RIGHT', 'X'],
              ['DOWN', 'RIGHT', 'X'],
              ['Z'],
              ['DOWN', 'Z'],
              ['LEFT', 'Z'],
              ['DOWN', 'LEFT', 'Z'],
              ['RIGHT', 'Z'],
              ['DOWN', 'RIGHT', 'Z']]
    MAX_HEALTH = 176
    FPS = 60
    MAX_ROUND_SEC = 99
    XPOS1_START = 205
    XPOS2_START = 307
    XPOS_MIN = 55
    XPOS_MAX = 457
    OLD_TIMER_START = 9923
    TIMER_START = 39208
    YDISTANCE_MIN = 0
    YDISTANCE_MAX = 67
    YPOS_MIN = 0
    YPOS_MAX = 67
    XDISTANCE_MIN = 0
    XDISTANCE_MAX = 187
    XDISTANCE_START = 79
    NO_ACTION = 0
    # status - 512 if standing, 514 if crouching, 516 if jumping, 518 blocking, 522 if normal attack, 524 if special attack, 526 if hit stun or dizzy, 532 if thrown
    STANDING_STATUS = 512
    CROUCHING_STATUS = 514
    JUMPING_STATUS = 516
    ACTIONABLE_STATUSES = [STANDING_STATUS, CROUCHING_STATUS, JUMPING_STATUS]
    JUMP_LAG = 4
    ACTION_BUTTONS = ['X', 'Y', 'Z', 'A', 'B', 'C']
