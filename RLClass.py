import gym
import numpy as np
import math
import pandas as pd
from gym import spaces
from datetime import datetime
import api_util
from timeit import default_timer as timer


class RLClass(gym.Env):
    def __init__(self, name, ev):
        super(RLClass, self).__init__()
        # Action space
        # Actions can take sell(0), idle(1) and buy(2)
        # Price in [0,40] where each step is 0.5 ct
        self.action_space = spaces.MultiDiscrete([3, 81])
        # Observations
        # min_trade_rate, max_trade_rate, soc, available, hour, month
        self.observation_space = spaces.Box(low=0, high=100,
                                            shape=(6,), dtype=np.float32)
        self.last_market_stats = None
        self.market_slot_trades = None
        self.name = name
        self.ev = ev
        self.unavailable_lower_bound = api_util.to_minutes(self.ev.unavailable_start)
        self.unavailable_upper_bound = api_util.to_minutes(self.ev.unavailable_end)
        self.ev_available = True
        self.trade_results = None
        self.reward = 0.0
        self.translated_actions = None
        self.last_action = None
        self.observations = np.array([0.5, 1., 0., 35., 0., 1.]).astype(np.float32)
        """
        self.df_ev = pd.DataFrame(
            columns=['interval', 'soc', 'available', 'min_trade_rate', 'max_trade_rate', 'hour', 'month',
                     'avg_trade_rate', 'action', 'action_price', 'traded_energy', 'reward'])
        """
        self.reward_list = []

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        return np.array([0.5, 1, 0, 35, 0, 1]).astype(np.float32)

    def step(self, action):
        # Translate and set the next action according to the agent's choice
        self.translated_actions = self.translate_actions(action)
        self.translated_actions.append(self.name)
        self.device.update_action(self.translated_actions)

        self.event_device.set()
        self.event_rl.clear()

        self.event_rl.wait()

        self.last_action = self.translated_actions

        if not self.last_action:
            self.last_action = self.translated_actions
        # Market slot timestamp can't be None
        if self.last_market_stats:
            datetime_object = datetime.strptime(self.last_market_stats['market_slot'], '%Y-%m-%dT%H:%M')
            current_time_in_minutes = (datetime_object.hour * 60) + datetime_object.minute

            if not api_util.check_availability(self.unavailable_lower_bound,
                                               self.unavailable_upper_bound, current_time_in_minutes):
                self.ev_available = False
            else:
                self.ev_available = True

            # Calculate soc based on trades
            if self.ev_available:
                if self.market_slot_trades:
                    self.trade_results = self.calculate_trade_results(self.market_slot_trades)
                    self.ev.update_capacity_trades(self.trade_results[0])

            # Swap None values to avoid errors
            if not self.trade_results:
                self.trade_results = [0.0, 0.0]

            # Swap None values to avoid errors
            for key, value in self.last_market_stats.items():
                if api_util.check_if_none(value):
                    self.last_market_stats[key] = 0.0

            self.reward = self.get_reward(current_time_in_minutes,
                                          self.last_market_stats['avg_trade_rate'], self.last_action)
            self.reward_list.append(self.reward)

            # Simulated use of EV
            start_simulate = timer()
            if self.ev_available and current_time_in_minutes == self.unavailable_upper_bound:
                self.ev.simulate_consumption()
            end_simulate = timer()

            self.observations = np.array(
                [self.ev.get_soc(), int(self.ev_available), round(self.last_market_stats['min_trade_rate'], 2),
                 round(self.last_market_stats['max_trade_rate'], 2), datetime_object.hour,
                 datetime_object.month]).astype(np.float32)

        # Set last action to currently chosen for nex market slot
        self.trade_results = None
        done = False
        info = {}

        return self.observations, self.reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass

    def get_reward(self, current_time_in_minutes, avg_trade_rate, last_action):
        if self.unavailable_lower_bound == current_time_in_minutes:
            if not self.ev.get_soc() >= 0.8000:
                return -1000.

        # Check availability
        if (not self.ev_available) and (not last_action[0] == 'idle'):
            return -100.

        # Check if bought energy exceeds battery max_capacity
        if (self.ev.get_soc() > 1.0) or (self.ev.get_soc() < 0.0):
            return -100.

        # Evaluate idle action
        if last_action[0] == 'idle' and (self.ev.get_soc() >= 0.8000 or not self.ev_available):
            return 50.0
        elif last_action[0] == 'idle':
            return -100.0


        if self.trade_results[0] == 0.0 and (last_action[0] == 'sell' or last_action[0] == 'buy'):
            return -5.
        # Normal reward calculation
        # Calculate the difference
        rate_difference = round(self.trade_results[1] - avg_trade_rate, 2)
        # Determine if positive or negative
        sign = math.copysign(1, rate_difference)
        # print(f'Sign: {sign}')
        # If difference is 0 (trade_price == avg_trade_price)
        if rate_difference == 0.0:
            return round(2.5 * abs(self.trade_results[0]) * 5.0, 2)
        # Only relative when buying energy
        elif last_action[0] == 'buy' and sign == 1:
            return -round(rate_difference * abs(self.trade_results[0]) * 5.0, 2)
        elif last_action[0] == 'buy' and sign == -1:
            return round(abs(rate_difference) * abs(self.trade_results[0]) * 5.0, 2)
        else:
            return round(rate_difference * abs(self.trade_results[0]) * 5.0, 2)

    def update_last_market_stats(self, last_market_stats):
        self.last_market_stats = last_market_stats

    def update_trade_info(self, market_slot_trades):
        self.market_slot_trades = market_slot_trades

    def calculate_trade_results(self, ev_trades):
        balance = 0.0
        trade_list = []
        if ev_trades:
            for element in ev_trades:
                if element['trade_type'] == 'bought':
                    balance = round(balance + element['traded_energy'], 4)
                    trade_list.append(round(element['trade_rate'] * element['traded_energy'], 4))
                elif element['trade_type'] == 'sold':
                    balance = round(balance - element['traded_energy'], 4)
                    trade_list.append(round(element['trade_rate'] * element['traded_energy'], 4))
        rate = round(sum(trade_list) / abs(balance), 2)
        if balance == 0.0:
            return [balance, 0.0]
        return balance, rate

    def translate_actions(self, actions):
        basic_action = self.calculate_basic_action(actions[0])
        price_action = self.calculate_price_action(actions[1])
        return [basic_action, price_action]

    def calculate_basic_action(self, action):
        switcher = {
            0: 'sell',
            1: 'idle',
            2: 'buy',
        }
        return switcher.get(action)

    def calculate_price_action(self, action):
        price_array = np.arange(0, 40.5, 0.5)
        return price_array[action - 1]

    def set_events(self, event_rl, event_device):
        self.event_rl = event_rl
        self.event_device = event_device

    def set_device(self, device):
        self.device = device
