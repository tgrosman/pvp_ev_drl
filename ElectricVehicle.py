from numpy.random import normal
from random import randrange


class ElectricVehicle():
    #NORMAL_DIST_MEAN = 0.5
    #NORMAL_DIST_STD_DEV = 0.1

    def __init__(self, max_capacity, init_soc, unavailable_start, unavailable_end):
        self.max_capacity = max_capacity
        self.soc = init_soc
        self.unavailable_start = unavailable_start
        self.unavailable_end = unavailable_end
        self.current_capacity = self.max_capacity * self.soc

    def update_capacity_trades(self, amount):
        sum_cap = self.current_capacity + amount
        self.current_capacity = sum_cap
        self.soc = self.current_capacity / self.max_capacity

    def get_soc(self):
        return self.soc

    def simulate_consumption(self):
	"""
        soc_states = [0.3, 0.4, 0.5, 0.6, 0.7]
        new_soc = randrange(5)
        self.soc = soc_states[new_soc]
        print(f'Simulated New SoC: {self.soc}')
        self.current_capacity = self.max_capacity * self.soc
        """
        normal_sample = 0.0
        while not (self.soc > normal_sample > 0.0):
            normal_sample = normal(self.NORMAL_DIST_MEAN, self.NORMAL_DIST_STD_DEV)
        self.soc = round(normal_sample, 4)
        self.current_capacity = round(self.max_capacity * self.soc, 4)
