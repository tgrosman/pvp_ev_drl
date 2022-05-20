# flake8: noqa
"""
Template file for a trading strategy through the d3a API client
"""

import os
import pandas as pd
from threading import Thread, Event
from time import sleep
from gsy_e_sdk.redis_device import RedisDeviceClient
from gsy_e_sdk.rest_device import RestDeviceClient
from gsy_e_sdk.types import aggregator_client_type
from gsy_e_sdk.utils import get_area_uuid_from_area_name_and_collaboration_id
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from timeit import default_timer as timer

from RLClass import RLClass
from ElectricVehicle import ElectricVehicle
from stable_baselines3 import A2C
import api_util

current_dir = os.path.dirname(__file__)
print(current_dir)

################################################
# CONFIGURATIONS
################################################


# TODO update each of the below according to the assets the API will manage
automatic = True

oracle_name = 'oracle'

load_names = ['load_1', 'load_2', 'load_3', 'load_4', 'ev_1_load']
pv_names = ['pv_1', 'pv_2', 'pv_3', 'pv_4', 'ev_1_pv']
storage_names = []

# set market parameters
ticks = 10  # leave as is
TICK_INCREASE = 2.5

# Dict holding the EVs names and asset_uuids
ev_asset_uuids = {}
# TODO
ev_1_devices = ['ev_1_load', 'ev_1_pv']

evs_list = [ev_1_devices]


################################################
# ORACLE STRUCTURE
################################################


class Oracle(aggregator_client_type):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_finished = False
        self.market_slot_counter = 0
        # initialise variables
        self.asset_strategy = {}  # dictionary containing information and pricing strategy for each asset
        self.load_energy_requirements = {}
        self.generation_energy_available = {}
        self.storage_soc = {}
        self.starting_prices_load = {'load_1': 10.0, 'load_2': 5.0, 'load_3': 5.0, 'load_4': 0.0}
        self.starting_prices_pv = {'pv_1': 35.0, 'pv_2': 30.0, 'pv_3': 30.0, 'pv_4': 25.0}
        self.market_slot_trades1 = []
        self.grid_uuid = None
        self.community_uuid = None
        self.obs_ev_1 = None
        self.actions = []


    def set_events(self, event_rl, event_device):
        self.event_rl = event_rl
        self.event_device = event_device

    def set_RLEnv(self, rl_env):
        self.rl_env = rl_env

    def update_action(self, action):
        self.actions.append(action)

    ################################################
    # TRIGGERS EACH MARKET CYCLE
    ################################################
    def on_market_cycle(self, market_info):
        """
        Places a bid or an offer whenever a new market is created. The amount of energy
        for the bid/offer depends on the available energy of the PV, the required
        energy of the load, or the amount of energy in the battery.
        :param market_info: Incoming message containing the newly-created market info
        :return: None
        """
        start = timer()
        #print(f'Trades: {self.market_slot_trades1}')
        #if(self.event_rl.isSet()):
         #   print('SIMULATION RUNS TOO FAST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        # termination conditions (leave as is)
        if self.is_finished is True:
            return

        for area_uuid, area_dict in self.latest_grid_tree_flat.items():
            if area_dict['area_name'] == 'Grid':
                self.grid_uuid = area_uuid
            elif area_dict['area_name'] == 'Community':
                self.community_uuid = area_uuid

        last_market_stats = market_info['grid_tree'][self.grid_uuid]['children'][self.community_uuid][
            'last_market_stats']
        last_market_stats['market_slot'] = market_info['market_slot']



        env_ev_1.update_last_market_stats(last_market_stats)
        env_ev_1.update_trade_info(self.market_slot_trades1)


        self.event_rl.set()
        self.event_device.clear()

        self.event_device.wait()

        self.market_slot_trades1.clear()

        end = timer()
        print(f'Gym Envs: {end-start}')
        #print(f'Action to be taken: {self.actions}')
        ################################################
        # GET MARKET AND ENERGY FEATURES
        ################################################

        ################################################
        # SET ASSETS' STRATEGIES
        ################################################
        # TODO configure bidding / offer strategy for each tick
        """
        a dictionary is created to store values for each assets such as the grid fee to the market maker, the buy and sell price strategies,...
        current strategy is a simple ramp between 2 thresholds: Feed-in Tariff and Market Maker including grid fees.
        buy and sell strategies are stored in array of length 10, which are posted in ticks 0 through 9
        the default is that each device type you control has the same strategy, adapted with the grid fees
        """

        for area_uuid, area_dict in self.latest_grid_tree_flat.items():
            if "asset_info" not in area_dict or area_dict["asset_info"] is None:
                continue

            # Check if asset is an EV. If true, add asset_uuid to dictionary.
            if api_util.is_ev(area_dict['area_name']):
                if area_dict['area_name'] not in ev_asset_uuids:
                    ev_asset_uuids[area_dict['area_name']] = area_uuid
                continue

            self.asset_strategy[area_uuid] = {}
            self.asset_strategy[area_uuid]["asset_name"] = area_dict["area_name"]

            # Load strategy from forecast
            if 'energy_requirement_kWh' in area_dict["asset_info"]:
                load_strategy = []
                initial_price_load = self.starting_prices_load[self.asset_strategy[area_uuid]['asset_name']]
                for i in range(1, ticks + 1):
                    load_strategy.append(round(initial_price_load + (i * TICK_INCREASE), 2))
                self.asset_strategy[area_uuid]["buy_rates"] = load_strategy

            # Generation strategy from forecast
            if 'available_energy_kWh' in area_dict["asset_info"]:
                gen_strategy = []
                initial_price_pv = self.starting_prices_pv[self.asset_strategy[area_uuid]['asset_name']]
                for i in range(1, ticks + 1):
                    gen_strategy.append(round(initial_price_pv - (i * TICK_INCREASE), 2))
                self.asset_strategy[area_uuid]["sell_rates"] = gen_strategy

        #print()
        #print(f'Asset strategy: {self.asset_strategy}')
        #print()

        # Add agent's action to

        ################################################
        # POST INITIAL BIDS AND OFFERS FOR MARKET SLOT
        ################################################
        # takes the first element in each asset strategy to post the first bids and offers
        # all bids and offers are aggregated in a single batch and then executed
        # TODO how would you self-balance your managed energy assets?
        if self.market_slot_counter == 0:
            # Remove first four elements of actions
            self.actions = self.actions[4:]
            print('First market slot.')

        #print(f'Actions before placement: {self.actions}')


        for area_uuid, area_dict in self.latest_grid_tree_flat.items():

            if "asset_info" not in area_dict or area_dict["asset_info"] is None:
                continue

            # EV strategy
            # Either bid or offer per market slot, not both
            if api_util.is_ev(area_dict['area_name']):
                for action in self.actions:
                    if api_util.device_belongs_to_ev(area_dict['area_name'], action[2]):
                        if "energy_requirement_kWh" in area_dict["asset_info"] \
                                and area_dict["asset_info"]["energy_requirement_kWh"] > 0.0:
                            rate = action[1]
                            energy = area_dict["asset_info"]["energy_requirement_kWh"]
                            if action[0] == 'buy':
                                self.add_to_batch_commands.bid_energy_rate(
                                    asset_uuid=area_uuid, rate=rate, energy=energy)
                        elif "available_energy_kWh" in area_dict["asset_info"] \
                                and area_dict["asset_info"]["available_energy_kWh"] > 0.0:
                            if action[0] == 'sell':
                                self.add_to_batch_commands.offer_energy_rate(
                                    asset_uuid=area_uuid, rate=rate, energy=energy)

            if not api_util.is_ev(area_dict['area_name']):
                # Load strategy
                if (
                        "energy_requirement_kWh" in area_dict["asset_info"]
                        and area_dict["asset_info"]["energy_requirement_kWh"] > 0.0):
                    rate = self.asset_strategy[area_uuid]["buy_rates"][0]
                    energy = area_dict["asset_info"]["energy_requirement_kWh"]
                    self.add_to_batch_commands.bid_energy_rate(
                        asset_uuid=area_uuid, rate=rate, energy=energy)

                # Generation strategy
                if (
                        "available_energy_kWh" in area_dict["asset_info"]
                        and area_dict["asset_info"]["available_energy_kWh"] > 0.0):
                    rate = self.asset_strategy[area_uuid]["sell_rates"][0]
                    energy = area_dict["asset_info"]["available_energy_kWh"]
                    self.add_to_batch_commands.offer_energy_rate(
                        asset_uuid=area_uuid, rate=rate, energy=energy)

        response_slot = self.execute_batch_commands()

        self.market_slot_counter += 1
        # print(f'Market slot counter: {self.market_slot_counter}')

        self.actions.clear()
    ################################################
    # TRIGGERS EACH TICK
    ################################################
    """
    Places a bid or an offer 10% of the market slot progression. The amount of energy
    for the bid/offer depends on the available energy of the PV, the required
    energy of the load, or the amount of energy in the battery and the energy already traded.
    """

    def on_tick(self, tick_info):

        i = int(float(tick_info['slot_completion'].strip('%')) / ticks)  # tick num for index
        # print(f'Tick number: {i}')
        ################################################
        # ADJUST BID AND OFFER STRATEGY (if needed)
        ################################################
        # TODO manipulate tick strategy if required
        self.asset_strategy = self.asset_strategy

        ################################################
        # UPDATE/REPLACE BIDS AND OFFERS EACH TICK
        ################################################

        for area_uuid, area_dict in self.latest_grid_tree_flat.items():
            if "asset_info" not in area_dict or area_dict["asset_info"] is None:
                continue

            # Check if asset is an EV. If true, add asset_uuid to dictionary.
            if api_util.is_ev(area_dict['area_name']):
                continue

            # Load Strategy
            if "energy_requirement_kWh" in area_dict["asset_info"] \
                    and area_dict["asset_info"]["energy_requirement_kWh"] > 0.0:
                rate = self.asset_strategy[area_uuid]["buy_rates"][i]
                energy = area_dict["asset_info"]["energy_requirement_kWh"]
                self.add_to_batch_commands.bid_energy_rate(
                    asset_uuid=area_uuid, rate=rate, energy=energy)

            # Generation strategy
            if "available_energy_kWh" in area_dict["asset_info"] \
                    and area_dict["asset_info"]["available_energy_kWh"] > 0.0:
                rate = self.asset_strategy[area_uuid]["sell_rates"][i]
                energy = area_dict["asset_info"]["available_energy_kWh"]
                self.add_to_batch_commands.offer_energy_rate(asset_uuid=area_uuid, rate=rate,
                                                             energy=energy, replace_existing=True)
        response_tick = self.execute_batch_commands()

    ################################################
    # TRIGGERS EACH COMMAND RESPONSE AND EVENT
    ################################################
    def on_event_or_response(self, message):
        if "trade_list" in message:
            current_trades1 = api_util.search_trade_history(message['trade_list'], ev_1_devices)
            self.market_slot_trades1.extend(current_trades1)


    ################################################
    # SIMULATION TERMINATION CONDITION
    ################################################
    def on_finish(self, finish_info):
        # TODO export relevant information stored during the simulation (if needed)
        # TODO Save model
        model1.save('bachelor_thesis/model_ev_1')
        rew_ev_1 = pd.DataFrame(env_ev_1.reward_list, columns=['reward'])
        rew_ev_1.to_csv('bachelor_thesis/ev_1_results.csv')
        self.is_finished = True


################################################
# REGISTER FOR DEVICES AND MARKETS
################################################
def get_assets_name(indict: dict) -> dict:
    """
    This function is used to parse the grid tree and returned all registered assets
    wrapper for _get_assets_name
    """
    if indict == {}:
        return {}
    outdict = {"Area": [], "Load": [], "PV": [], "Storage": []}
    _get_assets_name(indict, outdict)
    return outdict


def _get_assets_name(indict: dict, outdict: dict):
    """
    Parse the collaboration / Canary Network registry
    Returns a list of the Market, Load, PV and Storage names the user is registered to
    """
    for key, value in indict.items():
        if key == "name":
            name = value
        if key == "type":
            area_type = value
        if key == "registered" and value:
            outdict[area_type].append(name)
        if 'children' in key:
            for children in indict[key]:
                _get_assets_name(children, outdict)


aggr = Oracle(aggregator_name=oracle_name)

if os.environ["API_CLIENT_RUN_ON_REDIS"] == "true":
    DeviceClient = RedisDeviceClient
    device_args = {"autoregister": True, "pubsub_thread": aggr.pubsub}
else:
    DeviceClient = RestDeviceClient
    simulation_id = os.environ["API_CLIENT_SIMULATION_ID"]
    domain_name = os.environ["API_CLIENT_DOMAIN_NAME"]
    websockets_domain_name = os.environ["API_CLIENT_WEBSOCKET_DOMAIN_NAME"]
    device_args = {"autoregister": False, "start_websocket": False}
    if automatic:
        registry = aggr.get_configuration_registry()
        registered_assets = get_assets_name(registry)
        load_names = registered_assets["Load"]
        pv_names = registered_assets["PV"]
        storage_names = registered_assets["Storage"]


def register_device_list(asset_names, asset_args, asset_uuid_map):
    for d in asset_names:
        print('Registered device:', d)
        if os.environ["API_CLIENT_RUN_ON_REDIS"] == "true":
            asset_args['area_id'] = d
        else:
            uuid = get_area_uuid_from_area_name_and_collaboration_id(simulation_id, d, domain_name)
            asset_args['area_id'] = uuid
            asset_uuid_map[uuid] = d
        asset = DeviceClient(**asset_args)
        if os.environ["API_CLIENT_RUN_ON_REDIS"] == "true":
            asset_uuid_map[asset.area_uuid] = asset.area_id
        asset.select_aggregator(aggr.aggregator_uuid)
    return asset_uuid_map


print()
print('Registering assets ...')
asset_uuid_map = {}
asset_uuid_map = register_device_list(load_names, device_args, asset_uuid_map)
asset_uuid_map = register_device_list(pv_names, device_args, asset_uuid_map)
asset_uuid_map = register_device_list(storage_names, device_args, asset_uuid_map)
aggr.device_uuid_map = asset_uuid_map
print()
print('Summary of assets registered:')
print()
print(aggr.device_uuid_map)


################################################
# Multithreading
################################################
def thread_rl(model, env):
    model.learn(N_ITERATIONS)
    # The reset method is called at the beginning of an episode
    obs = env.reset()

    for i in range(1, N_ITERATIONS):
        action, _states = model.predict(obs)
        obs, reward, dones, info = env.step(action)


def thread_sim():
    # Infinite loop in order to leave the client running in the background
    while True:
        sleep(0.5)


N_ITERATIONS = 96*180
ev_1 = ElectricVehicle(75., 0.5, 6, 14)

env_ev_1 = RLClass('ev_1', ev_1)


# Init multithreading events
event_rl = Event()
event_sim = Event()

# Set events according to both env
aggr.set_RLEnv(env_ev_1)
env_ev_1.set_device(aggr)
aggr.set_events(event_rl, event_sim)
env_ev_1.set_events(event_rl, event_sim)

model1 = A2C('MlpPolicy', env_ev_1, verbose=1)



# Init both threads
rl_thread = Thread(name='ev_1_thread', target=thread_rl, args=(model1, env_ev_1))
sim_thread = Thread(name='Simulation Thread', target=thread_sim, args=())

# Starting the threads
rl_thread.start()
sim_thread.start()
