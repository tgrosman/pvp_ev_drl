import os
import pandas as pd

current_dir = os.path.dirname(__file__)


def read_forecast(house_number: int, load_id: int):
    """
    This method is used to load forecasted load and pv data. The data is used to place bids/offers according to the forecast.
    """
    load_forecast = pd.read_csv(
        os.path.join(current_dir, f'thesis_resources/load_forecast_id={load_id}_load={house_number}_kwh.csv'))
    load_values = load_forecast['Power (W)'].values
    pv_forecast = pd.read_csv(
        os.path.join(current_dir, f'thesis_resources/pv_forecast_id={load_id}_pv={house_number}_kwh.csv'))
    pv_values = pv_forecast['Power (W)'].values

    return load_values, pv_values


def is_ev(area_name: str) -> bool:
    """
    This method checks based on the assets name if the asset corresponds to an EV.
    """
    split_array = area_name.split('_')
    if split_array[0] != 'ev':
        return False
    return True


def search_trade_history(trade_list, device_list):
    trade_info = []
    for device in device_list:
        for element in trade_list:
            if element['buyer'] == device:
                trade_rate = round(element['trade_price'] / element['traded_energy'], 4)
                trade_info.append(
                    {'trade_type': 'bought', 'trade_rate': trade_rate, 'traded_energy': element['traded_energy']})
            elif element['seller'] == device:
                trade_rate = round(element['trade_price'] / element['traded_energy'], 4)
                trade_info.append(
                    {'trade_type': 'sold', 'trade_rate': trade_rate, 'traded_energy': element['traded_energy']})
    return trade_info


def check_availability(lower_bound, upper_bound, current_time_in_minutes):
    if lower_bound <= current_time_in_minutes < upper_bound:
        return False
    else:
        return True


def check_soc_constraint(lower_bound, current_time_in_minutes):
    if lower_bound == current_time_in_minutes:
        return True
    return False


def to_minutes(bound):
    return bound * 60


def check_if_none(value):
    if value is None:
        return True
    else:
        return False


def device_belongs_to_ev(area_name, ev_str):
    split_area_name = area_name.split('_')
    split_ev_name = ev_str.split('_')
    if split_area_name[1] == split_ev_name[1]:
        return True
    return False


