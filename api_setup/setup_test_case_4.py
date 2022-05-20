# flake8: noqa

import os
import platform

from gsy_e.models.area import Area
from gsy_framework.constants_limits import ConstSettings
from gsy_e.models.strategy.external_strategies.pv import PVUserProfileExternalStrategy
from gsy_e.models.strategy.external_strategies.load import LoadProfileExternalStrategy
from gsy_e.models.strategy.market_maker_strategy import MarketMakerStrategy
from gsy_e.models.strategy.storage import StorageStrategy

current_dir = os.path.dirname(__file__)
print(current_dir)
print(platform.python_implementation())


def get_setup(config):
    ConstSettings.MASettings.MARKET_TYPE = 2
    ConstSettings.GeneralSettings.DEFAULT_MARKET_MAKER_RATE = 35

    area = Area(
        "Grid",
        [
            Area(
                "Community",
                [
                    Area("House 1",
                         [
                             Area("load_1", strategy=LoadProfileExternalStrategy(
                                 daily_load_profile=os.path.join(current_dir,
                                                                 "forecast/load_forecast_id=150_load=1.csv"),
                             ),
                                  ),
                             Area("pv_1", strategy=PVUserProfileExternalStrategy(
                                 power_profile=os.path.join(current_dir,
                                                            "forecast/pv_forecast_id=150_pv=1.csv"),
                                 panel_count=1,
                             ),
                                  ),
                             Area("ev_1_load", strategy=LoadProfileExternalStrategy(
                                 daily_load_profile=os.path.join(current_dir,
                                                                 "constant_power_2020.csv"),
                             ),
                                  ),
                             Area("ev_1_pv", strategy=PVUserProfileExternalStrategy(
                                 power_profile=os.path.join(current_dir,
                                                            "constant_power_2020.csv"),
                                 panel_count=1,
                             ),
                                  ),
                         ]),

                    Area("House 2",
                         [
                             Area("load_2", strategy=LoadProfileExternalStrategy(
                                 daily_load_profile=os.path.join(current_dir,
                                                                 "forecast/load_forecast_id=117_load=2.csv"),
                             ),
                                  ),
                             Area("pv_2", strategy=PVUserProfileExternalStrategy(
                                 power_profile=os.path.join(current_dir,
                                                            "forecast/pv_forecast_id=117_pv=2.csv"),
                                 panel_count=1,
                             ),
                                  ),
                             Area("ev_2_load", strategy=LoadProfileExternalStrategy(
                                 daily_load_profile=os.path.join(current_dir,
                                                                 "constant_power_2020.csv"),
                             ),
                                  ),
                             Area("ev_2_pv", strategy=PVUserProfileExternalStrategy(
                                 power_profile=os.path.join(current_dir,
                                                            "constant_power_2020.csv"),
                                 panel_count=1,
                             ),
                                  ),
                         ]),

                    Area("House 3",
                         [
                             Area("load_3", strategy=LoadProfileExternalStrategy(
                                 daily_load_profile=os.path.join(current_dir,
                                                                 "forecast/load_forecast_id=58_load=3.csv"),
                             ),
                                  ),
                             Area("pv_3", strategy=PVUserProfileExternalStrategy(
                                 power_profile=os.path.join(current_dir,
                                                            "forecast/pv_forecast_id=58_pv=3.csv"),
                                 panel_count=1,
                             ),
                                  ),
                             Area("ev_3_load", strategy=LoadProfileExternalStrategy(
                                 daily_load_profile=os.path.join(current_dir,
                                                                 "constant_power_2020.csv"),
                             ),
                                  ),
                             Area("ev_3_pv", strategy=PVUserProfileExternalStrategy(
                                 power_profile=os.path.join(current_dir,
                                                            "constant_power_2020.csv"),
                                 panel_count=1,
                             ),
                                  ),
                         ]),

                    Area("House 4",
                         [
                             Area("load_4", strategy=LoadProfileExternalStrategy(
                                 daily_load_profile=os.path.join(current_dir,
                                                                 "forecast/load_forecast_id=30_load=4.csv"),
                             ),
                                  ),
                             Area("pv_4", strategy=PVUserProfileExternalStrategy(
                                 power_profile=os.path.join(current_dir,
                                                            "forecast/pv_forecast_id=30_pv=4.csv"),
                                 panel_count=1,
                             ),
                                  ),
                             Area("ev_4_load", strategy=LoadProfileExternalStrategy(
                                 daily_load_profile=os.path.join(current_dir,
                                                                 "constant_power_2020.csv"),
                             ),
                                  ),
                             Area("ev_4_pv", strategy=PVUserProfileExternalStrategy(
                                 power_profile=os.path.join(current_dir,
                                                            "constant_power_2020.csv"),
                                 panel_count=1,
                             ),
                                  ),

                         ]),
                    Area("community_battery", strategy=StorageStrategy(
                        initial_soc=50,
                        battery_capacity_kWh=40,
                        max_abs_battery_power_kW=40,
                        final_buying_rate=16.99,
                        final_selling_rate=17.01
                    ),
                         ),

                ], grid_fee_constant=0, external_connection_available=True),

            Area("Market Maker", strategy=MarketMakerStrategy(energy_rate=35, grid_connected=True)),
        ],
        config=config, grid_fee_constant=0, external_connection_available=True
    )
    return area
