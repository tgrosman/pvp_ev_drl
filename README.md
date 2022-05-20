# p2p_with_ev

Bachelor Thesis: P2P energy trading with electric vehicles

This is the code repository for the bachelor thesis containing all relevant code, data, and plots.

To reproduce these results, the following commands are needed: 

Start the simulation:

gsy-e -l INFO run -d 31d -t 15s -s 15m --setup api_setup.setup_test_case_4  --start-date 2020-01-01 --enable-external-connection --paused --seed 123

Start the api client: 

gsy-e-sdk --log-level INFO run --base-setup-path /Users/timo/bachelor_thesis --setup asset_api_test_case_4_predict --run-on-redis