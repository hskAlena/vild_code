#!/bin/bash
#python code/vild_main.py --il_method vild --env_id 21 --rl_method trpo --seed 6 --robo_task full --vild_loss_type BCE
python vild_main.py --il_method infogail --env_id 21 --rl_method trpo --seed 6 --robo_task full --vild_loss_type BCE
#python test_model.py --il_method vild --env_id 21 --robo_task full --seed 6 --rl_method trpo --vild_loss_type BCE
#python test_model.py --il_method infogail --env_id 21 --robo_task full --seed 6 --rl_method trpo --vild_loss_type BCE
#cd code; python test_model.py --il_method infogail --env_id 21 --robo_task full --seed 4 --rl_method trpo --vild_loss_type BCE
#cd code; python test_model.py --il_method vild --env_id 21 --robo_task full --seed 4 --rl_method trpo --vild_loss_type BCE
#cd code; python plot_il.py --il_method vild --env_id 21 --robo_task full --rl_method trpo --vild_loss_type BCE #--plot_save
