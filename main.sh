#cd code; python vild_main.py --il_method vild --env_id 21 --rl_method trpo --robo_task full --vild_loss_type BCE
#cd code; python plot_il.py --il_method vild --env_id 21 --robo_task full --rl_method trpo --vild_loss_type BCE #--plot_save
cd code; python test_model.py --il_method vild --env_id 21 --robo_task full --rl_method trpo --vild_loss_type BCE
