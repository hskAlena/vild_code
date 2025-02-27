# Source code of Variational Imitation Learning with Diverse-quality Demonstrations
Source code of "Variational Imitation Learning with Diverse-quality Demonstrations" in ICML 2020.
This github repository includes python code and datasets used in the experiments. 

## Requirements
Experiments were run with Python 3.6.9 and these packages:
* pytorch == 1.3.1
* numpy == 1.14.0
* scipy == 1.0.1
* gym == 0.10.5
* mujoco-py == 1.50.1.68
* robosuite == 0.1.0
* colorama, termcolor, 
* h5py == 2.10.0

## Important scripts
* code/vild_main.py - Script to run experiments with RL-based IL algorithms. 
* code/bc_main.py - Script to run experiments with SL-based IL algorithms. 
* code/args_parser.py - Script for parsing arguments. Default hyper-parameters can be found here.
* code/core/irl.py - Script implementing classes of IRL/GAIL baseline algorithms. 
* code/core/vild.py - Script implementing VILD algorithm. The VILD class extends IRL class.
* code/core/ac.py - Script implementing classes of RL algorithms (TRPO, PPO, SAC).
* code/plot_il.py - Script for plotting experimental results in the paper. Directory code/results_IL contains log files of experimental results reported in the paper.

## Code examples
`cd code; python vild_main.py --il_method vild --env_id 21 --rl_method sac --vild_loss_type BCE`


## About RoboTurk Dataset
demo len = 1144
* f['data']['demo_9'].keys() = ['gripper_actuations', 'joint_velocities', 'left_dpos', 'left_dquat', 'right_dpos', 'right_dquat', 'states']
* f['data'].attrs.keys() =['date', 'env', 'repository_version', 'time']
* f['data']['demo_9']['states'].shape = (824, 47)
* f['data']['demo_9']['left_dpos'].shape = (0,)
* f['data']['demo_9']['right_dpos'].shape = (824, 3)
* f['data']['demo_9']['right_dquat'].shape = (824, 4)
* f['data']['demo_9']['left_dquat'].shape = (0,)
* f['data']['demo_9']['joint_velocities'].shape = (824, 7) == expert_actions
* f['data']['demo_9']['gripper_actuations'].shape = (824, 1)
* f['data']['demo_9'].attrs['model_file'] = 'model_9.xml'
* f['data'].attrs['env'] = 'SawyerNutAssemblyRound'


* generate .p file with state2obs_exact_chosen.py
* sort and generate .txt file with sort_robo_demo.py
* check quality of demo playback_chosen_demo.py
* vild_main.py imitation/RL_models 
* imitation_data/data_process.py -> .h5
* test_model.py .txt -> .npy

## Important arguments of these scripts
* To set IL algorithms, set argument --il_method *algorithm_name*.
*algorithm_name* can be: vild, irl (maximum entropy irl), gail, airl, vail, infogail.
Without setting --il_method, the default behavior of the code is to perform RL with the true reward function.
* To set RL algorithms, set argument --rl_method *algorithm_name*.
*algorithm_name* can be: trpo, sac, ppo.
* To set environments, set argument --env_id *env_id*.
Each env_id corresponds to each gym environment, e.g., 2 = HalfCheetah-v2 (see env_dict variable in args_parser.py).
* To run VILD with truncated importance sampling (default), set --per_alpha 2. To run VILD without importance sampling, set --per_alpha 0.
* To run VILD with log-sigmoid reward function (used for LunarLander and Robosuite), set --vild_loss_type BCE. The default behavior yields rewards that are always positive. To obtain negative rewards (used for LunarLander), set --bce_negative 1.

## Demonstration datasets
* Datasets for experiments except the Robosuite Reacher experiment are included in directory imitation_data/TRAJ_h5. 
* Original dmonstrations for robosuite experiments can be obtained from the Roboturk website: https://roboturk.stanford.edu/dataset_sim.html (Mandlekar et al., 2018).
We include processed demonstrations used in our Robosuite Reacher experiment in directory imitation_data/TRAJ_robo, where we terminate original demonstrations when the end-effector touch the object and choose 10 demonstrations whose length are approximatedly 500 time steps. 
