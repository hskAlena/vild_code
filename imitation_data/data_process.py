import h5py as hp
import glob
import pickle as pkl
import os
import pathlib
import argparse
import numpy as np


def removeP(traj_path, demo_list):
    pfiles = glob.glob(traj_path+'/*.p')

    demo_set = set()
    for ep_count in range(0, len(demo_list)):
        ep_i = demo_list[ep_count]
        demo_set.add(ep_i)
        
    lenf = len(pfiles)
    if lenf>len(demo_list):
        for i in range(lenf):
            demonum = int(pfiles[i].split('_')[-1][4:].split('.')[0])
            if demonum not in demo_set:
                os.remove(pfiles[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=int, default=21, help='Id of of the environment to run')
    parser.add_argument('--robo_task', action="store", default="reach", choices=["reach", "grasp", "full"], help='task')   
    args = parser.parse_args()
    
    env_dict = {
                ## Robosuite
                21 : "SawyerNutAssemblyRound",
                22 : "SawyerNutAssemblySquare",
                23 : "SawyerNutAssembly",
                24 : "SawyerPickPlaceBread",
                25 : "SawyerPickPlaceCan",
                26 : "SawyerPickPlaceCereal",
                27 : "SawyerPickPlaceMilk",
                28 : "SawyerPickPlace",
    }
    # env_name = env_dict[args.env_id]
    
    
    demo_dict = {
                ## Robosuite
                21 : "pegs-RoundNut",   #
                22 : "pegs-SquareNut",  #
                23 : "pegs-full",
                24 : "bins-Bread",      #
                25 : "bins-Can",        # 
                26 : "bins-Cereal",
                27 : "bins-Milk",        # 
                28 : "bins-full",
    }
    demo_name = demo_dict[args.env_id]
    
    ## for loading hdf5 
    demo_path = "%s/projects/vild_code/imitation_data/RoboTurkPilot/%s" % (pathlib.Path.home(), demo_name)
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = hp.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    

    ## for loading pickle
    traj_path = "%s/projects/vild_code/imitation_data/TRAJ_robo/%s" % (pathlib.Path.home(), env_name) 
    traj_path += "_%s" % args.robo_task 
    
    demo_list = []
    step_list = []
    if args.robo_task != "full":
        filename = "%s/projects/vild_code/imitation_data/TRAJ_robo/%s_%s/%s_%s_chosen.txt" % (pathlib.Path.home(), env_name, args.robo_task, env_name, args.robo_task)
    else:
        filename = "%s/projects/vild_code/imitation_data/TRAJ_robo/%s_%s/%s_chosen.txt" % (pathlib.Path.home(), env_name, args.robo_task, env_name)
    demo_idx = -1
    step_idx = -1
    with open(filename, 'r') as ff:
        i = 0
        for line in ff:

            i = i + 1
            line = line.replace(":", " ").replace("(", " ").replace(")", " ").replace(",", " ").split()
            if demo_idx == -1:
                demo_idx = line.index("demo") + 1
                step_idx = line.index("step") + 1
            demo_list += [int(line[demo_idx])]
            step_list += [int(line[step_idx])]
    
    removeP(traj_path, demo_list)
    print(demo_list)
    
    for ep_count in range(0, len(demo_list)):
        
        if ep_count > 9:
            break   # use only 10 demonstrations in experiments. 

        ep_i = demo_list[ep_count]

        # ep =  demos[ep_i]
        ep = "demo_%d" % ep_i 

        print("Epi %d" % (ep_i))
        if args.robo_task == "reach":
            pickle_path = os.path.join(traj_path,env_name+'_%s_demo%d.p'%(args.robo_task,ep_i))
        else:
            pickle_path = os.path.join(traj_path,env_name+'_demo%d.p'%(ep_i))
        
        pickleF = pkl.load(open(pickle_path,'rb'))
        
        states_Act = pickleF[0][0]
        masks = pickleF[1][0]
        rewards = pickleF[2][0]
        
        expert_state_list = []
        expert_action_list = []
        
        
        act_dim = f["data"]["demo_%d" %ep_i]['joint_velocities'].shape[1] + f["data"]["demo_%d" %ep_i]['gripper_actuations'].shape[1]
        
        if args.robo_task == "reach":
            for i in range(len(states_Act)):            
                expert_state_list.append(states_Act[i][:-act_dim])
                expert_action_list.append(states_Act[i][-act_dim:-1])
        else:
            for i in range(len(states_Act)):            
                expert_state_list.append(states_Act[i][:-act_dim])
                expert_action_list.append(states_Act[i][-act_dim:])
          
        
        expert_states = np.array(expert_state_list)
        expert_actions = np.array(expert_action_list)
        expert_masks = np.array(masks)
        expert_rewards = np.array(rewards)

        print(expert_states.shape)
        print(expert_actions.shape)
        print(expert_masks.shape)
        print(expert_rewards.shape)

        """ save data """
        traj_filename = traj_path + ("/%s_%s_TRAJ-ID%d" % (env_name, args.robo_task, ep_i))
        
        hf = hp.File(traj_filename + ".h5", 'w')
        hf.create_dataset('expert_source_path', data=pickle_path)    # network model file. 
        hf.create_dataset('expert_states', data=expert_states)
        hf.create_dataset('expert_actions', data=expert_actions)
        hf.create_dataset('expert_masks', data=expert_masks)
        hf.create_dataset('expert_rewards', data=expert_rewards)
        try:
            pickleF.close()
            hf.close()
        except:
            continue

        print("TRAJ result is saved at %s" % traj_filename)
    f.close()


  
    
'''
files = glob.glob('RoboTurkPilot/*/demo.hdf5')
for f in files:
    a = hp.File(f,'r')
    dirname = a['data'].attrs['env']
    if not os.path.isdir('RoboTurkPilot/'+dirname+'_reach'):
        os.mkdir('RoboTurkPilot/'+dirname+'_reach')
    textpath = os.path.join(os.getcwd(), 'RoboTurkPilot/'+dirname, dirname+'_reach_chosen.txt'))
    writef = open(textpath, 'w')
    for demo in a['data'].keys():
        length = a['data'][demo]['states'].shape[0]
        if length>=500 and length<515:
            writef.write('demo '+demo.split('_')[-1]+', step '+str(length)+', return '+
            
'''