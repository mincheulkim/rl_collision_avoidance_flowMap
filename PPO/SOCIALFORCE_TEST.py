import numpy as np
import pysocialforce as psf

# 1. initial states
initial_state = np.zeros((6, 6))
initial_state[0, :] = np.array([-3.0, -3.0, 1.0, 0.0, 3.0, 0.0])
initial_state[1, :] = np.array([-3.0, -3.5, 1.0, 0.0, 3.0, 0.0])
initial_state[2, :] = np.array([-3.5, -3.0, 1.0, 0.0, 3.0, 0.0])
initial_state[3, :] = np.array([-3.0, 3.0, -0.5, -0.5, 0.0, -3.0])
initial_state[4, :] = np.array([-3.5, 3.0, -0.5, -0.5, 0.0, -3.0])
initial_state[5, :] = np.array([3.5, 3.0, 0.0, -1.0, -3.0, -3.0])

# 2. group #################################
groups = []
groups.append([])  # 0 grp 
groups.append([])  # 1 grp
groups.append([])  # 2 groups

# assign humans to groups
groups[0].append(0)
groups[0].append(1)
groups[0].append(2)
groups[1].append(3)
groups[1].append(4)
groups[2].append(5)

obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]




# initiate simulator
psf_sim = psf.Simulator(
        #initial_state, groups=groups, obstacles=None, config_file="./pysocialforce/config/default.toml"
        initial_state, groups=groups, obstacles=obs, config_file="./pysocialforce/config/example.toml"
    )
# do 1 updates
psf_sim.step(n=20)
ped_states, group_states = psf_sim.get_states()
#print('ped states:',ped_states)
#print('group states:',group_states)

with psf.plot.SceneVisualizer(psf_sim, "output_image_sf") as sv:
    sv.animate()
