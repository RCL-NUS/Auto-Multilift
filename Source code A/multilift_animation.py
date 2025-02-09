from Dynamics_animation import *
import numpy as np
import matplotlib.pyplot as plt
from casadi import *

# change the quadrotor number first!!!
uav_para   = np.array([1, 0.02, 0.02, 0.04, 6]) # 3,6 quadrotors
load_para  = np.array([5, 1]) # 5 kg, 1 m
cable_para = np.array([5e3, 1e-2, 2])

dt_sample  = 5e-3 # used in the 'step' function for simulating the environment
dt_ctrl    = 2e-2 # for control
ratio      = int(dt_ctrl/dt_sample)
system     = multilifting(uav_para, load_para, cable_para, dt_ctrl)
system.model()

# Ref_load   = np.load('closeloop 3 & 6 quad/trained data (6quad_heavier_load_best_learned_18)/Ref_Load_s.npy')
Ref_load   = np.load('new_learning_results_update_sample_acados/trained data (6quad_cl_acados_2m_update_grad_sample_3)/Ref_Load_s.npy')
# Load_State = np.load('closeloop 3 & 6 quad/trained data (6quad_heavier_load_best_learned_18)/Load_State.npy')
# Quad_State = np.load('closeloop 3 & 6 quad/trained data (6quad_heavier_load_best_learned_18)/Quad_State.npy')

# Load_State = np.load('openloop 3 & 6 quad/trained data(6quad_heavier_lood_openloop_backup_learned_11)/Load_State.npy')
# Quad_State = np.load('openloop 3 & 6 quad/trained data(6quad_heavier_lood_openloop_backup_learned_11)/Quad_State.npy')
# Load_State = np.load('new_learning_results_update_sample_acados/trained data (6quad_cl_acados_2m_update_grad_sample_3)/Load_State.npy')
# Quad_State = np.load('new_learning_results_update_sample_acados/trained data (6quad_cl_acados_2m_update_grad_sample_3)/Quad_State.npy')


# Load_State = np.load('new_learning_results_update_sample_acados/trained data (6quad_ol_acados_2m_update_sample_3)/Load_State.npy')
# Quad_State = np.load('new_learning_results_update_sample_acados/trained data (6quad_ol_acados_2m_update_sample_3)/Quad_State.npy')

# Load_State = np.load('new_learning_results_update_sample_acados/Evaluation results (6quad_cl_tensionref_11_fig8_comparison_usethis)/Load_State_tension11_fig8.npy')
# Quad_State = np.load('new_learning_results_update_sample_acados/Evaluation results (6quad_cl_tensionref_11_fig8_comparison_usethis)/Quad_State_tension11_fig8.npy')

Load_State = np.load('new_learning_results_update_sample_acados/Evaluation results (6quad_cl_tensionref_11_fig8_comparison_usethis)/Load_State_no_dt.npy')
Quad_State = np.load('new_learning_results_update_sample_acados/Evaluation results (6quad_cl_tensionref_11_fig8_comparison_usethis)/Quad_State_no_dt.npy')
# loss_episode = np.load('trained data (tension_ref_new_angle_para_best)/Loss_episode.npy')
# print('loss=',loss_episode)

length = 4000 #3000 for learnning weightings, 4000 for tensionref

STATE_0 = np.zeros((13,length))
STATE_1 = np.zeros((13,length))
STATE_2 = np.zeros((13,length))
STATE_3 = np.zeros((13,length))
STATE_4 = np.zeros((13,length))
STATE_5 = np.zeros((13,length))
STATE_l = np.zeros((13,length))
STATE_l0 = np.zeros((13,length))
STATE_l1 = np.zeros((13,length))
STATE_l9 = np.zeros((13,length))
REF_P_l = np.zeros((13,length))
for i in range(length):
    REF_P_l[:,i:i+1] = Ref_load[i]

k_train =0 


for i in range(length):
    STATE_0[:,i:i+1] = Quad_State[i+k_train*length][0]
    STATE_1[:,i:i+1] = Quad_State[i+k_train*length][1]
    STATE_2[:,i:i+1] = Quad_State[i+k_train*length][2]
    STATE_3[:,i:i+1] = Quad_State[i+k_train*length][3]
    STATE_4[:,i:i+1] = Quad_State[i+k_train*length][4]
    STATE_5[:,i:i+1] = Quad_State[i+k_train*length][5]
    STATE_l[:,i:i+1] = Load_State[i+k_train*length]
    # STATE_l0[:,i:i+1] = Load_State[i]
    # STATE_l1[:,i:i+1] = Load_State[i+length]
    # STATE_l9[:,i:i+1] = Load_State[i+4*length]

wing_len = 0.6
load_len = 1

# # play animation
# system.play_animation(k_train,wing_len, load_len, STATE_0, STATE_1, STATE_2, STATE_3, STATE_4, STATE_5, STATE_l, STATE_l0,STATE_l1,STATE_l9, REF_P_l, dt_sample)
k_train = 0 # in tensionref: k=0 for w/o dt, k=1 for w/ dt 
system.play_animation_ref(k_train,wing_len, load_len, STATE_0, STATE_1, STATE_2, STATE_3, STATE_4, STATE_5, STATE_l, dt_sample)