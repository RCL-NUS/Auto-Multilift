"""
This is the main function that tests the distributed MPC control and autotuning for multi-lifting systems.
----------------------------------------------------------------------------
Wang, Bingheng at Control and Simulation Lab, ECE Dept. NUS, Singapore
1st version: 16 Feb,2024
2nd version: 13 Mar,2024
3rd version: 19 April, 2024
------------------------Reference------------------------------
[4] Tao, R., Cheng, S., Wang, X., Wang, S. and Hovakimyan, N., 2023.
    "DiffTune-MPC: Closed-loop Learning for Model Predictive Control"
    arXiv preprint arXiv:2312.11384 (2023).
"""
import Dynamics
import Robust_Flight_MPC
from casadi import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os
import math
from scipy.spatial.transform import Rotation as Rot
import NeuralNet
import torch
from multiprocessing import Process, Array, Manager
import time as TM
from concurrent.futures import ThreadPoolExecutor

print("========================================")
print("Main code for training or evaluating Distributed Autotuning Multilifting Controller")
print("PLease choose mode")
mode = input("enter 't' or 'e' without the quotation mark:")
print("PLease choose ctrlmode")
ctrlmode = input("enter 's' or 'p' without the quotation mark:") # s: sequential, p: parallel
print("PLease choose sensmode")
sensmode = input("enter 's' or 'p' without the quotation mark:")
print("PLease choose trainmode")
trainmode = input("enter 'c' or 'o' without the quotation mark:") # c: closed-lopp, o: open-loop
print("========================================")


"""--------------------------------------Load environment---------------------------------------"""
uav_para     = np.array([1, 0.02, 0.02, 0.04, 3, 0.2]) # L quadrotors
load_para    = np.array([5, 1]) # 5 kg, 1 m
cable_para   = np.array([5e3, 1e-2, 2])
Jl           = np.array([[1, 1, 1.25]]).T # payload's moment of inertia
rg           = np.array([[0.1, 0.1, -0.1]]).T # coordinate of the payload's CoM in {Bl}
dt_sample    = 1e-2 # used in the 'step' function for simulating the environment
dt_ctrl      = 2e-2 # for control, 50Hz
ratio        = int(dt_ctrl/dt_sample)
stm          = Dynamics.multilifting(uav_para, load_para, cable_para, dt_ctrl)
stm.model()
horizon      = 10 # MPC's horizon
horizon_loss = 20 # horizon of the high-level loss for training, which can be longer than the MPC's horizon
nxl          = stm.nxl # dimension of the payload's state
nxi          = stm.nxi # dimension of the quadrotor's state
nui          = stm.nui # dimension of the quadrotor's control 
nul          = stm.nul # dimension of the payload's control which is equal to the number of quadrotors
nwsi         = 12 # dimension of the quadrotor state weightings
nwsl         = 12 # dimension of the payload state weightings
# learning rate
lr_nn        = 1e-4
lr_lp        = 1e-8

"""--------------------------------------Define neural network models-----------------------------------------"""
# quadrotor and load parameters
nq         = int(uav_para[4])
alpha      = 2*np.pi/nq
rl         = load_para[1]
L0         = cable_para[2]
loadp      = np.vstack((Jl,rg)) # payload's inertial parameter
Di_in, Di_h, Di_out = 6, 30, 2*nwsi + nui # for quadrotors
Dl_in, Dl_h, Dl_out = 12, 30, 2*nwsl + nul # for the payload
npi        = 2*nwsi + nui
npl        = 2*nwsl + nul
nlp        = len(loadp)

"""--------------------------------------Define controller--------------------------------------------------"""
gamma      = 1e-4 # barrier parameter, cannot be too small
gamma2     = 1e-15
GeoCtrl    = Robust_Flight_MPC.Controller(uav_para, dt_ctrl)
DistMPC    = Robust_Flight_MPC.MPC(uav_para, load_para, cable_para, dt_ctrl, horizon, gamma, gamma2)
DistMPC.SetStateVariable(stm.xi,stm.xq,stm.xl,stm.index_q)
DistMPC.SetCtrlVariable(stm.ui,stm.ul,stm.ti)
DistMPC.SetLoadParameter(stm.Jldiag,stm.rg)
DistMPC.SetDyn(stm.model_i,stm.model_l)
DistMPC.SetLearnablePara()
DistMPC.SetQuadrotorCostDyn()
DistMPC.SetPayloadCostDyn()
DistMPC.SetConstraints_Load()


"""--------------------------------------Define reference trajectories----------------------------------------"""
coeffa = np.load('Reference_traj_circle/coeffa.npy')
def Reference_for_MPC(time_traj, angle_t):
    Ref_xq  = [] # quadrotors' state reference trajectories for MPC, ranging from the current k to future k + horizon
    Ref_uq  = [] # quadrotors' control reference trajectories for MPC, ranging from the current k to future k + horizon
    Ref_xl  = np.zeros((nxl,horizon+1))
    Ref_ul  = np.zeros((nul,horizon))
    Ref0_xq = [] # current quadrotors' reference position and velocity
    # quadrotor's reference
    for i in range(nq):
        Ref_xi  = np.zeros((nxi,horizon+1))
        Ref_ui  = np.zeros((nui,horizon))
        for j in range(horizon):
            ref_p, ref_v, ref_a   = stm.new_circle_quadrotor(coeffa,time_traj + j*dt_ctrl, angle_t, i)
            # ref_p, ref_v, ref_a   = stm.hovering_quadrotor(angle_t, i)
            if i==0: # we only need to compute the payload's reference for an arbitrary quadrotor
                ref_pl, ref_vl, ref_al   = stm.new_circle_load(coeffa,time_traj + j*dt_ctrl)
                # ref_pl, ref_vl, ref_al   = stm.hovering_load()
            qd, wd, f_ref, fl_ref, M_ref = GeoCtrl.system_ref(ref_a, load_para[0], ref_al)
            ref_xi    = np.vstack((ref_p,ref_v,qd,wd))
            ref_ui    = np.vstack((f_ref,M_ref)) 
            Ref_xi[:,j:j+1] = ref_xi
            Ref_ui[:,j:j+1] = ref_ui
            if i==0:
                qld       = np.array([[1,0,0,0]]).T # desired quaternion of the payload, representing the identity matrix
                wld       = np.zeros((3,1)) # deisred angular velocity of the payload
                ref_xl    = np.vstack((ref_pl, ref_vl, qld, wld))
                ref_ul    = fl_ref/nul*np.ones((nul,1))
                Ref_xl[:,j:j+1] = ref_xl
                Ref_ul[:,j:j+1] = ref_ul
                if j==0:
                    Ref0_l   = ref_xl
            if j == 0:
                Ref0_xq += [np.vstack((ref_p,ref_v))]
               
        ref_p, ref_v, ref_a  = stm.new_circle_quadrotor(coeffa,time_traj + horizon*dt_ctrl, angle_t, i)
        # ref_p, ref_v, ref_a   = stm.hovering_quadrotor(angle_t, i)
        if i==0:
            ref_pl, ref_vl, ref_al   = stm.new_circle_load(coeffa,time_traj + horizon*dt_ctrl)
            # ref_pl, ref_vl, ref_al   = stm.hovering_load()
        qd, wd, f_ref, fl_ref, M_ref = GeoCtrl.system_ref(ref_a, load_para[0], ref_al)
        ref_xi    = np.vstack((ref_p,ref_v,qd,wd))
        Ref_xi[:,horizon:horizon+1] = ref_xi
        Ref_xq   += [Ref_xi]
        Ref_uq   += [Ref_ui]
        if i==0:
            ref_xl    = np.vstack((ref_pl, ref_vl, qld, wld))
            Ref_xl[:,horizon:horizon+1] = ref_xl
        
    return Ref_xq, Ref_uq, Ref_xl, Ref_ul, Ref0_xq, Ref0_l



"""--------------------------------------Define MPC gradient------------------------------------------------"""
MPCgrad    = Robust_Flight_MPC.MPC_gradient(stm.xi,stm.xl,stm.ti,DistMPC.para_i,DistMPC.para_l,DistMPC.loadp,horizon)


"""--------------------------------------Define sensitivity propagation-------------------------------------"""
Sensprop   = Robust_Flight_MPC.Sensitivity_propagation(uav_para,stm.xi,stm.xl,stm.ul,
                                                       DistMPC.para_i,DistMPC.para_l,DistMPC.loadp,horizon_loss,horizon)


"""--------------------------------------Parameterization of neural network output--------------------------"""
pmin, pmax = 0.01, 100 # lower and upper boundaries
def SetPara_quadrotor(nn_i_output):
    Qik_diag   = np.zeros((1,nwsi)) # diagonal weighting for the quadrotor's state in the running cost
    QiN_diag   = np.zeros((1,nwsi)) # diagonal weighting for the quadrotor's state in the terminal cost
    Rik_diag   = np.zeros((1,nui)) # diagonal weighting for the quadrotor's control in the running cost
    for k in range(nwsi):
        Qik_diag[0,k] = pmin + (pmax-pmin)*nn_i_output[0,k]
        QiN_diag[0,k] = pmin + (pmax-pmin)*nn_i_output[0,nwsi+k]
    for k in range(nui):
        Rik_diag[0,k] = pmin + (pmax-pmin)*nn_i_output[0,2*nwsi+k]
    weight_i   = np.hstack((Qik_diag,QiN_diag,Rik_diag))

    return weight_i

def SetPara_load(nn_l_output):
    Qlk_diag   = np.zeros((1,nwsl)) # diagonal weighting for the payload's state in the running cost
    QlN_diag   = np.zeros((1,nwsl)) # diagonal weighting for the payload's state in the terminal cost
    Rlk_diag   = np.zeros((1,nul)) # diagonal weighting for the payload's control in the running cost
    for k in range(nwsl):
        Qlk_diag[0,k] = pmin + (pmax-pmin)*nn_l_output[0,k]
        QlN_diag[0,k] = pmin + (pmax-pmin)*nn_l_output[0,nwsl+k]
    for k in range(nul):
        Rlk_diag[0,k] = pmin + (pmax-pmin)*nn_l_output[0,2*nwsl+k]
    weight_l   = np.hstack((Qlk_diag,QlN_diag,Rlk_diag))

    return weight_l

def chainRule_gradient_quad(nn_i_output):
    tunable = SX.sym('tp',1,Di_out)
    Qik_dg  = SX.sym('Qik_dg',1,nwsi)
    QiN_dg  = SX.sym('QiN_dg',1,nwsi)
    Rik_dg  = SX.sym('Rik_dg',1,nui)
    for k in range(nwsi):
        Qik_dg[0,k] = pmin + (pmax-pmin)*tunable[0,k]
        QiN_dg[0,k] = pmin + (pmax-pmin)*tunable[0,nwsi+k]
    for k in range(nui):
        Rik_dg[0,k] = pmin + (pmax-pmin)*tunable[0,2*nwsi+k]
    weighti = horzcat(Qik_dg,QiN_dg,Rik_dg)
    w_i_jaco= jacobian(weighti,tunable)
    w_i_jaco_fn = Function('w_i_jaco',[tunable],[w_i_jaco],['tp0'],['w_i_jacof'])
    weight_i_grad = w_i_jaco_fn(tp0=nn_i_output)['w_i_jacof'].full()
    return weight_i_grad

def chainRule_gradient_load(nn_l_output):
    tunable = SX.sym('tp',1,Dl_out)
    Qlk_dg  = SX.sym('Qlk_dg',1,nwsl)
    QlN_dg  = SX.sym('QlN_dg',1,nwsl)
    Rlk_dg  = SX.sym('Rlk_dg',1,nul)
    for k in range(nwsl):
        Qlk_dg[0,k] = pmin + (pmax-pmin)*tunable[0,k]
        QlN_dg[0,k] = pmin + (pmax-pmin)*tunable[0,nwsl+k]
    for k in range(nul):
        Rlk_dg[0,k] = pmin + (pmax-pmin)*tunable[0,2*nwsl+k]
    weightl = horzcat(Qlk_dg,QlN_dg,Rlk_dg)
    w_l_jaco= jacobian(weightl,tunable)
    w_l_jaco_fn = Function('w_l_jaco',[tunable],[w_l_jaco],['tp0'],['w_l_jacof'])
    weight_l_grad = w_l_jaco_fn(tp0=nn_l_output)['w_l_jacof'].full()
    return weight_l_grad

def convert_quadrotor_nn(nn_i_outcolumn):
    # convert a column tensor to a row np.array
    nn_i_row = np.zeros((1,Di_out))
    for i in range(Di_out):
        nn_i_row[0,i] = nn_i_outcolumn[i,0]
    return nn_i_row

def convert_load_nn(nn_l_outcolumn):
    # convert a column tensor to a row np.array
    nn_l_row = np.zeros((1,Dl_out))
    for i in range(Dl_out):
        nn_l_row[0,i] = nn_l_outcolumn[i,0]
    return nn_l_row

def QuadrotorMPC(xi_fb, xq_traj, uq_traj, xl_traj, ul_traj, Ref_xi, Ref_ui, Para_i, i, xi_temp, ui_temp, ci_quad, viol_xtemp, viol_utemp, viol_ctemp):
    opt_sol_i   = DistMPC.MPCsolverQuadrotor(xi_fb, xq_traj, xl_traj, ul_traj, Ref_xi, Ref_ui, Para_i, i)
    xi_opt      = opt_sol_i['xi_opt']
    ui_opt      = opt_sol_i['ui_opt']
    cox_opt_i   = opt_sol_i['costate_traj_opt']
    cox_ipopt_i = opt_sol_i['costate_ipopt']
    sum_viol_xi = 0
    sum_viol_ui = 0
    sum_viol_cxi= 0
    for ki in range(len(uq_traj[i])):
        sum_viol_xi  += LA.norm(xi_opt[ki,:]-xq_traj[i][ki,:])
        sum_viol_ui  += LA.norm(ui_opt[ki,:]-uq_traj[i][ki,:])
        sum_viol_cxi += LA.norm(cox_opt_i[ki,:]-cox_ipopt_i[ki,:])
    sum_viol_xi  += LA.norm(xi_opt[-1,:]-xq_traj[i][-1,:])
    viol_xi  = np.reshape(sum_viol_xi/len(xi_opt),(1,1))
    viol_ui  = np.reshape(sum_viol_ui/len(ui_opt),(1,1))
    viol_cxi = np.reshape(sum_viol_cxi/len(cox_opt_i),(1,1))
    xi_temp[:]  = np.reshape(xi_opt,(horizon+1)*nxi)
    ui_temp[:]  = np.reshape(ui_opt,horizon*nui)
    ci_quad[:]  = np.reshape(cox_ipopt_i,horizon*nxi)
    viol_xtemp[:]  = np.reshape(viol_xi,1)
    viol_utemp[:]  = np.reshape(viol_ui,1)
    viol_ctemp[:]  = np.reshape(viol_cxi,1)


def Distributed_forwardMPC(xq_fb, xl_fb, xq_traj_prev, uq_traj_prev, xl_traj_prev, ul_traj_prev, Ref_xq, Ref_uq, Ref_xl, Ref_ul, Para_q, Para_l, Jl, rg):
    epsilon = 1e-2 # threshold for stopping the iteration
    k_max   = 5 # maximum number of iterations
    max_violation = 5 # initial value of max_violation, defined as the maximum value of the differences between two trajectories in successive iterations for all quadrotors
    k       = 1
    xq_traj = []
    uq_traj = []
    for iq in range(nq):
        xiq_traj = np.zeros((horizon+1,nxi))
        uiq_traj = np.zeros((horizon,nui))
        xi_prev  = xq_traj_prev[iq]
        ui_prev  = uq_traj_prev[iq]
        for iqk in range(horizon):
            xiq_traj[iqk,:] = xi_prev[iqk+1,:] # note that we have moved forward by one time-step, so we take elements from [1,:]
            if iqk <horizon-1:
                uiq_traj[iqk,:] = ui_prev[iqk+1,:]
            else:
                uiq_traj[-1,:] = ui_prev[-1,:]
        xiq_traj[-1,:] = xi_prev[-1,:]
        xq_traj += [xiq_traj]
        uq_traj += [uiq_traj]
    xl_traj = np.zeros((horizon+1,nxl))
    ul_traj = np.zeros((horizon,nul))
    for il in range(horizon):
        xl_traj[il,:] = xl_traj_prev[il+1,:]
        if il <horizon-1:
            ul_traj[il,:] = ul_traj_prev[il+1,:]
        else:
            ul_traj[-1,:] = ul_traj_prev[-1,:]
    xl_traj[-1,:] = xl_traj_prev[-1,:]
    

    while max_violation>=epsilon and k<=k_max:
        viol_xtemp  = []
        viol_utemp  = []
        viol_cxtemp = []
        viol_x      = []
        viol_u      = []
        xq_temp     = [] # temporary list for saving the updated state trajectories during the 'for' loop
        uq_temp     = [] # temporary list for saving the updated control trajectories during the 'for' loop
        cx_quad     = []
        cx_temp     = []
        n_process   = []

        for _ in range(nq):
            xi_traj = Array('d',np.zeros((horizon+1)*nxi))
            ui_traj = Array('d',np.zeros((horizon)*nui))
            ci_traj = Array('d',np.zeros((horizon)*nxi))
            viol_xi = Array('d',np.zeros(1))
            viol_ui = Array('d',np.zeros(1))
            viol_ci = Array('d',np.zeros(1))
            xq_temp.append(xi_traj)
            uq_temp.append(ui_traj)
            cx_temp.append(ci_traj)
            viol_xtemp.append(viol_xi)
            viol_utemp.append(viol_ui)
            viol_cxtemp.append(viol_ci)
        

        for i in range(nq):
            p = Process(target=QuadrotorMPC,args=(xq_fb[i], xq_traj, uq_traj, xl_traj, ul_traj, Ref_xq[i], Ref_uq[i], Para_q[i], i, xq_temp[i], uq_temp[i], cx_temp[i], viol_xtemp[i], viol_utemp[i], viol_cxtemp[i]))
            p.start() 
            n_process.append(p)
        
        for p in n_process: 
            p.join()

        # futures = []
        # with ThreadPoolExecutor(max_workers=nq) as executor:
        #     for i in range(nq):
        #         futures.append(executor.submit(QuadrotorMPC,xq_fb[i], xq_traj, uq_traj, xl_traj, ul_traj, Ref_xq[i], Ref_uq[i], Para_q[i], i, xq_temp[i], uq_temp[i], cx_temp[i], viol_xtemp[i], viol_utemp[i], viol_cxtemp[i]))
        
        # for future in futures:
        #     future.result()
        
        for i in range(nq):
            xi_opt = np.reshape(xq_temp[i],(horizon+1, nxi))
            ui_opt = np.reshape(uq_temp[i],(horizon, nui))
            ci_ipopt = np.reshape(cx_temp[i],(horizon, nxi))
            violxi = np.reshape(viol_xtemp[i],(1))
            violui = np.reshape(viol_utemp[i],(1))
            violci = np.reshape(viol_cxtemp[i],(1))
            # print('iteration=',k,'quadrotor_ID=',i,'viol_xi=',format(violxi[0],'.5f'),'viol_ui=',format(violui[0],'.5f'),'viol_cxi=',format(violci[0],'.5f'))
            xq_traj[i] = xi_opt
            uq_traj[i] = ui_opt
            cx_quad   += [ci_ipopt]
            viol_x    += [violxi[0]]
            viol_u    += [violui[0]]
        

        # solve the MPC of the payload using the updated quadrotor trajectories xq_traj
        opt_sol_l   = DistMPC.MPCsolverPayload(xl_fb, xq_traj, Ref_xl, Ref_ul, Para_l, Jl, rg)
        xl_opt      = opt_sol_l['xl_opt']
        ul_opt      = opt_sol_l['ul_opt']
        cox_opt_l   = opt_sol_l['costatel_traj_opt']
        cox_ipopt_l = opt_sol_l['costatel_ipopt']
        sum_viol_xl = 0
        sum_viol_ul = 0
        sum_viol_cxl= 0
        for kl in range(len(ul_traj)):
            sum_viol_xl  += LA.norm(xl_opt[kl,:]-xl_traj[kl,:])
            sum_viol_ul  += LA.norm(ul_opt[kl,:]-ul_traj[kl,:])
            sum_viol_cxl += LA.norm(cox_opt_l[kl,:]-cox_ipopt_l[kl,:])
        sum_viol_xl  += LA.norm(xl_opt[-1,:]-xl_traj[-1,:])
        viol_xl  = sum_viol_xl/len(xl_opt)
        viol_ul  = sum_viol_ul/len(ul_opt)
        viol_cxl = sum_viol_cxl/len(cox_opt_l)
        viol_x  += [viol_xl]
        viol_u  += [viol_ul]
        # initial_error = LA.norm(np.reshape(xl_opt[0,:],(nxl,1))-xl_fb)
        print('iteration=',k,'payload:','viol_xl=',format(viol_xl,'.5f'),'viol_ul=',format(viol_ul,'.5f'),'viol_cxl=',format(viol_cxl,'.5f'))
        # update the payload's trajectories
        xl_traj  = xl_opt
        ul_traj  = ul_opt

        # compute the maximum violation value
        viol  = np.concatenate((viol_x,viol_u))
        if k>1:
             max_violation = np.max(viol)
        print('iteration=',k,'max_violation=',format(max_violation,'.5f'))
        # update the iteration number
        k += 1
    
    # output
    opt_system = {"xq_traj":xq_traj,
                  "uq_traj":uq_traj,
                  "xl_traj":xl_traj,
                  "ul_traj":ul_traj,
                  "cx_quad":cx_quad,
                  "cx_load":cox_ipopt_l}
        
    return opt_system



def QuadrotorSensitivity(Quads_Mat, Load_Mat, Xl_pl, index, Xi_piD, Xi_plD, Xl_piD, FXi_plD,):
    quad_sensitivity = Sensprop.quadrotor_sensitivity_parallel(Quads_Mat, Load_Mat, Xl_pl, index)
    Xi_pi  = quad_sensitivity['Xi_pi']
    Xi_pl  = quad_sensitivity['Xi_pl']
    Xl_pi  = quad_sensitivity['Xl_pi']
    FXi_pl = quad_sensitivity['FXi_pl']
    Xi_piD[index]  = Xi_pi
    Xi_plD[index]  = Xi_pl
    Xl_piD[index]  = Xl_pi
    FXi_plD[index] = FXi_pl

def Distributed_sensitivity(Quads_Mat, Load_Mat):
    epsilon = 1e-4    # threshold for stopping the iteration, 1e-4 for testing the convergence step, 1e-3 is used in training
    k_max   = 5      # maximum number of iterations
    max_violation = 5 # initial value of max_violation
    k = 1             # iteration index
    # initialization
    Xi_Pi_Prev    = []
    Xi_Pl_Prev    = []
    Xl_Pi_Prev    = []
    for i in range(nq):
        Xi_pi_prev    = (horizon_loss+1)*[np.zeros((nxi, npi))]
        Xi_pl_prev    = (horizon_loss+1)*[np.zeros((nxi, npl))]
        Xl_pi_prev    = (horizon_loss+1)*[np.zeros((nxl, npi))]
        Xi_Pi_Prev   += [Xi_pi_prev]
        Xi_Pl_Prev   += [Xi_pl_prev]
        Xl_Pi_Prev   += [Xl_pi_prev]
        
    Xl_pl         = (horizon_loss+1)*[np.zeros((nxl, npl))] 
    Xl_pl_prev    = (horizon_loss+1)*[np.zeros((nxl, npl))]

    manager       = Manager()
    Xi_piD        = manager.dict()
    Xi_plD        = manager.dict()
    Xl_piD        = manager.dict()
    FXi_plD       = manager.dict()
    k_step        = 0

    while max_violation>=epsilon and k<=k_max:
        sumXi_pl         = (horizon_loss)*[np.zeros((nxl, npl))] # initialize sumXi_pl
        viol_X_list      = []
        All_quad_sens    = [] # list for saving all the updated sensitivity trajectories during the 'for' loop
        n_process        = []
        for i in range(nq):
            ps = Process(target=QuadrotorSensitivity,args=(Quads_Mat, Load_Mat, Xl_pl, i, Xi_piD, Xi_plD, Xl_piD, FXi_plD))
            ps.start()
            n_process.append(ps)
        
        for ps in n_process:
            ps.join()
        
        for i in range(nq):
            Xi_pi  = Xi_piD[i]
            Xi_pl  = Xi_plD[i]
            Xl_pi  = Xl_piD[i]
            FXi_pl = FXi_plD[i]
            All_quad_sens += [Xi_pi]
            sum_Xi_pi_viol = 0
            sum_Xi_pl_viol = 0
            sum_Xl_pi_viol = 0
            sum_Xi_lp_viol = 0
            for t in range(horizon_loss):
                sum_Xi_pi_viol += LA.norm(Xi_pi[t] - Xi_Pi_Prev[i][t], ord=2)
                sum_Xi_pl_viol += LA.norm(Xi_pl[t] - Xi_Pl_Prev[i][t], ord=2)
                sum_Xl_pi_viol += LA.norm(Xl_pi[t] - Xl_Pi_Prev[i][t], ord=2)
                sumXi_pl[t]    = sumXi_pl[t] + FXi_pl[t]
                
            sum_Xi_pi_viol += LA.norm(Xi_pi[-1] - Xi_Pi_Prev[i][-1], ord=2)
            sum_Xi_pl_viol += LA.norm(Xi_pl[-1] - Xi_Pl_Prev[i][-1], ord=2)
            sum_Xl_pi_viol += LA.norm(Xl_pi[-1] - Xl_Pi_Prev[i][-1], ord=2)
            viol_Xi_pi = sum_Xi_pi_viol/(horizon_loss+1)
            viol_Xi_pl = sum_Xi_pl_viol/(horizon_loss+1)
            viol_Xl_pi = sum_Xl_pi_viol/(horizon_loss+1)
            viol_X_list += [viol_Xi_pi]
            viol_X_list += [viol_Xi_pl]
            viol_X_list += [viol_Xl_pi]
            print('sensitivity: iteration-',k,'quadrotor-index=',i,'viol_Xi_pi=',format(viol_Xi_pi,'.5f'),'viol_Xi_pl=',format(viol_Xi_pl,'.5f'),'viol_Xl_pi=',format(viol_Xl_pi,'.5f'))
            # update
            Xi_Pi_Prev[i] = Xi_pi
            Xi_Pl_Prev[i] = Xi_pl
            Xl_Pi_Prev[i] = Xl_pi

        # solve the sensitivity of the payload using the updated quadrotor trajectories sumXi_pl
        Xl_pl  = Sensprop.payload_sensitivity(Load_Mat, sumXi_pl) 
        sum_Xl_pl_viol = 0
        for t in range(horizon_loss+1):
            sum_Xl_pl_viol += LA.norm(Xl_pl[t] - Xl_pl_prev[t], ord=2)
        viol_Xl_pl = sum_Xl_pl_viol/(horizon_loss+1)
        print('sensitivity: iteration-',k,'payload:','viol_Xl_pl=',format(viol_Xl_pl,'.5f'))
        k_step = k
        viol_X_list += [viol_Xl_pl]
        # compute the maximum violation value
        if k>1:
            max_violation = np.max(viol_X_list)
        print('sensitivity: iteration-',k,'max_violation=',format(max_violation,'.5f'))
        # update
        Xl_pl_prev = Xl_pl
        # update the iteration number
        k += 1

    return All_quad_sens, Xl_pl, k_step



"""=========================Evaluation process======================="""
def Evaluate():
    T_end      = 15 # total simulation duration
    N          = int(T_end/dt_sample) # total iterations
    if not os.path.exists("Evaluation results"):
        os.makedirs("Evaluation results")

    # Lists for saving the evaluation results
    Quad_State   = []
    Quad_Control = []
    Load_State   = []
    Tension_Load_Actual = []
    Tension_Load_MPC    = []
    Ref_Quad     = []
    Ref_Load     = []
    TIME         = []
    EULER_l      = np.zeros((3,N))
    STATE_l      = np.zeros((3,N))
    REF_P_l      = np.zeros((3,750))
    # initial values used in the low-pass filter
    sig_f_prev   = 0
    u_prev       = np.array([[uav_para[0]*9.81,0,0,0]]).T
    # initial state prediction
    z_hat        = np.zeros((3,1))
    Sig_f_prev   = []
    U_prev       = []
    Z_hat        = []
    for i in range(nq):
        Sig_f_prev += [sig_f_prev]
        U_prev     += [u_prev]
        Z_hat      += [z_hat]
    
    # Load the quadrotors' network models
    NN_Quad    = []
    for i in range(nq):
        PATH_1 = "trained data/trained_nn_quad_"+str(i)+".pt"
        nn_quad_i = torch.load(PATH_1)
        NN_Quad  += [nn_quad_i]
    # Load the payload's network model
    PATHl_2 = "trained data/trained_nn_load.pt"
    nn_load    = torch.load(PATHl_2)
    
    # Initialization of the system states
    # initial palyload's state
    x0         = np.random.normal(stm.rc,0.01)
    y0         = np.random.normal(0,0.01)
    z0         = np.random.normal(stm.hc,0.01)
    pl         = np.array([[x0,y0,z0]]).T
    vl         = np.reshape(np.random.normal(0,0.01,3),(3,1)) # initial velocity of CO in {Bl}
    Eulerl     = np.reshape(np.random.normal(0,0.01,3),(3,1))
    Rl0        = stm.dir_cosine(Eulerl)
    r          = Rot.from_matrix(Rl0)  
    # quaternion in the format of x, y, z, w 
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html)
    ql0        = r.as_quat() 
    ql         = np.array([[ql0[3], ql0[0], ql0[1], ql0[2]]]).T
    wl         = np.reshape(np.random.normal(0,0.01,3),(3,1))
    # set of the quadrotors' initial states
    Pi         = [] 
    Vi         = [] 
    Qi         = [] 
    Wi         = []
    EULERi     = []
    for i in range(nq):
        rli        = np.array([[(rl+L0*math.sin(angle_t))*math.cos(i*alpha),(rl+L0*math.sin(angle_t))*math.sin(i*alpha),0]]).T
        pai        = pl + Rl0@rli
        pi0        = pai + np.array([[0,0,L0*math.cos(angle_t)]]).T # the cables are assumed to be intially slack
        Pi        += [pi0]
        vi0        = np.reshape(np.random.normal(0,0.01,3),(3,1)) # initial velocity of CoM in {I}
        Vi        += [vi0]
        Euleri0    = np.reshape(np.random.normal(0,0.01,3),(3,1))
        EULERi    += [Euleri0]
        Ri0        = stm.dir_cosine(Euleri0)
        ri0        = Rot.from_matrix(Ri0)
        qi0        = ri0.as_quat() # in the format of x, y, z, w
        qi0        = np.array([[qi0[3], qi0[0], qi0[1], qi0[2]]]).T
        Qi        += [qi0]
        wi0        = np.reshape(np.random.normal(0,0.01,3),(3,1))
        Wi        += [wi0]
                

    # Initial time
    time_traj  = 0 # elipsed time for the flight trajectory
    # flag for selecting the reference trajectory
    flag = 0
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}) # show np.array with 3 decimal places
    # main simulation loop whose running frequency equals 1/dt_sample, which can be larger than 1/dt_ctrl
    k_ctrl       = 0

    # initial values used in the low-pass filter
    sig_f_prev   = 0
    u_prev       = np.array([[uav_para[0]*9.81,0,0,0]]).T

    # initial state prediction
    z_hat        = np.zeros((3,1))

    Sig_f_prev   = []
    U_prev       = []
    Z_hat        = []
    for i in range(nq):
        Sig_f_prev += [sig_f_prev]
        U_prev     += [u_prev]
        Z_hat      += [z_hat]
    df           = np.zeros((3,1))
    for k in range(N): 
        # payload's state
        xl           = np.vstack((pl, vl, ql, wl))
        # quadrotors' current states 
        xq           = []
        for i in range(nq):
            xq       += [np.vstack((Pi[i],Vi[i],Qi[i],Wi[i]))]
        Quad_State  += [xq]
        Load_State  += [xl]
        TIME        += [time_traj]
        EULER_l[:,k:k+1]= Eulerl 
        # contorl loop whose running frequency equals 1/dt_ctrl
        if (k%ratio)==0:    
            #============================= Forward Path (Distributed MPC) ==============================#
            # compute reference trajectories for MPC
            Ref_xq, Ref_uq, Ref_xl, Ref_ul, Ref0_xq, Ref0_l = Reference_for_MPC(time_traj)
            Ref_Quad  += [Ref0_xq]
            Ref_Load  += [Ref0_l]
            STATE_l[:,k_ctrl:k_ctrl+1] = np.reshape(xl[0:3,0],(3,1))
            REF_P_l[:,k_ctrl:k_ctrl+1] = np.reshape(Ref0_l[0:3,0],(3,1))
            if flag ==0: # initialize the trajectories using the reference trajectories
                xq_traj, uq_traj = [],[]
                ul_traj = np.zeros((horizon,nul))
                for ki in range(len(Ref_uq)):
                    xq_traj  += [Ref_xq[ki].T]
                    uq_traj  += [Ref_uq[ki].T]
                xq_traj += [Ref_xq[-1].T]
                xl_traj  = Ref_xl.T
                ul_traj  = Ref_ul.T
                flag = 1
            # generate adaptive MPC's weightings online through the networks
            CTRL_gain       = [] # control gain list for all the quadrotors
            NN_input_quad   = [] # network input list for all the quadrotors
            NN_output_quad  = [] # network output list for all the quadrotors
                
            for i in range(nq):
                track_e_i   = xq[i][0:6,0]-Ref0_xq[i][0:6,0]
                input_i     = np.reshape(track_e_i,(Di_in,1))
                nn_i_output = convert_quadrotor_nn(NN_Quad[i](input_i))
                weight_i    = SetPara_quadrotor(nn_i_output)
                print('time step=',k,'ctrl step=',k_ctrl,'quadrotor:',i,'Qi_k=',weight_i[0,0:3],'Ri_k=',weight_i[0,2*nwsi:])
                NN_input_quad  += [input_i]
                NN_output_quad += [nn_i_output]
                   
                CTRL_gain  += [weight_i]
            pl_error     = np.reshape(xl[0:3,0] - Ref0_l[0:3,0],(3,1))
            vl_error     = np.reshape(xl[3:6,0] - Ref0_l[3:6,0],(3,1))
            ql           = xl[6:10,0]
            qlref        = Ref0_l[6:10,0]
            Rl           = stm.q_2_rotation(ql,1)
            Rlref        = stm.q_2_rotation(qlref,1)
            error_Rl     = Rlref.T@Rl - Rl.T@Rlref
            att_error_l  = 1/2*stm.vee_map(error_Rl)
            w_error_l    = np.reshape(xl[10:13,0] - Ref0_l[10:13,0],(3,1))
            track_e_l    = np.vstack((pl_error,vl_error,att_error_l,w_error_l))
            input_l      = np.reshape(track_e_l,(Dl_in,1))
            nn_l_output  = convert_load_nn(nn_load(input_l))
            Para_l       = SetPara_load(nn_l_output)
                
            print('time step=',k,'payload=','Ql_k[0:3]=',Para_l[0,0:3],'Ql_k[6:9]=',Para_l[0,6:9])
            print('time step=',k,'payload=','Ql_N[0:3]=',Para_l[0,12:15],'Rl_k=',Para_l[0,2*nwsl:],'ctrlmode:',ctrlmode)
            # solve the distributed MPC to generate optimal state and control trajectories
            if ctrlmode == 's':
                opt_system   = DistMPC.Distributed_forwardMPC(xq,xl,xq_traj,uq_traj,xl_traj,ul_traj,
                                                              Ref_xq,Ref_uq,Ref_xl,Ref_ul,CTRL_gain,Para_l,Jl,rg)
            else:
                opt_system   = Distributed_forwardMPC(xq,xl,xq_traj,uq_traj,xl_traj,ul_traj,
                                                              Ref_xq,Ref_uq,Ref_xl,Ref_ul,CTRL_gain,Para_l,Jl,rg)   
            xq_traj      = opt_system['xq_traj']
            uq_traj      = opt_system['uq_traj']
            xl_traj      = opt_system['xl_traj']
            ul_traj      = opt_system['ul_traj']
            cx_quad_traj = opt_system['cx_quad']
            cx_load_traj = opt_system['cx_load']

            # robustify the MPC control using L1 adaptive control (piecewise-constant adaptation law)
            Uad_lpf      = [] # list of the filtered estimation of the matched disturbance 
            Dm, Dum      = [], [] # list of the mathced and unmatched disturbances
            for i in range(nq):
                xi       = xq[i]
                z_hat    = Z_hat[i]
                dm_hat, dum_hat, A_s = GeoCtrl.L1_adaptive_law(xi,z_hat)
                Dm      += [dm_hat]
                Dum     += [dum_hat]
                # Low-pass filter
                wf_coff  = 15 
                time_constf = 1/wf_coff
                f_prev   = Sig_f_prev[i]
                f_lpf    = GeoCtrl.lowpass_filter(time_constf,dm_hat,f_prev)
                Uad_lpf += [f_lpf]
        
        #============================= Interaction with simulation environment ==============================#
        ul           = np.zeros((nul,1)) # the payload's control that contains all the cable forces
        uq           = [] # the collection of all the quadrotors' control
        for i in range(nq):
            # compute the cable forces
            xi           = Quad_State[-1][i]
            f_t          = stm.ith_cable_force(xi, xl, i) 
            ul[i,0]      = f_t
            # only the first control command is applied to the system
            ui           = np.reshape(uq_traj[i][0,:],(nui,1)) 
            # robustify the nominal control using the L1-AC compensation
            ui[0,0]     += -Uad_lpf[i]
            uq          += [ui]
        if (k%ratio) == 0:
            # update the state prediction in L1-AC
            for i in range(nq):
                z_hat    = Z_hat[i]
                xi       = xq[i]
                ui       = uq[i]
                ti       = ul_traj[0,i]
                dm       = Dm[i]
                dum      = Dum[i]
                z_hatnew = stm.predictor_L1(z_hat, xi, ui, xl, ti, dm, dum, A_s, i, dt_ctrl)
                Z_hat[i] = z_hatnew
                
        Quad_Control += [uq]
        Tension_Load_Actual += [ul]
        Tension_Load_MPC    += [ul_traj[0,:]]
        print('k=',k,'Tension =',ul.T,'ul=',ul_traj[0,:])
        print('k=',k,'Tension difference =',ul_traj[0,:]-ul.T,'disturbance =',df.T,'L1-AC estimation =',np.reshape(Uad_lpf,(1,nul)))
        print('k=',k,'ref_pl=',Ref0_l[0:3,0].T,'pl=',pl.T,'Eulerl=',np.reshape(57.3*Eulerl,(3)))
        for i in range(nq):
            print('k=',k,'quadrotor:',i,'ref_p=',Ref0_xq[i][0:3,0].T,'p=',Quad_State[-1][i][0:3,0].T,'Euler=',np.reshape(57.3*EULERi[i],(3)))
            print('k=',k,'quadrotor:',i,'ctrl=',uq[i].T)
            

        # update the system states using the 'step' function
        xq           = np.zeros((nxi,nq))
        for i in range(nq):
            xq[:,i:i+1]  = Quad_State[-1][i]
        output_l     = stm.step_load(xl, ul, xq, Jl, rg, dt_sample, df) # ul
        pl           = output_l['pl_new']
        vl           = output_l['vl_new']
        ql           = output_l['ql_new']
        wl           = output_l['wl_new']
        Eulerl       = output_l['Euler_l_new']

        for i in range(nq):
            xi           = Quad_State[-1][i]
            ui           = uq[i]
            # pass the control signal through a low-pass filter to simulate the time-delay effect caused by the motor dynamics
            wf_coff      = 33 # same as that used in 'NeuroBEM' paper 
            time_constf  = 1/wf_coff
            u_prev       = U_prev[i]
            ui_lpf       = GeoCtrl.lowpass_filter(time_constf,ui,u_prev)
            U_prev[i]    = ui_lpf
            output_i     = stm.step_quadrotor(xi, ui_lpf, xl, ul[i,0], i, dt_sample, df) 
            pi           = output_i['pi_new']
            vi           = output_i['vi_new']
            qi           = output_i['qi_new']
            wi           = output_i['wi_new']
            Euleri       = output_i['Euler_i_new']
            Pi[i]        = pi
            Vi[i]        = vi
            Qi[i]        = qi
            Wi[i]        = wi
            EULERi[i]    = Euleri
        # update the external disturbance
        df          = stm.disturbance(df, dt_sample)
        time_traj  += dt_sample
   
        
        # save the evaluation results
        np.save('Evaluation results/Quad_State',Quad_State)
        np.save('Evaluation results/Quad_Control',Quad_Control)
        np.save('Evaluation results/Load_State',Load_State)
        np.save('Evaluation results/Tension_Load_Actual',Tension_Load_Actual)
        np.save('Evaluation results/Tension_Load_MPC',Tension_Load_MPC)
        np.save('Evaluation results/TIME',TIME)
        np.save('Evaluation results/EULERl',EULER_l)
        np.save('Evaluation results/Ref_Load',Ref_Load)

    
    if not os.path.exists("plots_test"):
        os.makedirs("plots_test")
    # plotting
    plt.figure(2,dpi=400)
    for i in range(nq):
        Tension_i_actual = []
        for k in range(N):
            Tension_i_actual += [Tension_Load_Actual[k][i,0]]
        plt.plot(TIME,Tension_i_actual,linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Actual tension force [N]')
    plt.legend(['Cable0', 'Cable1', 'Cable2', 'Cable3'])
    plt.grid()
    plt.savefig('plots_test/cable_actual_tensions_circle.png',dpi=400)
    plt.show()

    plt.figure(3,dpi=400)
    for i in range(nq):
        Tension_i_MPC = []
        for k in range(N):
            Tension_i_MPC += [Tension_Load_MPC[k][i]]
        plt.plot(TIME,Tension_i_MPC,linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('MPC tension force [N]')
    plt.legend(['Cable0', 'Cable1', 'Cable2', 'Cable3'])
    plt.grid()
    plt.savefig('plots_test/cable_MPC_tensions_circle.png',dpi=400)
    plt.show()

    plt.figure(4,dpi=400)
    plt.plot(TIME,EULER_l[0,:],linewidth=1)
    plt.plot(TIME,EULER_l[1,:],linewidth=1)
    plt.plot(TIME,EULER_l[2,:],linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Payload attitude [deg]')
    plt.legend(['roll', 'pitch', 'yaw'])
    plt.grid()
    plt.savefig('plots_test/payload_attitude_MPCcontrol_circle.png',dpi=400)
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3,sharex=True, dpi=400)
    pxl,  pyl,  pzl   = [], [], []
    refxl,refyl,refzl = [], [], []
    t_ref = []
    kref  = 0
    for k in range(N):
        if (k%ratio)==0:  
            pxl   += [Load_State[k][0,0]]
            pyl   += [Load_State[k][1,0]]
            pzl   += [Load_State[k][2,0]]
            refxl += [Ref_Load[kref][0,0]]
            refyl += [Ref_Load[kref][1,0]]
            refzl += [Ref_Load[kref][2,0]]
            t_ref += [TIME[k]]
            kref  += 1
    ax1.plot(t_ref,pxl,linewidth=1)
    ax1.plot(t_ref,refxl,linewidth=1)
    ax2.plot(t_ref,pyl,linewidth=1)
    ax2.plot(t_ref,refyl,linewidth=1)
    ax3.plot(t_ref,pzl,linewidth=1)
    ax3.plot(t_ref,refzl,linewidth=1)
    ax1.set_ylabel('Payload x [m]',labelpad=0)
    ax2.set_ylabel('Payload y [m]',labelpad=0)
    ax3.set_ylabel('Payload z [m]',labelpad=0)
    ax3.set_xlabel('TIme [s]',labelpad=0)
    ax1.tick_params(axis='x',which='major',pad=0,length=1)
    ax1.tick_params(axis='y',which='major',pad=0,length=1)
    ax2.tick_params(axis='x',which='major',pad=0,length=1)
    ax2.tick_params(axis='y',which='major',pad=0,length=1)
    ax3.tick_params(axis='x',which='major',pad=0,length=1)
    ax3.tick_params(axis='y',which='major',pad=0,length=1)
    leg=ax1.legend(['Actual','Desired'],loc='upper center')
    leg.get_frame().set_linewidth(0.5)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.savefig('plots_test/payload_position_MPCcontrol_circle.png',dpi=400)
    plt.show()

    plt.figure(6,dpi=400)
    ax = plt.axes(projection="3d")
    ax.plot3D(STATE_l[0,:], STATE_l[1,:], STATE_l[2,:], linewidth=1.5)
    ax.plot3D(REF_P_l[0,:], REF_P_l[1,:], REF_P_l[2,:], linewidth=1, linestyle='--')
    plt.legend(['Actual', 'Desired'])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.savefig('plots_test/payload_3D_MPCcontrol_circle.png',dpi=400)
    plt.show()


"""=========================Main function======================="""

if __name__ == '__main__':

    if mode != "t":
        Evaluate()

    T_end      = stm.Tc # total simulation duration, 10s for hovering
    N          = int(T_end/dt_sample) # total iterations
    if not os.path.exists("trained data"):
        os.makedirs("trained data")

    # Lists for saving the training results
    Quad_State_s   = [] # sampled state for control
    Quad_Control   = []
    Load_State_s   = [] # sampled state for control
    Quad_State     = []
    Load_State     = []
    EULER_l        = []
    Tension_Load_Actual = []
    Tension_Load_MPC    = []
    TIME_train   = []
    Quads_Mat_s  = [] # sampled for control
    Load_Mat_s   = [] # sampled for control
    Ref_Quad_s   = [] # sampled reference for control
    Ref_Load_s   = [] # sampled reference for control
    Loss         = []
    K_Train      = []
    Loss_episode = []
    # load_openloop = np.load('trained data/Loss_episode.npy')
    # Quad_state0    = np.load('trained data/Quad_State.npy')
    # Load_state0    = np.load('trained data/Load_State.npy')
    # Initial threshold for stopping the training
    eps        = 1e1
    # Initial change of the high-level loss
    delta_loss = 1e10

    # Training iteration index
    k_train    = 0
    
    # Initialization of the quadrotors' network models
    NN_Quad    = []
    for i in range(nq):
        PATH_init = "trained data/initial_nn_quad_"+str(i)
        nn_quad_i = NeuralNet.Net(Di_in,Di_h,Di_out)
        # torch.save(nn_quad_i,PATH_init)
        NN_Quad  += [nn_quad_i]
    # Initialization of the payload's network model
    PATHl_init = "trained data/initial_nn_load.pt"
    nn_load    = NeuralNet.Net(Dl_in,Dl_h,Dl_out)
    # torch.save(nn_load,PATHl_init)

    # Use the saved initial models for training
    # NN_Quad    = []
    # for i in range(nq):
    #     PATH_init = "trained data/trained_nn_quad_"+str(i)+".pt"
    #     nn_quad_i = torch.load(PATH_init)
    #     NN_Quad  += [nn_quad_i]
    # PATHl_init = "trained data/trained_nn_load.pt"
    # nn_load    = torch.load(PATHl_init)
    
  
    time_train = 0 # elipsed time in training, which can be longer than time_traj if k_train >1
    flag_loss  = 0
    flag_train = 0
    Sens_time  = []
    Sens_step  = []
    MPC_time   = []

    loadpG     = []
    angle_t = np.pi/9

    # while delta_loss>=eps:
    for k_train in range(1):
        
        # if k_train>1: # currculum learning strategy
        #     if delta_loss <= mean_loss/5e2:
        #         angle_t = np.pi/4
        #     elif delta_loss <= mean_loss/5e1:
        #         angle_t = np.pi/6
    
        if k_train <1:
            # Initialization of the system states
            # initial palyload's state
            x0         = np.random.normal(stm.rc,0.01)
            y0         = np.random.normal(0,0.01)
            z0         = np.random.normal(stm.hc,0.01) # large vertical error (-stm.hc) for testing the convergence of the DSP algorithm, stm.hc for training
            pl         = np.array([[x0,y0,z0]]).T
            
            vl         = np.reshape(np.random.normal(0,0.01,3),(3,1)) # initial velocity of CO in {Bl}
            Eulerl     = np.reshape(np.random.normal(0,0.01,3),(3,1))
            Rl0        = stm.dir_cosine(Eulerl)
            r          = Rot.from_matrix(Rl0)  
            # quaternion in the format of x, y, z, w 
            # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html)
            ql0        = r.as_quat() 
            ql         = np.array([[ql0[3], ql0[0], ql0[1], ql0[2]]]).T
            wl         = np.reshape(np.random.normal(0,0.01,3),(3,1))


            # xl         = Load_state0[-1]
            # pl         = np.reshape(xl[0:3,0],(3,1))
            # vl         = np.reshape(xl[3:6,0],(3,1))
            # ql         = np.reshape(xl[6:10,0],(4,1))
            # wl         = np.reshape(xl[10:13,0],(3,1))
            # Rl0        = stm.q_2_rotation(ql,1)
            # gamma_l    = np.arctan(Rl0[2, 1]/Rl0[1, 1])
            # theta_l    = np.arctan(Rl0[0, 2]/Rl0[0, 0])
            # psi_l      = np.arcsin(-Rl0[0, 1])
            # Eulerl     = np.array([[gamma_l, theta_l, psi_l]]).T
            # set of the quadrotors' initial states
            Pi         = [] 
            Vi         = [] 
            Qi         = [] 
            Wi         = []
            EULERi     = []
            for i in range(nq):
                rli        = np.array([[(rl+L0*math.sin(angle_t))*math.cos(i*alpha),(rl+L0*math.sin(angle_t))*math.sin(i*alpha),0]]).T
                pai        = pl + Rl0@rli
                pi0        = pai + np.array([[0,0,L0*math.cos(angle_t)]]).T # the cables are assumed to be intially slack
                vi0        = np.reshape(np.random.normal(0,0.01,3),(3,1)) # initial velocity of CoM in {I}
                Euleri0    = np.reshape(np.random.normal(0,0.01,3),(3,1))
                EULERi    += [Euleri0]
                Ri0        = stm.dir_cosine(Euleri0)
                ri0        = Rot.from_matrix(Ri0)
                qi0        = ri0.as_quat() # in the format of x, y, z, w
                qi0        = np.array([[qi0[3], qi0[0], qi0[1], qi0[2]]]).T
                wi0        = np.reshape(np.random.normal(0,0.01,3),(3,1))
                # xi0        = Quad_state0[-1][i]
                # pi0        = np.reshape(xi0[0:3,0],(3,1))
                # vi0        = np.reshape(xi0[3:6,0],(3,1))
                # qi0        = np.reshape(xi0[6:10,0],(4,1))
                # wi0        = np.reshape(xi0[10:13,0],(3,1))
                # Ri0        = stm.q_2_rotation(ql,1)
                # gamma_i    = np.arctan(Ri0[2, 1]/Ri0[1, 1])
                # theta_i    = np.arctan(Ri0[0, 2]/Ri0[0, 0])
                # psi_i      = np.arcsin(-Ri0[0, 1])
                # Euler_i    = np.array([[gamma_i, theta_i, psi_i]]).T
                Pi        += [pi0]
                Vi        += [vi0]
                Qi        += [qi0]
                Wi        += [wi0]
                # EULERi    += [Euleri0]



        TIME_traj  = []
        # Initial time
        time_traj  = 0 # elipsed time for the flight trajectory
        # flag for selecting the reference trajectory
        flag = 0
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}) # show np.array with 3 decimal places
        # main simulation loop whose running frequency equals 1/dt_sample, which can be larger than 1/dt_ctrl
        k_ctrl       = 0
        sum_loss     = 0 # used to compute the mean_loss for one episode

        # initial values used in the low-pass filter
        sig_f_prev   = 0
        u_prev       = np.array([[uav_para[0]*9.81,0,0,0]]).T

        # initial state prediction
        z_hat        = np.zeros((3,1))

        Sig_f_prev   = []
        U_prev       = []
        Z_hat        = []
        for i in range(nq):
            Sig_f_prev += [sig_f_prev]
            U_prev     += [u_prev]
            Z_hat      += [z_hat]
        
        for k in range(600): 
            # payload's state
            xl           = np.vstack((pl, vl, ql, wl))
            # quadrotors' current states 
            xq           = []
            for i in range(nq):
                xq       += [np.vstack((Pi[i],Vi[i],Qi[i],Wi[i]))]
            Quad_State  += [xq]
            Load_State  += [xl]
            TIME_traj   += [time_traj]
            EULER_l     += [Eulerl]
            # contorl loop whose running frequency equals 1/dt_ctrl
            if (k%ratio)==0: 
                Quad_State_s  += [xq]
                Load_State_s  += [xl]
                
                #============================= Forward Path (Distributed MPC) ==============================#
                # compute reference trajectories for MPC
                Ref_xq, Ref_uq, Ref_xl, Ref_ul, Ref0_xq, Ref0_l = Reference_for_MPC(time_traj, angle_t)
                Ref_Quad_s  += [Ref0_xq]
                Ref_Load_s  += [Ref0_l]
                if flag ==0: # initialize the trajectories using the reference trajectories
                    xq_traj, uq_traj = [],[]
                    ul_traj = np.zeros((horizon,nul))
                    for ki in range(len(Ref_uq)):
                        xq_traj  += [Ref_xq[ki].T]
                        uq_traj  += [Ref_uq[ki].T]
                    xq_traj += [Ref_xq[-1].T]
                    xl_traj  = Ref_xl.T
                    ul_traj  = Ref_ul.T
                    flag = 1
                # generate adaptive MPC's weightings online through the networks
                CTRL_gain       = [] # control gain list for all the quadrotors
                NN_input_quad   = [] # network input list for all the quadrotors
                NN_output_quad  = [] # network output list for all the quadrotors
                
                for i in range(nq):
                    track_e_i   = xq[i][0:6,0]-Ref0_xq[i][0:6,0]
                    input_i     = np.reshape(track_e_i,(Di_in,1))
                    nn_i_output = convert_quadrotor_nn(NN_Quad[i](input_i))
                    weight_i    = SetPara_quadrotor(nn_i_output)
                    print('k_train=',k_train,'time step=',k,'ctrl step=',k_ctrl,'quadrotor:',i,'Qi_k=',weight_i[0,0:3],'Ri_k=',weight_i[0,2*nwsi:])
                    NN_input_quad  += [input_i]
                    NN_output_quad += [nn_i_output]
                   
                    CTRL_gain  += [weight_i]
                pl_error     = np.reshape(xl[0:3,0] - Ref0_l[0:3,0],(3,1))
                vl_error     = np.reshape(xl[3:6,0] - Ref0_l[3:6,0],(3,1))
                ql           = xl[6:10,0]
                qlref        = Ref0_l[6:10,0]
                Rl           = stm.q_2_rotation(ql,1)
                Rlref        = stm.q_2_rotation(qlref,1)
                error_Rl     = Rlref.T@Rl - Rl.T@Rlref
                att_error_l  = 1/2*stm.vee_map(error_Rl)
                w_error_l    = np.reshape(xl[10:13,0] - Ref0_l[10:13,0],(3,1))
                track_e_l    = np.vstack((pl_error,vl_error,att_error_l,w_error_l))
                input_l      = np.reshape(track_e_l,(Dl_in,1))
                nn_l_output  = convert_load_nn(nn_load(input_l))
                Para_l       = SetPara_load(nn_l_output)
                
                print('k_train=',k_train,'time step=',k,'payload=','Ql_k[0:3]=',Para_l[0,0:3],'Ql_k[6:9]=',Para_l[0,6:9])
                print('k_train=',k_train,'time step=',k,'payload=','Ql_N[0:3]=',Para_l[0,12:15],'Rl_k=',Para_l[0,2*nwsl:],'ctrlmode:',ctrlmode)
                # solve the distributed MPC to generate optimal state and control trajectories
                if ctrlmode == 's':
                    start_time = TM.time()
                    opt_system   = DistMPC.Distributed_forwardMPC(xq,xl,xq_traj,uq_traj,xl_traj,ul_traj,
                                                              Ref_xq,Ref_uq,Ref_xl,Ref_ul,CTRL_gain,Para_l,Jl,rg)
                    mpctime = (TM.time() - start_time)*1000
                    print("s:--- %s ms ---" % format(mpctime,'.2f'))
                else:
                    start_time = TM.time()
                    opt_system   = Distributed_forwardMPC(xq,xl,xq_traj,uq_traj,xl_traj,ul_traj,
                                                              Ref_xq,Ref_uq,Ref_xl,Ref_ul,CTRL_gain,Para_l,Jl,rg)   
                    mpctime = (TM.time() - start_time)*1000
                    print("p:--- %s ms ---" % format(mpctime,'.2f'))
                MPC_time += [mpctime]
                xq_traj      = opt_system['xq_traj']
                uq_traj      = opt_system['uq_traj']
                xl_traj      = opt_system['xl_traj']
                ul_traj      = opt_system['ul_traj']
                cx_quad_traj = opt_system['cx_quad']
                cx_load_traj = opt_system['cx_load']

                # robustify the MPC control using L1 adaptive control (piecewise-constant adaptation law)
                Uad_lpf      = [] # list of the filtered estimation of the matched disturbance 
                Dm, Dum      = [], [] # list of the mathced and unmatched disturbances
                for i in range(nq):
                    xi       = xq[i]
                    z_hat    = Z_hat[i]
                    dm_hat, dum_hat, A_s = GeoCtrl.L1_adaptive_law(xi,z_hat)
                    Dm      += [dm_hat]
                    Dum     += [dum_hat]
                    # Low-pass filter
                    wf_coff  = 15
                    time_constf = 1/wf_coff
                    f_prev   = Sig_f_prev[i]
                    f_lpf    = GeoCtrl.lowpass_filter(time_constf,dm_hat,f_prev)
                    Uad_lpf += [f_lpf]

                if trainmode == 'c':
                    #============================= Backward Path (Distributed learning) ==============================#
                    # compute the gradients of the system's MPC trajectories w.r.t the hyperparameters (using the PDP method)
                    Quad_k_Mat    = []
                    Load_k_Mat    = []
                    for i in range(nq):
                        auxSys_i    = DistMPC.GetAuxSys_quadrotor(i,xq_traj,uq_traj,xl_traj,ul_traj,cx_quad_traj,Ref_xq,Ref_uq,CTRL_gain)
                        quadi_mat   = MPCgrad.Gradient_solver_quadrotor(auxSys_i)
                        auxSys_li   = DistMPC.GetAuxSys_payload(i,xl_traj,ul_traj,xq_traj,cx_load_traj,Ref_xl,Ref_ul,Para_l,loadp)
                        loadi_mat   = MPCgrad.Gradient_solver_payload_2_quadi(auxSys_li)
                        Quad_k_Mat += [quadi_mat]
                        Load_k_Mat += [loadi_mat]
                    Quads_Mat_s    += [Quad_k_Mat]
                    Load_Mat_s     += [Load_k_Mat]
                else:
                    #============================= Backward Path (Openloop learning) ==============================#
                    Xq_p          = []
                    for i in range(nq):
                        auxSys_i    = DistMPC.GetAuxSys_quadrotor(i,xq_traj,uq_traj,xl_traj,ul_traj,cx_quad_traj,Ref_xq,Ref_uq,CTRL_gain)
                        Xi_p        = MPCgrad.Gradient_solver_quadrotor_openloop(auxSys_i)
                        Xq_p       += [Xi_p]
                    auxSys_li = DistMPC.GetAuxSys_payload(i,xl_traj,ul_traj,xq_traj,cx_load_traj,Ref_xl,Ref_ul,Para_l,loadp)
                    Xl_pl = MPCgrad.Gradient_solver_payload_openloop(auxSys_li)
                
                # train the networks
                if k_ctrl>=horizon_loss and k_train<1:
                    flag_train = 1
                if flag_train == 1:
                    TIME_train  += [time_train]
                    # compute the system's sensitivity propagation for online learning
                    start_time = TM.time()
                    if sensmode == 's':
                        All_quad_sens, Xl_pl, k_step = Sensprop.Distributed_sensitivity(Quads_Mat_s,Load_Mat_s)
                    else:
                        All_quad_sens, Xl_pl, k_step = Distributed_sensitivity(Quads_Mat_s,Load_Mat_s)
                    senstime = (TM.time() - start_time)*1000
                    print("--- %s ms ---" % format(senstime,'.2f'),'k_step=',k_step)
                    Sens_time += [senstime]
                    Sens_step += [k_step]
                    system_loss = 0 # the total loss at the time step k, system_loss = sum of loss_i + loss_l
                    # train the quadrotors' networks
                    for i in range(nq):
                        dlidw, loss_i = Sensprop.ChainRule_quadrotor_i(i,Quad_State_s,Ref_Quad_s,All_quad_sens)
                        system_loss  += loss_i
                        weight_i_grad = chainRule_gradient_quad(NN_output_quad[i])
                        dlidp         = dlidw@weight_i_grad # gradient of the loss_i w.r.t the i-th network's output
                        norm_grad_i   = LA.norm(dlidp)
                        print('k_train=',k_train,'time step=',k,'loss_i=',loss_i,'norm_grad_i=',format(norm_grad_i,'.4f'))
                        loss_nn_i     = NN_Quad[i].myloss(NN_Quad[i](NN_input_quad[i]),dlidp)
                        optimizer     = torch.optim.Adam(NN_Quad[i].parameters(),lr=lr_nn)
                        NN_Quad[i].zero_grad()
                        loss_nn_i.backward()
                        optimizer.step()
                    # train the payload's network
                    dlldw, loss_l = Sensprop.ChainRule_load(Load_State_s,Ref_Load_s,Xl_pl)
                    system_loss  += loss_l
                    weight_l_grad = chainRule_gradient_load(nn_l_output)
                    dlldp         = dlldw@weight_l_grad # gradient of the loss_l w.r.t the payload network's output
                    norm_grad_l   = LA.norm(dlldp)
                    print('k_train=',k_train,'time step=',k,'loss_l=',loss_l,'norm_grad_l=',format(norm_grad_l,'.4f'),'sensmode:',sensmode)
                    loss_nn_l     = nn_load.myloss(nn_load(input_l),dlldp)
                    optimizer_l   = torch.optim.Adam(nn_load.parameters(),lr=lr_nn)
                    nn_load.zero_grad()
                    loss_nn_l.backward()
                    optimizer_l.step()
                    Loss         += [int(system_loss)]
                    sum_loss     += system_loss
                    

                if k_ctrl > 2*horizon_loss:
                    print('k_train=',k_train,'ctrl step=',k_ctrl,'Loss[-1]=',Loss[-1],'Loss[-2]=',Loss[-2],'Loss[-10]=',Loss[-10],'Loss[-20]=',Loss[-20])
                    if k_train >0:
                        print('k_train=',k_train,'Loss_episode=',Loss_episode)
                        
                k_ctrl += 1
            #============================= Interaction with simulation environment ==============================#
            ul           = np.zeros((nul,1)) # the payload's control that contains all the cable forces
            uq           = [] # the collection of all the quadrotors' control
            for i in range(nq):
                # compute the cable forces
                xi           = Quad_State[-1][i]
                f_t          = stm.ith_cable_force(xi, xl, i) 
                ul[i,0]      = f_t
                # only the first control command is applied to the system
                ui           = np.reshape(uq_traj[i][0,:],(nui,1)) 
                # robustify the nominal control using the L1-AC compensation
                ui[0,0]     += -Uad_lpf[i]
                uq          += [ui]
            if (k%ratio) == 0:
                # update the state prediction in L1-AC
                for i in range(nq):
                    z_hat    = Z_hat[i]
                    xi       = xq[i]
                    ui       = uq[i]
                    ti       = ul_traj[0,i]
                    dm       = Dm[i]
                    dum      = Dum[i]
                    z_hatnew = stm.predictor_L1(z_hat, xi, ui, xl, ti, dm, dum, A_s, i, dt_ctrl)
                    Z_hat[i] = z_hatnew
                
            Quad_Control += [uq]
            Tension_Load_Actual += [ul]
            Tension_Load_MPC    += [ul_traj[0,:]]
            print('k=',k,'Tension =',ul.T,'ul=',ul_traj[0,:])
            print('k=',k,'Tension difference =',ul_traj[0,:]-ul.T,'L1-AC estimation =',np.reshape(Uad_lpf,(1,nul)))
            print('k=',k,'ref_pl=',Ref0_l[0:3,0].T,'angle_t=',format(angle_t*57.3,'.3f'),'pl=',pl.T,'Eulerl=',np.reshape(57.3*Eulerl,(3)))
            for i in range(nq):
                print('k=',k,'quadrotor:',i,'ref_p=',Ref0_xq[i][0:3,0].T,'p=',Quad_State[-1][i][0:3,0].T,'Euler=',np.reshape(57.3*EULERi[i],(3)))
                print('k=',k,'quadrotor:',i,'ctrl=',uq[i].T)
            

            # update the system states using the 'step' function
            xq           = np.zeros((nxi,nq))
            for i in range(nq):
                xq[:,i:i+1]  = Quad_State[-1][i]
            output_l     = stm.step_load(xl, ul, xq, Jl, rg, dt_sample) # ul
            pl           = output_l['pl_new']
            vl           = output_l['vl_new']
            ql           = output_l['ql_new']
            wl           = output_l['wl_new']
            Eulerl       = output_l['Euler_l_new']

            for i in range(nq):
                xi           = Quad_State[-1][i]
                ui           = uq[i]
                # pass the control signal through a low-pass filter to simulate the time-delay effect caused by the motor dynamics
                wf_coff      = 33 # same as that used in 'NeuroBEM' paper 
                time_constf  = 1/wf_coff
                u_prev       = U_prev[i]
                ui_lpf       = GeoCtrl.lowpass_filter(time_constf,ui,u_prev)
                U_prev[i]    = ui_lpf
                output_i     = stm.step_quadrotor(xi, ui_lpf, xl, ul[i,0], i, dt_sample) 
                pi           = output_i['pi_new']
                vi           = output_i['vi_new']
                qi           = output_i['qi_new']
                wi           = output_i['wi_new']
                Euleri       = output_i['Euler_i_new']
                Pi[i]        = pi
                Vi[i]        = vi
                Qi[i]        = qi
                Wi[i]        = wi
                EULERi[i]    = Euleri
            # update the external disturbance
            time_traj  += dt_sample
            time_train += dt_sample
            
        
            # save the trained network models
            # for i in range(nq):
            #     PATH1   = "trained data/trained_nn_quad_"+str(i)+".pt"
            #     torch.save(NN_Quad[i],PATH1)
            # PATH2   = "trained data/trained_nn_load.pt"
            # torch.save(nn_load,PATH2)
            # save the trained data
            # np.save('trained data/Quad_State',Quad_State)
            # np.save('trained data/Quad_Control',Quad_Control)
            # np.save('trained data/Load_State',Load_State)
            # np.save('trained data/Load_EULER',EULER_l)
            # np.save('trained data/Quad_State_s',Quad_State_s)
            # np.save('trained data/Load_State_s',Load_State_s)
            # np.save('trained data/Ref_Quad_s',Ref_Quad_s)
            # np.save('trained data/Ref_Load_s',Ref_Load_s)
            # np.save('trained data/Tension_Load_Actual',Tension_Load_Actual)
            # np.save('trained data/Tension_Load_MPC',Tension_Load_MPC)
            # np.save('trained data/TIME_train',TIME_train)
            # np.save('trained data/TIME_traj',TIME_traj)
            # np.save('trained data/Loss',Loss)
            np.save('trained data/SensCPUTime_s_15quad',Sens_time)
            np.save('trained data/SensStep_s_15quad',Sens_step)
            # if ctrlmode == 's':
            #     np.save('trained data/MPCtime_s',MPC_time)
            # else:
            #     np.save('trained data/MPCtime_p',MPC_time)
            # np.save('trained data/LoadPG',loadpG)
        if k_train == 0:
            mean_loss     = int(sum_loss/(N/ratio-horizon_loss))
        else:
            mean_loss     = int(sum_loss/(N/ratio))
        Loss_episode += [mean_loss]
        K_Train      += [k_train]
        if k_train ==0:
            Eps = mean_loss/1e3
            # Eps = load_openloop[0]/1e3
            if Eps <10:
                eps = 10
            else:
                eps = Eps
        if k_train >=1:
            delta_loss = abs(Loss_episode[-1] - Loss_episode[-2])

        # np.save('trained data/Loss_episode',Loss_episode)
        # np.save('trained data/K_Train',K_Train)
        
        k_train += 1
        # if flag_loss == 1:
        #     break
    
    # if not os.path.exists("plots_test"):
    #     os.makedirs("plots_test")
    # plt.figure(1,dpi=400)
    # plt.plot(TIME_train,Loss,linewidth=1.5)
    # plt.xlabel('Simulation time [s]')
    # plt.ylabel('Loss')
    # plt.grid()
    # plt.savefig('plots_test/Loss_training.png',dpi=400)
    # plt.show()

    # plt.figure(2,dpi=400)
    # plt.plot(K_Train,Loss_episode,linewidth=1.5)
    # plt.xlabel('Training episode')
    # plt.ylabel('Mean loss')
    # plt.grid()
    # plt.savefig('plots_test/Mean_loss.png',dpi=400)
    # plt.show()


    





    

    


            

        


    






    
    






