"""
Main function of the load planner (Tension Allocation)
------------------------------------------------------
1st version, Dr. Wang Bingheng, 19-Dec-2024
"""

from casadi import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import Dynamics_Euler
import Optimal_Allocation_DDP_Euler_autotuning_ADMM
import math
import time as TM
from scipy.spatial.transform import Rotation as Rot
import os

print("=============================================")
print("Main code for training or evaluating Automultilift")
print("Please choose mode")
mode = input("enter 't' or 'e' without the quotation mark:")
print("=============================================")


def dir_cosine(Euler):
    # Euler angles for roll, pitch and yaw
    roll, pitch, yaw = Euler[0,0], Euler[1,0], Euler[2,0]
    # below rotation matrice are used to convert a vector from body frame to world frame
    Rx  = vertcat(
            horzcat(1,0,0),
            horzcat(0,cos(roll),-sin(roll)),
            horzcat(0,sin(roll),cos(roll))
        ) # rotation about x axis that converts a vector in {B} to {I}
    Ry  = vertcat(
            horzcat(cos(pitch),0,sin(pitch)),
            horzcat(0,1,0),
            horzcat(-sin(pitch),0,cos(pitch))
        ) # rotation about y axis that converts a vector in {B} to {I}
    Rz  = vertcat(
            horzcat(cos(yaw),-sin(yaw),0),
            horzcat(sin(yaw),cos(yaw),0),
            horzcat(0,0,1)
        ) # rotation about z axis that converts a vector in {B} to {I}
    # 3-2-1 rotation sequence that rotates the basis of {I} to the basis of {B}.
    # In other words, a body frame is obtained by rotating {I} through the 3-2-1 rotation sequence
    R_wb = Rx.T@Ry.T@Rz.T # rotation matrix that transfers a vector (e.g., a basis) in {I} to {B}
    R_bw = Rz@Ry@Rx       # rotation matrix that transfers a vector in {B} to {I}

    return R_bw

"""--------------------------------------Load Environment---------------------------------------"""
sysm_para = np.array([3, 0.25, 0.25,0.25,0.25, 0.025,0.02,0, 6, 1.25, 0.125, 0.3])
dt        = 0.1 # step size 0.1s
rl        = sysm_para[1]
rq        = sysm_para[10]
ro        = sysm_para[11]
nq        = int(sysm_para[8])
cl0       = sysm_para[9] # cable length
sysm      = Dynamics_Euler.multilift_model(sysm_para,dt)
sysm.model()
nxl       = sysm.nxl # dimension of the load's state
nul       = 3*nq # total dimension of the load's control = 6 (wrench) + 3*6-6 (null-space vector)
nWl       = sysm.nWl

max_line_search_steps = 3
"""--------------------------------------Define Planner---------------------------------------"""
horizon   = 50
e_abs, e_rel = 1e-4, 1e-3
MPC_load  = Optimal_Allocation_DDP_Euler_autotuning_ADMM.MPC_Planner(sysm_para,dt,horizon,e_abs,e_rel)
MPC_load.SetStateVariable(sysm.xl)
MPC_load.SetCtrlVariable(sysm.Wl)
MPC_load.SetDyn(sysm.model_l)
MPC_load.SetLearnablePara()
pob1, pob2 = np.array([[2,3]]).T, np.array([[1,4]]).T # planar positions of the two obstacle in the world frame
print('obstacle_distance=',LA.norm(pob1-pob2))
MPC_load.SetConstraints_ADMM_Subp2(pob1,pob2)
MPC_load.SetCostDyn_ADMM()
MPC_load.ADMM_SubP2_Init()
MPC_load.system_derivatives_DDP_ADMM()
MPC_load.system_derivatives_SubP2_ADMM()
MPC_load.system_derivatives_SubP3_ADMM()

# parameters of RMSProp
lr0       = 1# 0.1 for better ADMM initalization
gamma     = 0.1 # small value leads to oscillations
epsilon   = 1e-8
v0        = np.zeros(MPC_load.n_Pauto)

# parameters of ADAM
m0        = np.zeros(MPC_load.n_Pauto)
beta1     = 0.95 # 0.8 for better ADMM initialization
beta2     = 0.5 # 0.5 for better ADMM initialization
"""--------------------------------------Define Gradient Solver---------------------------------------"""
Grad_Solver = Optimal_Allocation_DDP_Euler_autotuning_ADMM.Gradient_Solver(sysm_para, horizon,sysm.xl,sysm.Wl,MPC_load.sc_xl,MPC_load.sc_Wl,MPC_load.nv,MPC_load.P_auto,
                                                                       MPC_load.P_pinv,MPC_load.P_ns,e_abs,e_rel)

"""--------------------------------------Define Load Reference---------------------------------------"""
Coeffx        = np.zeros((2,8))
Coeffy        = np.zeros((2,8))
Coeffz        = np.zeros((2,8))
for k in range(2):
    Coeffx[k,:] = np.load('Reference_traj/coeffx'+str(k+1)+'.npy')
    Coeffy[k,:] = np.load('Reference_traj/coeffy'+str(k+1)+'.npy')
    Coeffz[k,:] = np.load('Reference_traj/coeffz'+str(k+1)+'.npy')
Ref_xl = np.zeros(nxl*(horizon+1))
Ref_ul = np.zeros(nul*horizon)
Ref_pl = np.zeros((3,horizon+1))
Ref_Wl = np.zeros(nWl*horizon)
Time   = []
time   = 0
for k in range(horizon):
    Time  += [time]
    ref_xl, ref_ul, ref_p, ref_Wl = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time)
    Ref_xl[k*nxl:(k+1)*nxl] = ref_xl
    Ref_ul[k*nul:(k+1)*nul] = ref_ul
    Ref_Wl[k*nWl:(k+1)*nWl] = ref_Wl
    Ref_pl[:,k:k+1] = ref_p
    time += dt
# Time  += [time]
ref_xl, ref_ul, ref_p, ref_Wl = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time)
Ref_xl[horizon*nxl:(horizon+1)*nxl] = ref_xl
Ref_pl[:,horizon:horizon+1] = ref_p

# initial palyload's state
x0         = np.random.normal(0,0.01)
y0         = np.random.normal(0,0.01)
z0         = np.random.normal(1,0.01)
pl         = np.array([[x0,y0,z0]]).T
vl         = np.reshape(np.random.normal(0,0.01,3),(3,1)) # initial velocity of CO in {Bl}
# Eulerl     = 0.5*np.array([[0.2,-0.15,0.1]]).T #np.reshape(np.random.normal(0,0.1,3),(3,1))
angel_max  = 10
roll       = np.clip(np.random.normal(0,0.2,1),-angel_max/57.3,angel_max/57.3)
pitch      = np.clip(np.random.normal(0,0.2,1),-angel_max/57.3,angel_max/57.3)
yaw        = np.clip(np.random.normal(0,0.2,1),-angel_max/57.3,angel_max/57.3)
while (abs(roll*57.3)>9 and abs(pitch*57.3)>9 and abs(yaw*57.3)>9) or (abs(roll*57.3)>9 and abs(pitch*57.3)>9) or (abs(pitch*57.3)>9 and abs(yaw*57.3)>9) or (abs(roll*57.3)>9 and abs(yaw*57.3)>9) or abs(roll*57.3)>9 or abs(pitch*57.3)>9:
    roll       = np.clip(np.random.normal(0,0.2,1),-angel_max/57.3,angel_max/57.3)
    pitch      = np.clip(np.random.normal(0,0.2,1),-angel_max/57.3,angel_max/57.3)
    yaw        = np.clip(np.random.normal(0,0.2,1),-angel_max/57.3,angel_max/57.3)
Eulerl     = np.reshape([roll,pitch,yaw],(3,1))

wl         = np.reshape(np.random.normal(0,0.01,3),(3,1))
xl_init    = np.reshape(np.vstack((pl,vl,Eulerl,wl)),nxl)

# MPC weights (learnable parameters, now manually tuned)
tunable_para0 = np.random.normal(0,0.01,MPC_load.n_Pauto) # initialization

# Solve the load's MPC planner
def train(m0,v0,lr0,xl_init,Ref_xl,Ref_Wl,tunable_para0):
    if not os.path.exists("trained_data"):
        os.makedirs("trained_data")
    tunable_para = tunable_para0
    i = 1
    i_max      = 50
    delta_loss = 1e2
    loss0      = 1e2
    epi        = 1e-1
    xl_train   = []
    Wl_train   = []
    scxl_train = []
    Tl_train   = []
    loss_train = []
    iter_train = []
    gradtimeRe1 = []
    gradtimeRe2 = []
    gradtimeNO1 = []
    gradtimeNO2 = []
    meanerror1  = []
    meanerror2  = []
    Auxtime1_1  = []
    Auxtime1_2  = []
    Auxtime2NO_1= []
    Auxtime2NO_2= []  
    start_time1 = TM.time()
    v          = v0
    m          = m0
    while delta_loss>epi and i<=i_max:
        weight     = Grad_Solver.Set_Parameters(tunable_para)
        p_weight1  = weight[0:MPC_load.n_P1]
        p_weight2  = weight[MPC_load.n_P1:MPC_load.n_P1 + MPC_load.n_P2]
        p1         = weight[-1]
        print('iter_train=',i,'Q=',p_weight1[0:12],'QN=',p_weight1[12:24],'R=',p_weight1[24:30])
        print('iter_train=',i,'nv_w=',p_weight2[0:MPC_load.n_P2],'p1=',p1)
        start_time = TM.time()
        Opt_Sol1, Opt_Sol2, Opt_Y, Opt_Eta  = MPC_load.ADMM_forward_MPC_DDP(xl_init,Ref_xl,Ref_Wl,p_weight1,p_weight2,p1)
        mpctime    = (TM.time() - start_time)*1000
        print("a:--- %s ms ---" % format(mpctime,'.2f'))
        # start_time = TM.time()
        Grad_Out1, Grad_Out2, Grad_Out3, GradTime, GradTimeNO, Meanerror, AuxTime1, AuxTime2NO = MPC_load.ADMM_Gradient_Solver(Opt_Sol1,Opt_Sol2,Opt_Y,Opt_Eta,Ref_xl,Ref_Wl,p_weight1,p_weight2,p1)
        # gradtime    = (TM.time() - start_time)*1000
        # print("g:--- %s ms ---" % format(gradtime,'.2f'))
        gradtimeRe1 += [GradTime[0]]
        gradtimeRe2 += [GradTime[1]]
        gradtimeNO1 += [GradTimeNO[0]]
        gradtimeNO2 += [GradTimeNO[1]]
        meanerror1  += [Meanerror[0]]
        meanerror2  += [Meanerror[1]]
        Auxtime1_1  += [AuxTime1[0]]
        Auxtime1_2  += [AuxTime1[1]]
        Auxtime2NO_1+= [AuxTime2NO[0]]
        Auxtime2NO_2+= [AuxTime2NO[1]]
        dldw, loss  = Grad_Solver.ChainRule(Opt_Sol1,Opt_Sol2,Ref_xl,Grad_Out1,Grad_Out2,p1)
        dwdp        = Grad_Solver.ChainRule_Gradient(tunable_para)
        dldp        = np.reshape(dldw@dwdp,MPC_load.n_Pauto)
        # RMSProp adaptive learning rate
        # for k in range(int(MPC_load.n_Pauto)):
        #     v[k]    = gamma*v[k] + (1-gamma)*dldp[k]**2
        #     lr      = lr0/(np.sqrt(v[k]+epsilon))
        #     tunable_para[k] = tunable_para[k] - lr*dldp[k]
        # ADAM adaptive learning rate
        for k in range(int(MPC_load.n_Pauto)):
            m[k]    = beta1*m[k] + (1-beta1)*dldp[k]
            m_hat   = m[k]/(1-beta1**i)
            v[k]    = beta2*v[k] + (1-beta2)*dldp[k]**2
            v_hat   = v[k]/(1-beta2**i)
            lr      = lr0/(np.sqrt(v_hat+epsilon))
            tunable_para[k] = tunable_para[k] - lr*m_hat

        loss_train += [loss[0]]
        xl_train  += [Opt_Sol1[1]['xl_opt']]
        Wl_train  += [Opt_Sol1[1]['Wl_opt']]
        scxl_train+= [Opt_Sol2[1]['scxl_opt']]
        Tl_train  += [Opt_Sol2[1]['Tl_opt']]
        iter_train += [i]
        if i==1:
            epi = 1e-3*loss
        if i>2:
            delta_loss = abs(loss-loss0)
        loss0      = loss
        print('iter_train=',i,'loss=',loss,'dldpQ=',dldp[0:12],'dldpR=',dldp[24:30],'dldpNv=',dldp[30:33],'dldpp1=',dldp[-1],'lr=',lr)
        i += 1
    traintime    = (TM.time() - start_time1)
    print("train:--- %s s ---" % format(traintime,'.2f'))
    np.save('trained_data/tunable_para_trained',tunable_para)
    np.save('trained_data/loss_train',loss_train)
    np.save('trained_data/xl_train',xl_train)
    np.save('trained_data/scxl_train',scxl_train)
    np.save('trained_data/Wl_train',Wl_train)
    np.save('trained_data/Tl_train',Tl_train)
    np.save('trained_data/training_time',traintime)
    np.save('trained_data/gradtimeRe1',gradtimeRe1)
    np.save('trained_data/gradtimeRe2',gradtimeRe2)
    np.save('trained_data/gradtimeNO1',gradtimeNO1)
    np.save('trained_data/gradtimeNO2',gradtimeNO2)
    np.save('trained_data/meanerror1',meanerror1)
    np.save('trained_data/meanerror2',meanerror2)
    np.save('trained_data/Auxtime1_1',Auxtime1_1)
    np.save('trained_data/Auxtime1_2',Auxtime1_2)
    np.save('trained_data/Auxtime2NO_1',Auxtime2NO_1)
    np.save('trained_data/Auxtime2NO_2',Auxtime2NO_2)
    plt.figure(1,figsize=(6,4),dpi=400)
    plt.plot(loss_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('trained_data/loss_train.png',dpi=300)
    plt.show()



def evaluate(i_train):
    xl_train = np.load('trained_data/xl_train.npy')
    scxl_train = np.load('trained_data/scxl_train.npy')
    Wl_train = np.load('trained_data/Wl_train.npy')
    Tl_train = np.load('trained_data/Tl_train.npy')
    xl_opt   = xl_train[i_train]
    scxl_opt = scxl_train[i_train]
    Wl_opt   = Wl_train[i_train]
    Tl_opt   = Tl_train[i_train]
    # System open-loop predicted trajectories
    P_pinv     = MPC_load.P_pinv # pseudo-inverse of P matrix
    P_ns       = MPC_load.P_ns # null-space of P matrix
    Pl         = np.zeros((3,horizon))
    scPl       = np.zeros((3,horizon))
    for k in range(horizon):
        Pl[:,k:k+1] = np.reshape(xl_opt[k,0:3],(3,1))
        scPl[:,k:k+1] = np.reshape(scxl_opt[k,0:3],(3,1))
    Xq         = [] # list that stores all quadrotors' predicted trajectories
    Aq         = [] # list that stores all cable attachments' trajectories in the world frame
    alpha      = 2*np.pi/nq
    Tq         = np.zeros((nq,horizon))
    for i in range(nq):
        Pi     = np.zeros((3,horizon))
        ri     = np.array([[rl*math.cos(i*alpha),rl*math.sin(i*alpha),0]]).T
        ai     = np.zeros((3,horizon))
        for k in range(horizon):
            wl_k  = np.reshape(Wl_opt[k,:],(6,1)) # 6-D wrench at the kth step
            nv_k  = np.reshape(Tl_opt[k,:],(3*nq-6,1)) # 3-D null-space vector at the kth step
            t_k   = P_pinv@wl_k + P_ns@nv_k # 9-D tension vector at the kth step in the load's body frame
            ti_k  = np.reshape(t_k[3*i:3*(i+1)],(3,1))
            pl_k  = np.reshape(xl_opt[k,0:3],(3,1))
            El_k  = np.reshape(xl_opt[k,6:9],(3,1))
            Rl_k  = dir_cosine(El_k)
            pi_k  = pl_k + Rl_k@(ri + cl0*ti_k/LA.norm(ti_k))
            ai_k  = pl_k + Rl_k@ri
            Pi[:,k:k+1] = pi_k
            ai[:,k:k+1] = ai_k
            Tq[i,k] = LA.norm(ti_k)
        Xq += [Pi]
        Aq += [ai]

    # Plots

    fig1, ax1 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax1.add_patch(obs1)
    ax1.add_patch(obs2)
    ax1.plot(Xq[0][0,:],Xq[0][1,:],label='1st quadrotor',linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,color='blue',fill=False)
        ax1.add_patch(quad)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True)
    fig1.savefig('Planning_plots/quadrotor1_traj_ddp_admm'+str(i_train)+'.png',dpi=400)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax2.add_patch(obs1)
    ax2.add_patch(obs2)
    ax2.plot(Xq[1][0,:],Xq[1][1,:],label='2nd quadrotor',linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,color='blue',fill=False)
        ax2.add_patch(quad)
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True)
    fig2.savefig('Planning_plots/quadrotor2_traj_ddp_admm'+str(i_train)+'.png',dpi=400)
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax3.add_patch(obs1)
    ax3.add_patch(obs2)
    ax3.plot(Xq[2][0,:],Xq[2][1,:],label='3rd quadrotor',linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,color='blue',fill=False)
        ax3.add_patch(quad)
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('y [m]')
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.grid(True)
    fig3.savefig('Planning_plots/quadrotor3_traj_ddp_admm'+str(i_train)+'.png',dpi=400)
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax4.add_patch(obs1)
    ax4.add_patch(obs2)
    ax4.plot(Xq[3][0,:],Xq[3][1,:],label='4th quadrotor',linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,color='blue',fill=False)
        ax4.add_patch(quad)
    ax4.set_xlabel('x [m]')
    ax4.set_ylabel('y [m]')
    ax4.set_aspect('equal')
    ax4.legend()
    ax4.grid(True)
    fig4.savefig('Planning_plots/quadrotor4_traj_admm'+str(i_train)+'.png',dpi=400)
    plt.show()

    fig5, ax5 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax5.add_patch(obs1)
    ax5.add_patch(obs2)
    ax5.plot(Xq[4][0,:],Xq[4][1,:],label='5th quadrotor',linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[4][0,k],Xq[4][1,k]),rq,color='blue',fill=False)
        ax5.add_patch(quad)
    ax5.set_xlabel('x [m]')
    ax5.set_ylabel('y [m]')
    ax5.set_aspect('equal')
    ax5.legend()
    ax5.grid(True)
    fig5.savefig('Planning_plots/quadrotor5_traj_admm'+str(i_train)+'.png',dpi=400)
    plt.show()

    fig6, ax6 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax6.add_patch(obs1)
    ax6.add_patch(obs2)
    ax6.plot(Xq[5][0,:],Xq[5][1,:],label='6th quadrotor',linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[5][0,k],Xq[5][1,k]),rq,color='blue',fill=False)
        ax6.add_patch(quad)
    ax6.set_xlabel('x [m]')
    ax6.set_ylabel('y [m]')
    ax6.set_aspect('equal')
    ax6.legend()
    ax6.grid(True)
    fig6.savefig('Planning_plots/quadrotor6_traj_admm'+str(i_train)+'.png',dpi=400)
    plt.show()

    fig7, ax7 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax7.add_patch(obs1)
    ax7.add_patch(obs2)
    ax7.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax7.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
    ax7.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
    kt = horizon/50
    for k in range(horizon):
        if k==1*kt or k==20*kt or k==26*kt or k==35*kt or k==49*kt:
            quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
            ax7.add_patch(quad1)
            quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
            ax7.add_patch(quad2)
            quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
            ax7.add_patch(quad3)
            quad4  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,fill=False)
            ax7.add_patch(quad4)
            quad5  = Circle((Xq[4][0,k],Xq[4][1,k]),rq,fill=False)
            ax7.add_patch(quad5)
            quad6  = Circle((Xq[5][0,k],Xq[5][1,k]),rq,fill=False)
            ax7.add_patch(quad6)
            ax7.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Xq[3][0,k],Aq[3][0,k]],[Xq[3][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Xq[4][0,k],Aq[4][0,k]],[Xq[4][1,k],Aq[4][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Xq[5][0,k],Aq[5][0,k]],[Xq[5][1,k],Aq[5][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Aq[2][0,k],Aq[3][0,k]],[Aq[2][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Aq[3][0,k],Aq[4][0,k]],[Aq[3][1,k],Aq[4][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Aq[4][0,k],Aq[5][0,k]],[Aq[4][1,k],Aq[5][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Aq[5][0,k],Aq[0][0,k]],[Aq[5][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
    ax7.set_xlabel('x [m]')
    ax7.set_ylabel('y [m]')
    ax7.set_aspect('equal')
    ax7.legend()
    ax7.grid(True)
    fig7.savefig('Planning_plots/system_traj_quadrotor_num6_ddp_admm'+str(i_train)+'.png',dpi=400)
    plt.show()


    plt.figure(8,figsize=(6,4),dpi=300)
    plt.plot(Time,Tq[0,:],linewidth=1,label='1st cable')
    plt.plot(Time,Tq[1,:],linewidth=1,label='2nd cable')
    plt.plot(Time,Tq[2,:],linewidth=1,label='3rd cable')
    plt.plot(Time,Tq[3,:],linewidth=1,label='4th cable')
    plt.plot(Time,Tq[4,:],linewidth=1,label='5th cable')
    plt.plot(Time,Tq[5,:],linewidth=1,label='6th cable')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('MPC tension force [N]')
    plt.grid()
    plt.savefig('Planning_plots/cable_MPC_tensions_6_ddp_admm'+str(i_train)+'.png',dpi=400)
    plt.show()

    plt.figure(9,figsize=(6,4),dpi=300)
    plt.plot(Time,xl_opt[:horizon,6]*57.3,linewidth=1,label='roll')
    plt.plot(Time,xl_opt[:horizon,7]*57.3,linewidth=1,label='pitch')
    plt.plot(Time,xl_opt[:horizon,8]*57.3,linewidth=1,label='yaw')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Euler angle [deg]')
    plt.grid()
    plt.savefig('Planning_plots/euler_6_ddp_admm'+str(i_train)+'.png',dpi=400)
    plt.show()

"""---------------------------------Main function-----------------------------"""
if mode =="t":
    train(m0,v0,lr0,xl_init,Ref_xl,Ref_Wl,tunable_para0)
else:
    loss_train = np.load('trained_data/loss_train.npy')
    evaluate(len(loss_train)-1)
    evaluate(0)
    evaluate(4)
