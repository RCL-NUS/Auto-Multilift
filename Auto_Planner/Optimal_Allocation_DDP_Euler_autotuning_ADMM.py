from casadi import *
import numpy as np
from numpy import linalg as LA
import math
from scipy.spatial.transform import Rotation as Rot
from scipy import linalg as sLA
from scipy.linalg import null_space
import time as TM

class MPC_Planner:
    def __init__(self, sysm_para, dt_ctrl, horizon, e_abs, e_rel):
        # Payload's parameters
        self.ml     = sysm_para[0] # the payload's mass
        self.rl     = sysm_para[1] # radius of the payload
        self.Jl     = np.diag(sysm_para[2:5])
        self.rg     = np.reshape(sysm_para[5:8],(3,1)) # position of the load CoM in {Bl}
        self.S_rg   = self.skew_sym_numpy(self.rg)
        self.nq     = sysm_para[8] # number of the quadrotors
        self.cl0    = sysm_para[9] # cable length 
        self.rq     = sysm_para[10] # quadrotor radius
        self.ro     = sysm_para[11] # obstacle radius
        self.alpha  = 2*np.pi/self.nq
        r0          = np.array([[self.rl,0,0]]).T # 1st cable attachment point in {Bl}
        self.ra     = r0
        S_r0        = self.skew_sym_numpy(r0)
        I3          = np.identity(3) # 3-by-3 identity matrix
        self.Pt      = np.vstack((I3,S_r0))
        for i in range(int(self.nq)-1):
            ri      = np.array([[self.rl*(math.cos((i+1)*self.alpha)),self.rl*(math.sin((i+1)*self.alpha)),0]]).T
            S_ri    = self.skew_sym_numpy(ri)
            Pi      = np.vstack((I3,S_ri))
            self.Pt = np.append(self.Pt,Pi,axis=1) # the tension mapping matrix: 6-by-3nq with a rank of 6
            self.ra = np.append(self.ra,ri,axis=1) # a matrix that stores the attachment points
        self.P_pinv = self.Pt.T@LA.inv(self.Pt@self.Pt.T) # pseudo-inverse of P
        self.P_ns   = null_space(self.Pt) # null-space of P, 3nq-by-(3nq-6)
        # Unit direction vector free of coordinate
        self.ex     = np.array([[1, 0, 0]]).T
        self.ey     = np.array([[0, 1, 0]]).T
        self.ez     = np.array([[0, 0, 1]]).T
        # Gravitational acceleration
        self.g      = 9.81      
        self.dt     = dt_ctrl
        # MPC's horizon
        self.N      = horizon
        # Tolerances used in ADMM
        self.e_abs  = e_abs
        self.e_rel  = e_rel

    def skew_sym_numpy(self, v):
        v_cross = np.array([
            [0, -v[2, 0], v[1, 0]],
            [v[2, 0], 0, -v[0, 0]],
            [-v[1, 0], v[0, 0], 0]]
        )
        return v_cross

    def SetStateVariable(self, xl):
        self.xl    = xl # payload's state
        self.n_xl  = xl.numel() # 12
        self.xl_lb = self.n_xl*[-1e19]# state constraint (infinity)
        self.xl_ub = self.n_xl*[1e19]
        self.sc_xl = SX.sym('sc_xl',self.n_xl,1) # safe copy state
        self.sc_xL = SX.sym('sc_xL',self.n_xl,1) # Lagrangian multiplier associated with the safe copy state

    def SetCtrlVariable(self, Wl):
        self.Wl    = Wl # load's control, 6-by-1 vector, wrench in the load body frame
        self.n_Wl  = Wl.numel()
        self.sc_Wl = SX.sym('sc_Wl',self.n_Wl,1) # safe copy control
        self.sc_WL = SX.sym('sc_WL',self.n_Wl,1) # Lagrangian multiplier associated with the safe copy control
        self.nv    = SX.sym('nv',3*int(self.nq)-6,1) # null-space vector
        self.n_nv  = self.nv.numel()
        self.ul    = vertcat(self.Wl,self.nv)
        self.n_ul  = Wl.numel() + self.nv.numel() # including the dimension of the null-space vector, totally 3nq
        tm_c       = 1e19 # bound of the tension modifier
        ke         = 1
        Wl_lb      = [-10*self.nq*ke,-10*self.nq*ke,0.1*self.nq*ke,-20*self.nq*self.rl*ke,-20*self.nq*self.rl*ke,-20*self.nq*self.rl*ke] # 6-D wrench, lower-bound
        self.Wl_lb = self.n_Wl*[-1e19] #Wl_lb # having the appropriate box constraints on control is very important
        Tm_lb      = (3*int(self.nq)-6)*[-tm_c]
        self.Tm_lb = Tm_lb
        ul_lb      = np.concatenate((self.Wl_lb,Tm_lb)) 
        self.ul_lb = ul_lb.tolist() #self.n_ul*[-tm_c]
        Wl_ub      = [10*self.nq*ke,10*self.nq*ke,20*self.nq*ke,20*self.nq*self.rl*ke,20*self.nq*self.rl*ke,20*self.nq*self.rl*ke] # 6-D wrench, upper-bound
        self.Wl_ub = self.n_Wl*[1e19] #Wl_ub
        Tm_ub      = (3*int(self.nq)-6)*[tm_c]
        self.Tm_ub = Tm_ub
        ul_ub      = np.concatenate((self.Wl_ub,Tm_ub))
        self.ul_ub = ul_ub.tolist() #self.n_ul*[tm_c]

    def SetDyn(self, model_l):
        self.Modell   = self.xl + self.dt*model_l # 4th-order Runge-Kutta discrete-time dynamics model
        self.MDynl_fn = Function('MDynl',[self.xl, self.ul],[self.Modell],['xl0', 'ul0'],['MDynlf'])
        self.MDynl_fn_admm = Function('MDynl_admm',[self.xl,self.Wl],[self.Modell],['xl0','Wl0'],['MDynlf_admm'])

    def dir_cosine(self, Euler):
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
    
    def vee_map(self, v):
        vect = vertcat(v[2, 1], v[0, 2], v[1, 0])
        return vect
    
    def SetLearnablePara(self):
        self.n_wsl    = 12 # dimension of the payload state weightings
        self.p1       = SX.sym('p1') # penalty parameter 1 in ADMM
        self.P1_l     = SX.sym('P1_l',1,(2*self.n_wsl+self.n_Wl)) # hyperparameters of subproblem 1 in ADMM
        self.n_P1     = self.P1_l.numel()
        self.P2_l     = SX.sym('P2',1,3) #self.n_nv hyperparameters of subproblem 2 in ADMM
        self.n_P2     = self.P2_l.numel()
        self.P_auto   = horzcat(self.P1_l,self.P2_l,self.p1) # total learnable parameters
        self.n_Pauto  = self.P_auto.numel()

    
    def SetCostDyn_ADMM(self):
        self.ref_xl   = SX.sym('ref_xl',self.n_xl,1)
        self.ref_Wl   = SX.sym('ref_Wl',self.n_Wl,1)
        
        # p_error_l     = self.xl[0:3,0] - self.ref_xl[0:3,0]
        # v_error_l     = self.xl[3:6,0] - self.ref_xl[3:6,0]
        # ql            = self.xl[6:10,0]
        # ref_ql        = self.ref_xl[6:10,0]
        # Rl            = self.q_2_rotation(ql)
        # Rdl           = self.q_2_rotation(ref_ql)
        # error_Rl      = Rdl.T@Rl - Rl.T@Rdl
        # att_error_l   = 1/2*self.vee_map(error_Rl) # attitude error in Lie group
        # w_error_l     = self.xl[10:13,0] - self.ref_xl[10:13,0]
        # track_error_l = vertcat(p_error_l,v_error_l,att_error_l,w_error_l)
        track_error_l = self.xl - self.ref_xl
        ctrl_error_l  = self.Wl - self.ref_Wl
        xl_const_v    = self.xl - self.sc_xl + self.sc_xL/self.p1 # state constraint violation
        Wl_const_v    = self.Wl - self.sc_Wl + self.sc_WL/self.p1 # control constraint violation
        self.Ql_k     = diag(self.P1_l[0,0:self.n_wsl])
        self.Ql_N     = diag(self.P1_l[0,self.n_wsl:2*self.n_wsl])
        self.Rl_k     = diag(self.P1_l[0,2*self.n_wsl:2*self.n_wsl+self.n_Wl])
        # path cost of Subproblem 1 (MPC)
        self.Jl_k_bar_admm = 1/2 * (track_error_l.T@self.Ql_k@track_error_l + ctrl_error_l.T@self.Rl_k@ctrl_error_l) 
        self.Jl_k_hat_admm = self.p1/2*xl_const_v.T@xl_const_v + self.p1/2*Wl_const_v.T@Wl_const_v
        self.Jl_k_admm     = self.Jl_k_bar_admm + self.Jl_k_hat_admm
        self.Jl_kfn_admm   = Function('Jl_k_admm',[self.xl, self.Wl, self.ref_xl, self.ref_Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1],[self.Jl_k_admm],['xl0', 'Wl0', 'refxl0', 'refWl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P1l0', 'p10'],['Jl_kf_admm'])
        # terminal cost of Subproblem 1 (MPC)
        self.Jl_N_admm     = 1/2 * track_error_l.T@self.Ql_N@track_error_l 
        self.Jl_Nfn_admm   = Function('Jl_N_admm',[self.xl, self.ref_xl, self.P1_l],[self.Jl_N_admm],['xl0', 'refxl0', 'P1l0'],['Jl_Nf_admm'])
        # path cost of Subproblem 2 (Static optimization)
        t                  = self.P_pinv@self.sc_Wl + self.P_ns@self.nv
        self.J2_k          = 0
        self.ref_ti        = vertcat(0,0,self.ml*self.g/self.nq)
        self.R2_k          = diag(self.P2_l)
        for i in range(int(self.nq)):
            ti   = t[(3*i):(3*(i+1))]
            tension_error = ti-self.ref_ti
            self.J2_k    += 1/2 *tension_error.T@self.R2_k@tension_error
        # self.J2_k          = 1/2 * self.nv.T@self.R2_k@self.nv
        self.J2_k_admm     = self.J2_k + self.p1/2*xl_const_v.T@xl_const_v + self.p1/2*Wl_const_v.T@Wl_const_v
        self.J2_kfn_admm   = Function('J2_k_admm',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1],[self.J2_k_admm],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10'],['J2_kf_admm'])
        self.J2_k_soft = self.J2_k_admm + self.T_cons + self.G_ij + self.G_io
  
    def SetConstraints_ADMM_Subp2(self, pob1, pob2):
        # k = 1.1
        # Lambda = [k*12.5,k*11,k*6.5]
        Wl   = self.sc_Wl
        tm   = self.nv
        pl   = self.sc_xl[0:3]
        El   = self.sc_xl[6:9]
        Rl   = self.dir_cosine(El)
        t    = self.P_pinv@Wl + self.P_ns@tm # 3nq-by-1 total tension vector in the load's body frame, at the kth step
        Pi   = [] # list that stores all the quadrotors' planar positions in the world frame
        Pil  = [] # list that stores all the quadrotors' planar positions in the body frame
        Pic  = [] # list that stores all the cables' planar positions
        Pic2  = [] # list that stores all the cables' planar positions
        self.Gij_admm   = [] # list that stores all the safe inter-robot inequality constraints
        self.Gi1_admm   = [] # list that stores the obstacle-avoidance constraints of all the quadrotors for the 1st obstacle
        self.Gijc_admm  = []
        self.Gijc2_admm  = []
        self.Gi2_admm   = [] # list that stores the obstacle-avoidance constraints of all the quadrotors for the 2nd obstacle
        self.Gi1c_admm   = []
        self.Gi2c_admm   = []
        self.Ti_admm    = [] # list that stores the tension magnitudes
        gol1 = norm_2(pl[0:2]-pob1) - (self.rl + self.ro) 
        self.gol1_fn_admm = Function('gol1_admm',[self.sc_xl],[gol1],['scxl0'],['gol1f_admm'])
        gol2 = norm_2(pl[0:2]-pob2) - (self.rl + self.ro)
        self.gol2_fn_admm = Function('gol2_admm',[self.sc_xl],[gol2],['scxl0'],['gol2f_admm'])
        self.T_cons = 0 # barrier functions of box constraints on the tension magnitude
        p_barrier    = 1e-6 # barrier parameter
        for i in range(int(self.nq)):
            ri   = np.reshape(self.ra[:,i],(3,1)) # ith attachment point in the load's body frame
            ti   = t[(3*i):(3*(i+1))] # ith tension, a 3-by-1 vector, in the load's body frame
            pi   = pl + Rl@(ri+self.cl0*ti/norm_2(ti)) # ith quadrotor's position in the world frame
            # lambdai = Lambda[i]
            # pi   = pl + (ri+self.cl0*ti/norm_2(ti)) # ith quadrotor's position in the world frame, approximated, R=I, denominator = lambdai
            pil  = (ri+self.cl0*ti/norm_2(ti))
            # pil  = (ri+self.cl0*ti/norm_2(ti)) #ith quadrotor's position in the body frame, approximated, R=I, denominator = lambdai
            pic  = pl + (ri+0.1*self.cl0*ti/norm_2(ti)) # ith cable middel point's position in the world frame
            pic2  = pl + (ri+0.5*self.cl0*ti/norm_2(ti)) # ith cable middel point's position in the world frame
            Pi  += [pi[0:2]]
            Pil += [pil[0:2]]
            Pic += [pic[0:2]]
            Pic2 += [pic2[0:2]]
            ti_m = norm_2(ti)
            ti_fn= Function('t_admm'+str(i),[self.sc_Wl,self.nv],[ti_m],['scWl0','tm0'],['t'+str(i)+'f_admm'])
            self.t_min = 0.1
            self.t_max = 20
            self.T_cons += -p_barrier * log(ti_m-self.t_min)
            self.T_cons += -p_barrier * log(self.t_max-ti_m)
            self.Ti_admm += [ti_fn]
        k = 0
        self.G_ij    = 0 # barrier functions of safe inter-robot constraints on quadrotors' planar positions
        self.G_io    = 0 # barrier functions of safe collision-avoidance constraints on quadrotors' planar positions 
        
        for i in range(int(self.nq)):
            go1   = norm_2(Pi[i]-pob1) - (self.rq + self.ro) # safe constriant between the obstacle 1 and the ith quadrotor, which should be positive
            go1_fn= Function('go1_admm'+str(i),[self.sc_xl,self.sc_Wl,self.nv],[go1],['scxl0','scWl0','tm0'],['go1'+str(i)+'f_admm'])
            self.G_io += -p_barrier * log(go1)
            go1c  = norm_2(Pic[i]-pob1) - (self.rq + self.ro) 
            go1c_fn=Function('go1c_admm'+str(i),[self.sc_xl,self.sc_Wl,self.nv],[go1c],['scxl0','scWl0','tm0'],['go1c'+str(i)+'f_admm'])
            self.Gi1_admm  += [go1_fn]
            self.Gi1c_admm += [go1c_fn]
            go2   = norm_2(Pi[i]-pob2) - (self.rq + self.ro) # safe constriant between the obstacle 2 and the ith quadrotor, which should be positive
            go2_fn= Function('go2_admm'+str(i),[self.sc_xl,self.sc_Wl,self.nv],[go2],['scxl0','scWl0','tm0'],['go2'+str(i)+'f_admm'])
            self.G_io += -p_barrier * log(go2)
            go2c  = norm_2(Pic[i]-pob2) - (self.rq + self.ro) 
            go2c_fn=Function('go2c_admm'+str(i),[self.sc_xl,self.sc_Wl,self.nv],[go2c],['scxl0','scWl0','tm0'],['go2c'+str(i)+'f_admm'])
            self.Gi2_admm  += [go2_fn]
            self.Gi2c_admm += [go2c_fn]

            for j in range(i+1,int(self.nq)):
                gij   = norm_2(Pil[i]-Pil[j]) - 2*self.rq # safe constraint between two quadrotors
                gijc  = norm_2(Pic[i]-Pic[j]) - 1.5*self.rq
                gijc2  = norm_2(Pic2[i]-Pic2[j]) - 1.5*self.rq
                gij_fn= Function('g_admm'+str(k),[self.sc_xl,self.sc_Wl,self.nv],[gij],['scxl0','scWl0','tm0'],['g'+str(k)+'f_admm'])
                gijc_fn= Function('g_admmc'+str(k),[self.sc_xl,self.sc_Wl,self.nv],[gijc],['scxl0','scWl0','tm0'],['gc'+str(k)+'f_admm'])
                gijc2_fn= Function('g_admmc2'+str(k),[self.sc_xl,self.sc_Wl,self.nv],[gijc2],['scxl0','scWl0','tm0'],['gc2'+str(k)+'f_admm'])
                self.G_ij += -p_barrier * log(gij)
                self.Gij_admm  += [gij_fn]
                self.Gijc_admm += [gijc_fn]
                self.Gijc2_admm += [gijc2_fn]
                k += 1

    
    def ADMM_SubP2_Init(self):
        # start with an empty NLP
        w        = [] # optimal trajectory list
        self.lbw2 = [] # lower boundary list of optimal variables
        self.ubw2 = [] # upper boundary list of optimal variables
        g        = [] # equality and inequality constraint list
        self.lbg2 = [] # lower boundary list of constraints
        self.ubg2 = [] # upper boundary list of constraints
        
        # hyperparameters + external signals
        P2l      = SX.sym('P2', (self.n_P2 # hyperparameters of Subproblem2
                                +self.n_xl # state of subproblem 1 at step k
                                +self.n_xl # Lagrangian associated with the state of subproblem 1 at step k
                                +self.n_Wl # control of subproblem 1 at step k
                                +self.n_Wl # Lagrangian associated with the control of subproblem 1 at step k
                                +1 # penalty parameters
                                ))
        
        # formulate the NLP
        P2_l     = P2l[0:self.n_P2]
        p1       = P2l[self.n_P2+2*self.n_xl+2*self.n_Wl]
        Xk          = SX.sym('x',self.n_xl,1) # safe-copy state
        w          += [Xk]
        self.lbw2  += self.xl_lb
        self.ubw2  += self.xl_ub
        xl_k        = P2l[(self.n_P2):(self.n_P2+self.n_xl)]
        # X0          = []
        # for i in range(self.n_xl):
        #     X0     += [xl_k[i]]
        # self.w02   += X0 #[0.5 * (x + y) for x,y in zip(self.xl_lb, self.xl_ub)]
        Wk          = SX.sym('w',self.n_Wl,1) # safe-copy control
        w          += [Wk]
        self.lbw2  += self.Wl_lb
        self.ubw2  += self.Wl_ub
        Wl_k        = P2l[(self.n_P2+2*self.n_xl):(self.n_P2+2*self.n_xl+self.n_Wl)]
        # W0          = []
        # for i in range(self.n_Wl):
        #     W0     += [Wl_k[i]]
        # self.w02   += W0 #[0.5 * (x + y) for x,y in zip(self.Wl_lb, self.Wl_ub)]
        Tk          = SX.sym('tm',self.n_nv,1) # null-space vector, tension modifier
        w          += [Tk]
        self.lbw2  += self.Tm_lb
        self.ubw2  += self.Tm_ub
        # self.w02   += [0.5 * (x + y) for x,y in zip(self.Tm_lb, self.Tm_ub)]
        xL_k        = P2l[(self.n_P2+self.n_xl):(self.n_P2+self.n_xl+self.n_xl)]
        WL_k        = P2l[(self.n_P2+2*self.n_xl+self.n_Wl):(self.n_P2+2*self.n_xl+self.n_Wl+self.n_Wl)]
        J           = self.J2_kfn_admm(nv0=Tk,xl0=xl_k,Wl0=Wl_k,scxl0=Xk,scxL0=xL_k,scWl0=Wk,scWL0=WL_k,P2l0=P2_l,p10=p1)['J2_kf_admm']
        # add inequality tension magnitude safe constraints
        for i in range(int(self.nq)):
            ti_k= self.Ti_admm[i](scWl0=Wk,tm0=Tk)['t'+str(i)+'f_admm']
            g  += [ti_k]
            self.lbg2 += [self.t_min] # to prevent it from being slack
            self.ubg2 += [self.t_max]
        # add inequality obstacle-avoidance constraints
        for i in range(int(self.nq)):
            gi1 = self.Gi1_admm[i](scxl0=Xk,scWl0=Wk,tm0=Tk)['go1'+str(i)+'f_admm']
            g += [gi1]
            self.lbg2 += [0.05]
            self.ubg2 += [100] # add an upbound for numerical stability
            gi2 = self.Gi2_admm[i](scxl0=Xk,scWl0=Wk,tm0=Tk)['go2'+str(i)+'f_admm']
            g += [gi2]
            self.lbg2 += [0.05]
            self.ubg2 += [100] # add an upbound for numerical stability

            # gi1c = self.Gi1c_admm[i](scxl0=Xk,scWl0=Wk,tm0=Tk)['go1c'+str(i)+'f_admm']
            # g += [gi1c]
            # self.lbg2 += [0.05]
            # self.ubg2 += [100] # add an upbound for numerical stability
            # gi2c = self.Gi2c_admm[i](scxl0=Xk,scWl0=Wk,tm0=Tk)['go2c'+str(i)+'f_admm']
            # g += [gi2c]
            # self.lbg2 += [0.05]
            # self.ubg2 += [100] # add an upbound for numerical stability

        # add inequality safe inter-robot constraints
        for i in range(len(self.Gij_admm)):
            gij = self.Gij_admm[i](scxl0=Xk,scWl0=Wk,tm0=Tk)['g'+str(i)+'f_admm']
            g += [gij]
            self.lbg2 += [0.05]
            self.ubg2 += [100] # add an upbound for numerical stability
            # gijc = self.Gijc_admm[i](scxl0=Xk,scWl0=Wk,tm0=Tk)['gc'+str(i)+'f_admm']
            # g += [gijc]
            # self.lbg2 += [0.05]
            # self.ubg2 += [100] # add an upbound for numerical stability
            # gijc2 = self.Gijc2_admm[i](scxl0=Xk,scWl0=Wk,tm0=Tk)['gc2'+str(i)+'f_admm']
            # g += [gijc2]
            # self.lbg2 += [0.05]
            # self.ubg2 += [100] # add an upbound for numerical stability

        # create an NLP solver and solve it
        opts = {}
        opts['ipopt.tol'] = 1e-8
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
        opts['ipopt.warm_start_init_point']='yes'
        opts['ipopt.max_iter']=1e3
        opts['ipopt.acceptable_tol']=1e-8
        opts['ipopt.mu_strategy']='adaptive'
        
        prob = {'f': J, 
                'x': vertcat(*w), 
                'p': P2l,
                'g': vertcat(*g)}
        
        self.solver2 = nlpsol('solver', 'ipopt', prob, opts)  
        # self.solver2 = nlpsol('solver', 'sqpmethod', prob, opts) 
    
    def ADMM_SubP2(self,P2l):
        P2_l     = P2l[0:self.n_P2]
        p1       = P2l[self.n_P2+2*self.N*self.n_xl+2*self.N*self.n_Wl]
        sc_xl_traj = np.zeros((self.N,self.n_xl))
        sc_Wl_traj = np.zeros((self.N,self.n_Wl))
        Tl_traj    = np.zeros((self.N,self.n_nv))
        ctrl_traj  = np.zeros((self.N,self.n_ul))
        for k in range(self.N):
            X0 = []
            W0 = []
            self.w02  = [] # initial guess list of optimal trajectory 
            xl_k        = P2l[(self.n_P2+k*self.n_xl):(self.n_P2+(k+1)*self.n_xl)]
            for i in range(self.n_xl):
                X0 += [xl_k[i]]
            self.w02 += X0
            xL_k        = P2l[(self.n_P2+self.N*self.n_xl+k*self.n_xl):(self.n_P2+self.N*self.n_xl+(k+1)*self.n_xl)]
            Wl_k        = P2l[(self.n_P2+2*self.N*self.n_xl+k*self.n_Wl):(self.n_P2+2*self.N*self.n_xl+(k+1)*self.n_Wl)]
            for i in range(self.n_Wl):
                W0 += [Wl_k[i]]
            self.w02 += W0
            self.w02   += [0.5 * (x + y) for x,y in zip(self.Tm_lb, self.Tm_ub)]
            WL_k        = P2l[(self.n_P2+2*self.N*self.n_xl+self.N*self.n_Wl+k*self.n_Wl):(self.n_P2+2*self.N*self.n_xl+self.N*self.n_Wl+(k+1)*self.n_Wl)]
            para2       = np.concatenate((P2_l,xl_k))
            para2       = np.concatenate((para2,xL_k))
            para2       = np.concatenate((para2,Wl_k))
            para2       = np.concatenate((para2,WL_k))
            para2       = np.concatenate((para2,[p1]))
            # Solve the NLP

            sol = self.solver2(x0=self.w02, 
                          lbx=self.lbw2, 
                          ubx=self.ubw2, 
                          p=para2,
                          lbg=self.lbg2, 
                          ubg=self.ubg2)
        
            w_opt = sol['x'].full().flatten()
            # take the optimal control and state
            sol_traj = np.reshape(w_opt, (-1, self.n_xl + self.n_Wl + self.n_nv))
            state_traj_opt = sol_traj[:, 0:self.n_xl]
            Wl_traj_opt    = sol_traj[:, self.n_xl:(self.n_xl+self.n_Wl)]
            Tl_traj_opt    = sol_traj[:, (self.n_xl+self.n_Wl):]
            control_traj_opt = sol_traj[:, self.n_xl:]
            sc_xl_traj[k:k+1,:] = state_traj_opt
            sc_Wl_traj[k:k+1,:] = Wl_traj_opt
            Tl_traj[k:k+1,:]    = Tl_traj_opt
            ctrl_traj[k:k+1,:]  = control_traj_opt

        # output
        opt_sol2 = {"scxl_opt":sc_xl_traj,
                   "scWl_opt":sc_Wl_traj,
                   "Tl_opt":Tl_traj,
                  "ul_opt":ctrl_traj
                  }
        return opt_sol2 
    
    def ADMM_SubP3(self,xL_opt,WL_opt,xl_opt,scxl_opt,Wl_opt,scWl_opt,p1):
        Y_new   = np.zeros((self.n_xl,self.N))
        Eta_new = np.zeros((self.n_Wl,self.N))
        for k in range(self.N):
            y_k        = np.reshape(xL_opt[:,k],(self.n_xl,1)) # old Lagrangian multiplier associated with the safe copy state
            xl_k       = np.reshape(xl_opt[k,:],(self.n_xl,1))
            scxl_k     = np.reshape(scxl_opt[k,:],(self.n_xl,1))
            eta_k      = np.reshape(WL_opt[:,k],(self.n_Wl,1)) # old Lagrangian multiplier associated with the safe copy control
            Wl_k       = np.reshape(Wl_opt[k,:],(self.n_Wl,1))
            scWl_k     = np.reshape(scWl_opt[k,:],(self.n_Wl,1))
            y_k_new    = y_k + p1*(xl_k - scxl_k)
            eta_k_new  = eta_k + p1*(Wl_k - scWl_k)
            Y_new[:,k:k+1] = y_k_new
            Eta_new[:,k:k+1] = eta_k_new

        return Y_new, Eta_new
    
    
    def adaptive_penalty(self,p,tau,mu,r_p,r_d):
        if r_p>(mu*r_d):
            p_new = tau*p
        elif r_d>(mu*r_p):
            p_new = p/tau
        else:
            p_new = p
        
        return p_new

    def system_derivatives_DDP_ADMM(self):
        alpha       = 1
        self.Vx     = SX.sym('Vx',self.n_xl)
        self.Vxx    = SX.sym('Vxx',self.n_xl,self.n_xl)
        # gradients of the system dynamics, the cost function, and the Q value function
        self.Fx     = jacobian(self.Modell,self.xl)
        self.Fx_fn  = Function('Fx',[self.xl,self.Wl],[self.Fx],['xl0','Wl0'],['Fx_f'])
        self.Fu     = jacobian(self.Modell,self.Wl)
        self.Fu_fn  = Function('Fu',[self.xl,self.Wl],[self.Fu],['xl0','Wl0'],['Fu_f'])
        self.lx     = jacobian(self.Jl_k_admm,self.xl)
        self.lxN    = jacobian(self.Jl_N_admm,self.xl)
        self.lxN_fn = Function('lxN',[self.xl,self.ref_xl,self.P1_l],[self.lxN],['xl0','refxl0','P1l0'],['lxNf'])
        self.lu     = jacobian(self.Jl_k_admm,self.Wl)
        self.Qx     = self.lx.T + self.Fx.T@self.Vx
        self.Qx_fn  = Function('Qx',[self.xl,self.Wl,self.Vx,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1],[self.Qx],['xl0','Wl0','Vx0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10'],['Qxf'])
        self.Qu     = self.lu.T + self.Fu.T@self.Vx
        self.Qu_fn  = Function('Qu',[self.xl,self.Wl,self.Vx,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1],[self.Qu],['xl0','Wl0','Vx0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10'],['Quf'])
        # hessians of the system dynamics, the cost function, and the Q value function
        self.FxVx   = self.Fx.T@self.Vx
        self.dFxVxdx= jacobian(self.FxVx,self.xl) # the hessian of the system dynamics may cause heavy computational burden
        self.dFxVxdu= jacobian(self.FxVx,self.Wl)
        self.FuVx   = self.Fu.T@self.Vx
        self.dFuVxdu= jacobian(self.FuVx,self.Wl)
        self.lxx    = jacobian(self.lx,self.xl)
        self.lxxN   = jacobian(self.lxN,self.xl)
        self.lxxN_fn= Function('lxxN',[self.P1_l],[self.lxxN],['P1l0'],['lxxNf'])
        self.lxu    = jacobian(self.lx,self.Wl)
        self.luu    = jacobian(self.lu,self.Wl)
        self.Qxx_bar    = self.lxx #+ alpha*self.dFxVxdx  # removing this model hessian can enhance the DDP stability for a larger time step!!!! The removal can also accelerate the DDP computation significantly!
        self.Qxx_bar_fn = Function('Qxx_bar',[self.xl,self.Wl,self.Vx,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1],[self.Qxx_bar],['xl0','Wl0','Vx0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10'],['Qxx_bar_f'])
        self.Qxx_hat    = self.Fx.T@self.Vxx@self.Fx
        self.Qxx_hat_fn = Function('Qxx_hat',[self.xl,self.Wl,self.Vxx],[self.Qxx_hat],['xl0','Wl0','Vxx0'],['Qxx_hat_f'])
        self.Qxu_bar    = self.lxu #+ alpha*self.dFxVxdu  # including the model hessian entails a very small time step size (e.g., 0.01s)
        self.Qxu_bar_fn = Function('Qxu_bar',[self.xl,self.Wl,self.Vx,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1],[self.Qxu_bar],['xl0','Wl0','Vx0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10'],['Qxu_bar_f'])
        self.Qxu_hat    = self.Fx.T@self.Vxx@self.Fu
        self.Qxu_hat_fn = Function('Qxu_hat',[self.xl,self.Wl,self.Vxx],[self.Qxu_hat],['xl0','Wl0','Vxx0'],['Qxu_hat_f'])
        self.Quu_bar    = self.luu #+ alpha*self.dFuVxdu
        self.Quu_bar_fn = Function('Quu_bar',[self.xl,self.Wl,self.Vx,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1],[self.Quu_bar],['xl0','Wl0','Vx0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10'],['Quu_bar_f'])
        self.Quu_hat    = self.Fu.T@self.Vxx@self.Fu 
        self.Quu_hat_fn = Function('Quu_hat',[self.xl,self.Wl,self.Vxx],[self.Quu_hat],['xl0','Wl0','Vxx0'],['Quu_hat_f'])
        # hessians w.r.t. the hyperparameters
        self.lxp    = jacobian(self.lx,self.P_auto)
        self.lxp_fn = Function('lxp',[self.xl,self.Wl,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1],[self.lxp],['xl0','Wl0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10'],['lxpf'])
        self.lup    = jacobian(self.lu,self.P_auto)
        self.lup_fn = Function('lup',[self.xl,self.Wl,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1],[self.lup],['xl0','Wl0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10'],['lupf'])
        self.lxNp   = jacobian(self.lxN,self.P_auto)
        self.lxNp_fn= Function('lxNp',[self.xl,self.ref_xl,self.P1_l],[self.lxNp],['xl0','refxl0','P1l0'],['lxNpf'])
    

    def Get_AuxSys_DDP(self,opt_sol1,Ref_xl,Ref_Wl,scxl_opt,scWl_opt,Y_l,Eta_l,weight1,p1):
        xl_opt   = opt_sol1['xl_opt']
        Wl_opt   = opt_sol1['Wl_opt']
        LxNp  = self.lxNp_fn(xl0=xl_opt[-1,:],refxl0=Ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl],P1l0=weight1)['lxNpf'].full()
        LxxN  = self.lxxN_fn(P1l0=weight1)['lxxNf'].full()
        Lxp = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        Lup = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        for k in range(self.N):
            Lxp[k] = self.lxp_fn(xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                scxl0=scxl_opt[k,:],scxL0=Y_l[:,k],
                                scWl0=scWl_opt[k,:],scWL0=Eta_l[:,k],P1l0=weight1,p10=p1)['lxpf'].full()
            Lup[k] = self.lup_fn(xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                scxl0=scxl_opt[k,:],scxL0=Y_l[:,k],
                                scWl0=scWl_opt[k,:],scWL0=Eta_l[:,k],P1l0=weight1,p10=p1)['lupf'].full()
        
        auxSys1 = { "HxxN":LxxN,
                    "HxNp":LxNp,
                    "Hxp":Lxp,
                    "Hup":Lup
                    }
        
        return auxSys1


    def DDP_ADMM_Subp1(self,xl_0,Ref_xl,Ref_Wl,weight1,scxl_opt_hat,scWl_opt_hat,Y_l,Eta_l,p1,max_iter,e_tol):
        reg = 1e-7 # Regularization term
        alpha_init = 1 # Initial alpha for line search
        alpha_min = 1e-2  # Minimum allowable alpha
        alpha_factor = 0.5 # 
        max_line_search_steps = 5
        iteration = 1
        ratio = 10
        X_nominal = np.zeros((self.n_xl,self.N+1))
        U_nominal = np.zeros((self.n_Wl,self.N))
        X_nominal[:,0:1] = np.reshape(xl_0,(self.n_xl,1))
        
        # Initial trajectory and initial cost 
        cost_prev = 0
        for k in range(self.N):
            u_k    = np.reshape(Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],(self.n_Wl,1))
            X_nominal[:,k+1:k+2] = self.MDynl_fn_admm(xl0=X_nominal[:,k],Wl0=u_k)['MDynlf_admm'].full()
            # X_nominal[:,k+1:k+2] = np.reshape(Ref_xl[k*self.n_xl:(k+1)*self.n_xl],(self.n_xl,1))
            U_nominal[:,k:k+1]   = u_k
            cost_prev     += self.Jl_kfn_admm(xl0=X_nominal[:,k],Wl0=u_k,refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                             scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                             scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1)['Jl_kf_admm'].full()
        cost_prev += self.Jl_Nfn_admm(xl0=X_nominal[:,-1],refxl0=Ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl],P1l0=weight1)['Jl_Nf_admm'].full()
        
        while ratio>e_tol and iteration<=max_iter:
            Qxx_bar     = self.N*[np.zeros((self.n_xl,self.n_xl))]
            Qxu_bar     = self.N*[np.zeros((self.n_xl,self.n_Wl))]
            Quu_bar     = self.N*[np.zeros((self.n_Wl,self.n_Wl))]
            Qxu         = self.N*[np.zeros((self.n_xl,self.n_Wl))]
            Quuinv      = self.N*[np.zeros((self.n_Wl,self.n_Wl))]
            Fx      = self.N*[np.zeros((self.n_xl,self.n_xl))]
            Fu      = self.N*[np.zeros((self.n_xl,self.n_Wl))]
            Vx      = (self.N+1)*[np.zeros((self.n_xl,1))]
            Vxx     = (self.N+1)*[np.zeros((self.n_xl,self.n_xl))]
            Vx[self.N] = self.lxN_fn(xl0=X_nominal[:,self.N],refxl0=Ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl],P1l0=weight1)['lxNf'].full()
            Vxx[self.N]= self.lxxN_fn(P1l0=weight1)['lxxNf'].full()
            # list of the control gains 
            K_fb    = self.N*[np.zeros((self.n_Wl,self.n_xl))] # feedback
            k_ff    = self.N*[np.zeros((self.n_Wl,1))] # feedforward
            # backward pass
            for k in reversed(range(self.N)): # N-1, N-2,...,0
                Qx_k  = self.Qx_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                    scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1)['Qxf'].full()
                Qu_k  = self.Qu_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                    scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1)['Quf'].full()
                Qxx_bar_k = self.Qxx_bar_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                    scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1)['Qxx_bar_f'].full()
                Qxx_hat_k = self.Qxx_hat_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vxx0=Vxx[k+1])['Qxx_hat_f'].full()
                Qxx_k     = Qxx_bar_k + Qxx_hat_k
                Qxu_bar_k = self.Qxu_bar_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                    scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1)['Qxu_bar_f'].full()
                Qxu_hat_k = self.Qxu_hat_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vxx0=Vxx[k+1])['Qxu_hat_f'].full()
                Qxu_k     = Qxu_bar_k + Qxu_hat_k
                Quu_bar_k = self.Quu_bar_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                    scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1)['Quu_bar_f'].full()
                Quu_hat_k = self.Quu_hat_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vxx0=Vxx[k+1])['Quu_hat_f'].full()
                Quu_k     = Quu_bar_k + Quu_hat_k
                # Quu_inv = solve(Quu_k+reg*np.identity(self.n_Wl),np.identity(self.n_Wl))
                Quu_inv = LA.inv(Quu_k) # more stable than 'solve'
                
                # compute the control gains
                K_fb[k]  = -Quu_inv@Qxu_k.T
                k_ff[k]  = -Quu_inv@Qu_k
                # compute the derivatives of the value function
                Vx[k]    = Qx_k - Qxu_k@Quu_inv@Qu_k
                Vxx[k]   = Qxx_k - Qxu_k@Quu_inv@Qxu_k.T
                Fx[k]    = self.Fx_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k])['Fx_f'].full()
                Fu[k]    = self.Fu_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k])['Fu_f'].full()
                Qxx_bar[k]   = Qxx_bar_k
                Qxu_bar[k]   = Qxu_bar_k
                Quu_bar[k]   = Quu_bar_k
                Quuinv[k]    = Quu_inv
                Qxu[k]       = Qxu_k
            # forward pass with adaptive alpha (line search), adaptive alpha makes the DDP more stable!
            alpha = alpha_init
            for i in range(max_line_search_steps):
                X_new = np.zeros((self.n_xl,self.N+1))
                U_new = np.zeros((self.n_Wl,self.N))
                X_new[:,0:1] = np.reshape(xl_0,(self.n_xl,1))
                cost_new = 0
                for k in range(self.N):
                    delta_x = np.reshape(X_new[:,k] - X_nominal[:,k],(self.n_xl,1))
                    u_1     = np.reshape(U_nominal[:,k],(self.n_Wl,1))
                    u_2     = K_fb[k]@delta_x
                    u_3     = alpha*k_ff[k]
                    u_k     = u_1 + u_2 + u_3
                    u_k     = np.reshape(u_k,(self.n_Wl,1))
                    X_new[:,k+1:k+2]  = self.MDynl_fn_admm(xl0=np.reshape(X_new[:,k],(self.n_xl,1)),Wl0=u_k)['MDynlf_admm'].full()
                    U_new[:,k:k+1]    = u_k
                    cost_new   += self.Jl_kfn_admm(xl0=X_new[:,k],Wl0=u_k,refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                             scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                             scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1)['Jl_kf_admm'].full()
                cost_new   += self.Jl_Nfn_admm(xl0=X_new[:,-1],refxl0=Ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl],P1l0=weight1)['Jl_Nf_admm'].full()
                # Check if the cost decreased
                if cost_new < cost_prev:
                    # update the trajectories
                    X_nominal = X_new
                    U_nominal = U_new
                    break
                alpha = np.clip(alpha*alpha_factor,alpha_min,alpha_init)  # Reduce alpha if cost did not improve

            ratio = np.abs(cost_new-cost_prev)/np.abs(cost_prev)
            print('iteration:',iteration,'ratio=',ratio)
            cost_prev = cost_new
            iteration += 1
        
        opt_sol={"xl_opt":X_nominal.T,
                 "Wl_opt":U_nominal.T,
                 "Vxx":Vxx,
                 "Vx":Vx,
                 "K_FB":K_fb,
                 "Hxx":Qxx_bar,
                 "Qxu":Qxu,
                 "Hxu":Qxu_bar,
                 "Huu":Quu_bar,
                 "H_k_inv":Quuinv,
                 "Fx":Fx,
                 "Fu":Fu}
        return opt_sol

    def Get_AuxSys_DDP_NOreuse(self,opt_sol,Ref_xl,Ref_Wl,weight1,scxl_opt,scWl_opt,Y_l,Eta_l,p1):
        xl_opt  = opt_sol['xl_opt']
        Wl_opt  = opt_sol['Wl_opt']
        Vx      = opt_sol['Vx']
        Hxx     = self.N*[np.zeros((self.n_xl,self.n_xl))]
        Hxu     = self.N*[np.zeros((self.n_xl,self.n_Wl))]
        Huu     = self.N*[np.zeros((self.n_Wl,self.n_Wl))]
        Fx      = self.N*[np.zeros((self.n_xl,self.n_xl))]
        Fu      = self.N*[np.zeros((self.n_xl,self.n_Wl))]
        for k in range(self.N):
            Hxx[k] = self.Qxx_bar_fn(xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt[k,:],scxL0=Y_l[:,k],
                                    scWl0=scWl_opt[k,:],scWL0=Eta_l[:,k],P1l0=weight1,p10=p1)['Qxx_bar_f'].full()
            Hxu[k] = self.Qxu_bar_fn(xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt[k,:],scxL0=Y_l[:,k],
                                    scWl0=scWl_opt[k,:],scWL0=Eta_l[:,k],P1l0=weight1,p10=p1)['Qxu_bar_f'].full()
            Huu[k] = self.Quu_bar_fn(xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt[k,:],scxL0=Y_l[:,k],
                                    scWl0=scWl_opt[k,:],scWL0=Eta_l[:,k],P1l0=weight1,p10=p1)['Quu_bar_f'].full()
            Fx[k]    = self.Fx_fn(xl0=xl_opt[k,:],Wl0=Wl_opt[k,:])['Fx_f'].full()
            Fu[k]    = self.Fu_fn(xl0=xl_opt[k,:],Wl0=Wl_opt[k,:])['Fu_f'].full()
        
        AuxSys_No = {"Hxx":Hxx,
                     "Hxu":Hxu,
                     "Huu":Huu,
                     "Fx":Fx,
                     "Fu":Fu   
                    }
        return AuxSys_No


    def system_derivatives_SubP2_ADMM(self):
        # gradients of the Lagrangian (augmented cost function with the soft constraints)
        self.Lscxl        = jacobian(self.J2_k_soft,self.sc_xl)
        self.LscWl        = jacobian(self.J2_k_soft,self.sc_Wl)
        self.Lnv          = jacobian(self.J2_k_soft,self.nv)
        # hessians
        self.Lscxlscxl    = jacobian(self.Lscxl,self.sc_xl)
        self.Lscxlscxl_fn = Function('Lscxlscxl',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1],[self.Lscxlscxl],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10'],['Lscxlscxlf']) 
        self.LscxlscWl    = jacobian(self.Lscxl,self.sc_Wl)
        self.LscxlscWl_fn = Function('LscxlscWl',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1],[self.LscxlscWl],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10'],['LscxlscWlf'])
        self.Lscxlnv      = jacobian(self.Lscxl,self.nv)
        self.Lscxlnv_fn   = Function('Lscxlnv',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1],[self.Lscxlnv],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10'],['Lscxlnvf'])
        self.LscWlscWl    = jacobian(self.LscWl,self.sc_Wl)
        self.LscWlscWl_fn = Function('LscWlscWl',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1],[self.LscWlscWl],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10'],['LscWlscWlf'])
        self.LscWlnv      = jacobian(self.LscWl,self.nv)
        self.LscWlnv_fn   = Function('LscWlnv',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1],[self.LscWlnv],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10'],['LscWlnvf'])
        self.Lnvnv        = jacobian(self.Lnv,self.nv)
        self.Lnvnv_fn     = Function('Lnvnv',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1],[self.Lnvnv],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10'],['Lnvnvf'])
        # hessians w.r.t. the hyperparameters
        self.Lscxlp       = jacobian(self.Lscxl,self.P_auto)
        self.Lscxlp_fn    = Function('Lscxlp',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1],[self.Lscxlp],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10'],['Lscxlpf'])
        self.LscWlp       = jacobian(self.LscWl,self.P_auto)
        self.LscWlp_fn    = Function('LscWlp',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1],[self.LscWlp],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10'],['LscWlpf'])
        self.Lnvp         = jacobian(self.Lnv,self.P_auto)
        self.Lnvp_fn     = Function('Lnvp',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1],[self.Lnvp],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10'],['Lnvpf'])


    def Get_AuxSys_SubP2(self,opt_sol1,opt_sol2,scxL_opt,scWL_opt,weight2,p1):
        xl_opt   = opt_sol1['xl_opt']
        Wl_opt   = opt_sol1['Wl_opt']
        scxl_opt = opt_sol2['scxl_opt']
        scWl_opt = opt_sol2['scWl_opt']
        Tl_opt   = opt_sol2['Tl_opt']
        Lscxlscxl_l      = self.N*[np.zeros((self.n_xl,self.n_xl))]
        LscxlscWl_l      = self.N*[np.zeros((self.n_xl,self.n_Wl))]
        Lscxlnv_l        = self.N*[np.zeros((self.n_xl,self.n_nv))]
        LscWlscWl_l      = self.N*[np.zeros((self.n_Wl,self.n_Wl))]
        LscWlnv_l        = self.N*[np.zeros((self.n_Wl,self.n_nv))]
        Lnvnv_l          = self.N*[np.zeros((self.n_nv,self.n_nv))]
        Lscxlp_l         = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        LscWlp_l         = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        Lnvp_l           = self.N*[np.zeros((self.n_nv,self.n_Pauto))]
        for k in range(self.N):
            Lscxlscxl_l[k] = self.Lscxlscxl_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1)['Lscxlscxlf'].full()
            LscxlscWl_l[k] = self.LscxlscWl_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1)['LscxlscWlf'].full()
            Lscxlnv_l[k]   = self.Lscxlnv_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1)['Lscxlnvf'].full()
            LscWlscWl_l[k] = self.LscWlscWl_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1)['LscWlscWlf'].full()
            LscWlnv_l[k]   = self.LscWlnv_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1)['LscWlnvf'].full()
            Lnvnv_l[k]     = self.Lnvnv_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1)['Lnvnvf'].full()
            Lscxlp_l[k]    = self.Lscxlp_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1)['Lscxlpf'].full()
            LscWlp_l[k]    = self.LscWlp_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1)['LscWlpf'].full()
            Lnvp_l[k]      = self.Lnvp_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1)['Lnvpf'].full()

        auxSys2 = {
                    "Lscxlscxl_l":Lscxlscxl_l,
                    "LscxlscWl_l":LscxlscWl_l,
                    "Lscxlnv_l":Lscxlnv_l,
                    "LscWlscWl_l":LscWlscWl_l,
                    "LscWlnv_l":LscWlnv_l,
                    "Lnvnv_l":Lnvnv_l,
                    "Lscxlp_l":Lscxlp_l,
                    "LscWlp_l":LscWlp_l,
                    "Lnvp_l":Lnvp_l
        }
        return auxSys2
    
    def system_derivatives_SubP3_ADMM(self):
        scxl_update = self.p1*(self.xl - self.sc_xl)
        scWl_update = self.p1*(self.Wl - self.sc_Wl)
        self.dscxl_updatedp = jacobian(scxl_update,self.P_auto)
        self.dscxl_updatedp_fn = Function('dscxl_update',[self.xl,self.sc_xl],[self.dscxl_updatedp],['xl0','scxl0'],['dscxl_updatef'])
        self.dscWl_updatedp = jacobian(scWl_update,self.P_auto)
        self.dscWl_updatedp_fn = Function('dscWl_update',[self.Wl,self.sc_Wl],[self.dscWl_updatedp],['Wl0','scWl0'],['dscWl_updatef'])

    def Get_AuxSys_SubP3(self,opt_sol1,opt_sol2):
        xl_opt   = opt_sol1['xl_opt']
        Wl_opt   = opt_sol1['Wl_opt']
        scxl_opt = opt_sol2['scxl_opt']
        scWl_opt = opt_sol2['scWl_opt']
        dscxl_updatedp_l = self.N*[self.n_xl,self.n_Pauto]
        dscWl_updatedp_l = self.N*[self.n_Wl,self.n_Pauto]
        for k in range(self.N):
            dscxl_updatedp_l[k] = self.dscxl_updatedp_fn(xl0=xl_opt[k,:],scxl0=scxl_opt[k,:])['dscxl_updatef'].full()
            dscWl_updatedp_l[k] = self.dscWl_updatedp_fn(Wl0=Wl_opt[k,:],scWl0=scWl_opt[k,:])['dscWl_updatef'].full()
        
        auxSys3 = {
                    "dscxl_updp":dscxl_updatedp_l,
                    "dscWl_updp":dscWl_updatedp_l
                }
        return auxSys3


    def ADMM_forward_MPC_DDP(self,xl_fb,Ref_xl,Ref_Wl,weight1,weight2,p1):
        r_primal = 1e2 # primal residual
        r_dual   = 1e2 # dual residual
        i        = 1 # ADMM iteration index
        max_iter_DDP = 5
        e_tol    = 1e-2
        xl_traj  = Ref_xl[0:self.N*self.n_xl]
        Wl_traj  = Ref_Wl
        xw       = np.concatenate((xl_traj,Wl_traj)) # collection of all local states and controls
        scxl_opt1 = np.zeros(((self.N)*self.n_xl)) # this initial guess is very important, the worse the initial guess, the larger the initial loss!
        scWl_opt = np.zeros((self.N*self.n_Wl)) # this initial guess is very important
        for k in range(self.N):
            u_k    = np.reshape(Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],(self.n_Wl,1)) # this initial guess is very important
            scxl_opt1[(k)*self.n_xl:(k+1)*self.n_xl] = np.reshape(self.MDynl_fn_admm(xl0=scxl_opt1[k*self.n_xl:(k+1)*self.n_xl],Wl0=u_k)['MDynlf_admm'].full(),self.n_xl)
            scWl_opt[k*self.n_Wl:(k+1)*self.n_Wl] = np.reshape(u_k,self.n_Wl)
        scxl_opt = scxl_opt1[0:self.N*self.n_xl]
        sc_xw    = np.concatenate((scxl_opt,scWl_opt)) # collection of all local safe copy states and controls
        norm_xw  = np.array([LA.norm(xw),LA.norm(sc_xw)])
        e_pri    = np.sqrt(2)*self.e_abs + self.e_rel * np.max(norm_xw)
        e_dual   = np.sqrt(2)*self.e_abs + self.e_rel * (p1*LA.norm(scxl_opt)+p1*LA.norm(scWl_opt))
        Y        = np.zeros((self.n_xl,self.N)) # Lagrangian multiplier trajectory associated with the safe copy state
        Eta      = np.zeros((self.n_Wl,self.N)) # Lagrangian multiplier trajectory associated with the safe copy control
        self.max_iter_ADMM = 2
        Opt_Sol1 = []
        Opt_Sol2 = []
        Opt_Y    = []
        Opt_Eta  = []
        for i_admm in range(int(self.max_iter_ADMM)):
            Y_l   = np.reshape(Y.T,self.N*self.n_xl) # old state Lagrangian multiplier trajectory
            Eta_l = np.reshape(Eta.T,self.N*self.n_Wl) # old control Lagrangian multiplier trajectory
            # solve Subproblem 1
            start_time = TM.time()
            opt_sol1   = self.DDP_ADMM_Subp1(xl_fb,Ref_xl,Ref_Wl,weight1,scxl_opt,scWl_opt,Y_l,Eta_l,p1,max_iter_DDP,e_tol)
            mpctime    = (TM.time() - start_time)*1000
            print("subprblem1:--- %s ms ---" % format(mpctime,'.2f'))
            xl_opt   = opt_sol1['xl_opt']
            xl_optr  = np.reshape(xl_opt,(self.N+1)*self.n_xl)
            xl_traj  = xl_optr[0:self.N*self.n_xl]
            Wl_opt   = opt_sol1['Wl_opt']
            Wl_traj  = np.reshape(Wl_opt,self.N*self.n_Wl)
            # solve Subproblem 2
            para2    = np.concatenate((weight2,xl_traj))
            para2    = np.concatenate((para2,Y_l))
            para2    = np.concatenate((para2,Wl_traj))
            para2    = np.concatenate((para2,Eta_l))
            para2    = np.concatenate((para2,[p1]))
            start_time = TM.time()
            opt_sol2 = self.ADMM_SubP2(para2)
            mpctime    = (TM.time() - start_time)*1000
            print("subprblem2:--- %s ms ---" % format(mpctime,'.2f'))
            scxl_traj= np.reshape(opt_sol2['scxl_opt'],self.N*self.n_xl) # new safe copy state trajectory
            scWl_traj= np.reshape(opt_sol2['scWl_opt'],self.N*self.n_Wl) # new safe copy control trajectory
            # solve Subproblem 3
            Y_new, Eta_new   = self.ADMM_SubP3(Y,Eta,np.reshape(xl_traj,(self.N,self.n_xl)),np.reshape(scxl_traj,(self.N,self.n_xl)),
                                       np.reshape(Wl_traj,(self.N,self.n_Wl)),np.reshape(scWl_traj,(self.N,self.n_Wl)),p1)
            # compute residual (for adaptive penalty parameters)
            r_px        = LA.norm(xl_traj-scxl_traj)
            r_dx        = LA.norm(p1*(scxl_traj-scxl_opt))
            r_pw        = LA.norm(Wl_traj-scWl_traj)
            r_dw        = LA.norm(p1*(scWl_traj-scWl_opt))
            r_primal    = np.sqrt(r_px**2+r_pw**2)
            r_dual      = np.sqrt(r_dx**2+r_dw**2)
            # update the safe copy trajectories, Lagrangian multipliers, and error
            scxl_opt = scxl_traj 
            scWl_opt = scWl_traj
            Y        = Y_new
            Eta      = Eta_new
            sc_xw    = np.concatenate((scxl_opt,scWl_opt))
            xw       = np.concatenate((xl_traj,Wl_traj))
            norm_xw  = np.array([LA.norm(xw),LA.norm(sc_xw)])
            e_pri    = np.sqrt(2)*self.e_abs + self.e_rel * np.max(norm_xw)
            e_dual   = np.sqrt(2)*self.e_abs + self.e_rel * (p1*LA.norm(scxl_opt)+p1*LA.norm(scWl_opt))
            print('ADMM iteration:',i,'r_primal=',r_primal,'r_dual=',r_dual,'e_pri=',e_pri,'e_dual=',e_dual,'p1=',p1)
            Opt_Sol1 += [opt_sol1]
            Opt_Sol2 += [opt_sol2]
            Opt_Y    += [Y]
            Opt_Eta  += [Eta]

        return Opt_Sol1, Opt_Sol2, Opt_Y, Opt_Eta
    
    def DDP_Gradient(self,opt_sol,auxSys1, scxl_grad, scxL_grad, scWl_grad, scWL_grad, p1):
        Huuinv, Qxu, K_fb, F, G  = opt_sol['H_k_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSys1['HxNp'], auxSys1['Hxp'], auxSys1['Hup']
        S          = (self.N+1)*[np.zeros((self.n_xl,self.n_Pauto))]
        S[self.N]  = HxNp
        v_FF       = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        xl_grad    = (self.N+1)*[np.zeros((self.n_xl,self.n_Pauto))] 
        Wl_grad    = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): 
            Hxp_k    = Hxp[k] + scxL_grad[k] - p1*np.identity(self.n_xl)@scxl_grad[k]
            Hup_k    = Hup[k] + scWL_grad[k] - p1*np.identity(self.n_Wl)@scWl_grad[k]
            v_FF[k]  = -Huuinv[k]@(Hup_k + G[k].T@S[k+1])
            S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
        #-------Foreward recursion-------#
        for k in range(self.N):
            Wl_grad[k]  = K_fb[k]@xl_grad[k]+v_FF[k]
            xl_grad[k+1]= F[k]@xl_grad[k]+G[k]@Wl_grad[k]

        grad_out ={"xl_grad":xl_grad,
                   "Wl_grad":Wl_grad
                }
        
        return grad_out
    

    def PDP_Gradient_NOreuse(self,auxsys_No,auxSys1, scxl_grad, scxL_grad, scWl_grad, scWL_grad, p1):
        Hxx, Hxu, Huu, F, G  = auxsys_No['Hxx'], auxsys_No['Hxu'], auxsys_No['Huu'], auxsys_No['Fx'], auxsys_No['Fu']
        HxxN, HxNp, Hxp, Hup = auxSys1['HxxN'], auxSys1['HxNp'], auxSys1['Hxp'], auxSys1['Hup']
        P          = self.N*[np.zeros((self.n_xl,self.n_xl))]
        S          = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        A          = self.N*[np.zeros((self.n_xl,self.n_xl))]
        R          = self.N*[np.zeros((self.n_xl,self.n_xl))]
        M_p        = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        invHuu     = self.N*[np.zeros((self.n_Wl,self.n_Wl))]
        invIRP     = self.N*[np.zeros((self.n_xl,self.n_xl))]
        P[self.N-1]  = HxxN
        S[self.N-1]  = HxNp
        xl_grad    = (self.N+1)*[np.zeros((self.n_xl,self.n_Pauto))] 
        Wl_grad    = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        I          = np.identity(self.n_xl)
        for k in range(self.N-1,0,-1):
            P_next = P[k]
            S_next = S[k]
            invHuu[k] = LA.inv(Huu[k])
            GinvHuu= G[k]@invHuu[k]
            HxuinvHuu = Hxu[k]@invHuu[k]
            A[k]    = F[k]-GinvHuu@Hxu[k].T
            R[k]    = GinvHuu@G[k].T
            M_p[k]  = -GinvHuu@Hup[k]
            Q_k    = Hxx[k]-HxuinvHuu@Hxu[k].T
            N_p_k  = Hxp[k]+ scxL_grad[k] - p1*np.identity(self.n_xl)@scxl_grad[k] - HxuinvHuu@(Hup[k]+ scWL_grad[k] - p1*np.identity(self.n_Wl)@scWl_grad[k])
            invIRP[k]=LA.inv(I+R[k]@P_next)
            temp_mat = A[k].T@P_next@invIRP[k]
            P_curr = Q_k + temp_mat@A[k]
            S_curr = temp_mat@(M_p[k] - R[k]@S_next) + A[k].T@S_next + N_p_k
            P[k-1] = P_curr
            S[k-1] = S_curr
        
        for k in range(self.N):
            temp_mat2 = G[k].T@P[k]@invIRP[k]
            Wl_grad[k]  = -invHuu[k]@((Hxu[k].T+temp_mat2@A[k])@xl_grad[k] + temp_mat2 @ (M_p[k]- R[k]@ S[k]) + G[k].T@S[k] + Hup[k])
            xl_grad[k+1] = F[k]@xl_grad[k] + G[k]@Wl_grad[k]

        grad_out ={"xl_grad":xl_grad,
                   "Wl_grad":Wl_grad
                }
        
        return grad_out
    
    
    def SubP2_Gradient(self,auxSys2,grad_out,scxL_grad,scWL_grad,p1):
        xl_grad      = grad_out['xl_grad']
        Wl_grad      = grad_out['Wl_grad']
        Lscxlscxl_l, LscxlscWl_l, Lscxlnv_l = auxSys2['Lscxlscxl_l'], auxSys2['LscxlscWl_l'], auxSys2['Lscxlnv_l']
        LscWlscWl_l, LscWlnv_l              = auxSys2['LscWlscWl_l'], auxSys2['LscWlnv_l']
        Lnvnv_l                             = auxSys2['Lnvnv_l']
        Lscxlp_l,    LscWlp_l,    Lnvp_l    = auxSys2['Lscxlp_l'], auxSys2['LscWlp_l'], auxSys2['Lnvp_l']
        scxl_grad    = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        scWl_grad    = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        nv_grad      = self.N*[np.zeros((self.n_nv,self.n_Pauto))]
        for k in range(self.N):
            L_hessian_k = vertcat(
                            horzcat(Lscxlscxl_l[k] + p1*np.identity(self.n_xl),  LscxlscWl_l[k],Lscxlnv_l[k]),
                            horzcat(LscxlscWl_l[k].T,LscWlscWl_l[k] + p1*np.identity(self.n_Wl),LscWlnv_l[k]),
                            horzcat(Lscxlnv_l[k].T,  LscWlnv_l[k].T,Lnvnv_l[k])
                            )
            L_trajp_k   = vertcat(
                            Lscxlp_l[k] - np.identity(self.n_xl)@scxL_grad[k] - p1*np.identity(self.n_xl)@xl_grad[k],
                            LscWlp_l[k] - np.identity(self.n_Wl)@scWL_grad[k] - p1*np.identity(self.n_Wl)@Wl_grad[k],
                            Lnvp_l[k]
                            )
            # L_hessian_k_inv = solve(L_hessian_k + reg*np.identity(self.n_xl+self.n_Wl+self.n_nv),np.identity(self.n_xl+self.n_Wl+self.n_nv))
            L_hessian_k_inv = LA.inv(L_hessian_k)
            grad_subp2_k = -L_hessian_k_inv@L_trajp_k
            scxl_grad[k] = grad_subp2_k[0:self.n_xl,:]
            scWl_grad[k] = grad_subp2_k[self.n_xl:(self.n_xl+self.n_Wl),:]
            nv_grad[k]   = grad_subp2_k[(self.n_xl+self.n_Wl):,:]
            
        grad_out2 = {"scxl_grad":scxl_grad,
                     "scWl_grad":scWl_grad,
                     "nv_grad":nv_grad
                    }
        return grad_out2
    
        
    def SubP3_Gradient(self,auxSys3,grad_out,grad_out2,scxL_grad,scWL_grad,p1):
        xl_grad      = grad_out['xl_grad']
        Wl_grad      = grad_out['Wl_grad']
        scxl_grad    = grad_out2['scxl_grad']
        scWl_grad    = grad_out2['scWl_grad']
        dscxl_updp   = auxSys3['dscxl_updp']
        dscWl_updp   = auxSys3['dscWl_updp']
        Y_grad_new   = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        Eta_grad_new = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]

        for k in range(self.N):
            Y_grad_k   = scxL_grad[k] # old Lagrangian gradient associated with the load's state xl
            Eta_grad_k = scWL_grad[k] # old Lagrangian gradient associated with the load's control Wl
            Y_grad_new[k]   = Y_grad_k + p1*(xl_grad[k] - scxl_grad[k]) + dscxl_updp[k]
            Eta_grad_new[k] = Eta_grad_k + p1*(Wl_grad[k] - scWl_grad[k]) + dscWl_updp[k]
        
        grad_out3 = {
                     "scxL_grad":Y_grad_new,
                     "scWL_grad":Eta_grad_new
                    } 
        
        return grad_out3

    def ADMM_Gradient_Solver(self,Opt_Sol1,Opt_Sol2,Opt_Y,Opt_Eta,Ref_xl,Ref_Wl,weight1,weight2,p1):
        # initialize the gradient trajectories of SubP2 and SubP3
        scxl_grad = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        scWl_grad = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        scxL_grad = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        scWL_grad = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        # initial trajectories, same as those used in the ADMM recursion in the forward pass
        scxl_opt  = np.zeros((self.N+1,self.n_xl))
        scWl_opt  = np.zeros((self.N,self.n_Wl))
        for k in range(self.N):
            u_k   = np.reshape(Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],(self.n_Wl,1))
            scxl_opt[k+1:k+2,:] = np.reshape(self.MDynl_fn_admm(xl0=scxl_opt[k,:],Wl0=u_k)['MDynlf_admm'].full(),(1,self.n_xl))
            scWl_opt[k:k+1,:] = np.reshape(u_k,(1,self.n_Wl))
        Y         = np.zeros((self.n_xl,self.N)) # Lagrangian multiplier trajectory associated with the safe copy state
        Eta       = np.zeros((self.n_Wl,self.N)) # Lagrangian multiplier trajectory associated with the safe copy control
        # lists for storing gradient trajectories
        Grad_Out1 = []
        Grad_Out2 = []
        Grad_Out3 = []
        GradTime  = []
        GradTimeNO= []
        Meanerror = []
        AuxTime1  = []
        AuxTime2NO= []
        for i_admm in range(int(self.max_iter_ADMM)):
            # gradients of Subproblem1
            opt_sol1  = Opt_Sol1[i_admm]
            start_time = TM.time()
            auxSys1   = self.Get_AuxSys_DDP(opt_sol1,Ref_xl,Ref_Wl,scxl_opt,scWl_opt,Y,Eta,weight1,p1)
            auxtime1    = (TM.time() - start_time)*1000
            print("auxSys1time:--- %s ms ---" % format(auxtime1,'.2f'))
            start_time = TM.time()
            grad_out  = self.DDP_Gradient(opt_sol1,auxSys1, scxl_grad, scxL_grad, scWl_grad, scWL_grad, p1)
            gradtimeRe    = (TM.time() - start_time)*1000
            print("g_reuse:--- %s ms ---" % format(gradtimeRe,'.2f'))
            start_time = TM.time()
            axuSys1No  = self.Get_AuxSys_DDP_NOreuse(opt_sol1,Ref_xl,Ref_Wl,weight1,scxl_opt,scWl_opt,Y,Eta,p1)
            auxtime2    = (TM.time() - start_time)*1000
            print("auxSys1Notime:--- %s ms ---" % format(auxtime2,'.2f'))
            start_time = TM.time()
            grad_outNO  = self.PDP_Gradient_NOreuse(axuSys1No,auxSys1, scxl_grad, scxL_grad, scWl_grad, scWL_grad, p1)
            gradtimeNO    = (TM.time() - start_time)*1000
            print("g_PDPNoreuse:--- %s ms ---" % format(gradtimeNO,'.2f'))
            # gradients of Subproblem2
            opt_sol2  = Opt_Sol2[i_admm]
            auxSys2   = self.Get_AuxSys_SubP2(opt_sol1,opt_sol2,Y,Eta,weight2,p1)
            grad_out2 = self.SubP2_Gradient(auxSys2,grad_out,scxL_grad,scWL_grad,p1)
            # gradients of Subproblem3
            auxSys3   = self.Get_AuxSys_SubP3(opt_sol1,opt_sol2)
            grad_out3 = self.SubP3_Gradient(auxSys3,grad_out,grad_out2,scxL_grad,scWL_grad,p1)
            # update
            scxl_opt  = opt_sol2['scxl_opt']
            scWl_opt  = opt_sol2['scWl_opt']
            Y         = Opt_Y[i_admm]
            Eta       = Opt_Eta[i_admm]
            scxl_grad = grad_out2['scxl_grad']
            scWl_grad = grad_out2['scWl_grad']
            scxL_grad = grad_out3['scxL_grad']
            scWL_grad = grad_out3['scWL_grad']
            # save the results
            Grad_Out1 += [grad_out]
            Grad_Out2 += [grad_out2]
            Grad_Out3 += [grad_out3]
            GradTime  += [gradtimeRe]
            GradTimeNO+= [gradtimeNO]
            AuxTime1  += [auxtime1]
            AuxTime2NO+= [auxtime2]
            # error between two gradient trajectories
            xl_grad = grad_out['xl_grad']
            xl_gradNO = grad_outNO['xl_grad']
            Error   = 0
            for i in range(self.N):
                error = xl_grad[i+1] - xl_gradNO[i+1]
                Error += LA.norm(error,2)
            meanerror = Error/self.N
            print('meanerror=',meanerror)
            Meanerror += [meanerror]

        return Grad_Out1, Grad_Out2, Grad_Out3, GradTime, GradTimeNO, Meanerror, AuxTime1, AuxTime2NO

            

class Gradient_Solver:
    def __init__(self, sysm_para, horizon, xl, Wl, scxl, scWl, nv, P_auto, P_pinv, P_ns, e_abs, e_rel):
        self.n_xl   = xl.numel()
        self.n_Wl   = Wl.numel()
        self.n_nv   = nv.numel()
        self.n_P    = P_auto.numel()
        self.xl     = xl
        self.Wl     = Wl
        self.nv     = nv
        self.P_auto  = P_auto
        self.scxl   = scxl
        self.scxl_pre = SX.sym('scxl_pre',self.n_xl)
        self.scWl   = scWl
        self.scWl_pre = SX.sym('scWl_pre',self.n_Wl)
        self.xl_ref = SX.sym('x_ref',self.n_xl)
        self.Wl_prev= SX.sym('Wl_prev',self.n_Wl)
        self.Wl_next= SX.sym('Wl_next',self.n_Wl)
        self.nv_prev= SX.sym('nv_prev',self.n_nv)
        self.nv_next= SX.sym('nv_next',self.n_nv)
        self.nq     = int(sysm_para[8]) # number of the quadrotors
        self.cl0    = sysm_para[9] # cable length 
        self.N      = horizon
        self.P_pinv = P_pinv
        self.P_ns   = P_ns
        # Tolerances used in ADMM
        self.e_abs  = e_abs
        self.e_rel  = e_rel
        # boundaries of the hyperparameters
        self.p_min  = 1e-2
        self.p_max  = 1e2
        #------------- loss definition -------------#
        # tracking loss
        self.w_track = 1
        tracking_error  = self.xl - self.xl_ref
        self.loss_track = tracking_error.T@tracking_error
        # primal residual loss
        self.w_rp = 1
        self.w_rd = 0
        r_primal_x    = self.xl - self.scxl
        r_primal_w    = self.Wl - self.scWl
        self.loss_rp  = r_primal_x.T@r_primal_x + r_primal_w.T@r_primal_w
        # dual residual loss
        self.p        = self.P_auto[-1]
        r_dual_x      = self.scxl - self.scxl_pre
        r_dual_w      = self.scWl - self.scWl_pre
        self.loss_rd  = self.p*(r_dual_x.T@r_dual_x + r_dual_w.T@r_dual_w)
        # penalty of the smoothness of the quadrotors' trajectories
        self.w_smooth = 1e-3
        self.loss_smooth_0  = 0 # for k=0
        self.loss_smooth_k  = 0 # for 0<k<N-2
        self.loss_smooth_N2 = 0 # for k=N-2
        # quadrotors' positions at k
        self.qp_k = self.nq*[np.zeros((3,1))] # in the load's body frame
        t         = self.P_pinv@self.scWl + self.P_ns@self.nv
        for i in range(self.nq):
            ti   = t[(3*i):(3*(i+1))] # ith tension, a 3-by-1 vector, in the load's body frame
            self.qp_k[i]   = self.cl0*ti/norm_2(ti)
        # quadrotors' positions at k-1
        self.qp_prev = self.nq*[np.zeros((3,1))] # in the load's body frame
        t_prev    = self.P_pinv@self.Wl_prev + self.P_ns@self.nv_prev
        for i in range(self.nq):
            ti_prev   = t_prev[(3*i):(3*(i+1))] # ith tension, a 3-by-1 vector, in the load's body frame
            self.qp_prev[i]   = self.cl0*ti_prev/norm_2(ti_prev)
        # quadrotors' positions at k+1
        self.qp_next = self.nq*[np.zeros((3,1))] # in the load's body frame
        t_next    = self.P_pinv@self.Wl_next + self.P_ns@self.nv_next
        for i in range(self.nq):
            ti_next   = t_next[(3*i):(3*(i+1))] # ith tension, a 3-by-1 vector, in the load's body frame
            self.qp_next[i]   = self.cl0*ti_next/norm_2(ti_next)
        for i in range(self.nq):
            # smooth loss at k=0
            position_difference1 = self.qp_next[i] - self.qp_k[i]
            self.loss_smooth_0 += position_difference1.T@position_difference1
            # smooth loss at 0<k<N-2
            position_difference2 = self.qp_k[i] - self.qp_prev[i]
            self.loss_smooth_k += position_difference2.T@position_difference2 + position_difference1.T@position_difference1
            # smooth loss at k=N-2
            self.loss_smooth_N2 += position_difference2.T@position_difference2


    def Set_Parameters(self,tunable_para):
        weight       = np.zeros(self.n_P)
        for k in range(self.n_P):
            weight[k]= self.p_min + (self.p_max - self.p_min) * 1/(1+np.exp(-tunable_para[k]))

        return weight
    
    def ChainRule_Gradient(self,tunable_para):
        Tunable      = SX.sym('Tp',1,self.n_P)
        Weight       = SX.sym('wp',1,self.n_P)
        for k in range(self.n_P):
            Weight[k]= self.p_min + (self.p_max - self.p_min) * 1/(1 + exp(-Tunable[k]))
        dWdT         = jacobian(Weight,Tunable)
        dWdT_fn      = Function('dWdT',[Tunable],[dWdT],['Tp0'],['dWdT_f'])
        weight_grad  = dWdT_fn(Tp0=tunable_para)['dWdT_f'].full()

        return weight_grad

    def loss(self,Opt_Sol1,Opt_Sol2,Ref_xl,p1):
        xl_opt     = Opt_Sol1[1]['xl_opt']
        Wl_opt     = Opt_Sol1[1]['Wl_opt']
        scxl_opt   = Opt_Sol2[1]['scxl_opt']
        scWl_opt   = Opt_Sol2[1]['scWl_opt']
        scxl_opt_pre = Opt_Sol2[0]['scxl_opt']
        scWl_opt_pre = Opt_Sol2[0]['scWl_opt']
        Tl_opt     = Opt_Sol2[1]['Tl_opt']
        loss_track  = 0
        loss_rp     = 0
        loss_rd     = 0
        loss_smooth = 0
        for k in range(self.N):
            tracking_error = np.reshape(xl_opt[k,:],(self.n_xl,1)) - np.reshape(Ref_xl[k*self.n_xl:(k+1)*self.n_xl],(self.n_xl,1))
            loss_track    += tracking_error.T@tracking_error
            r_primal_x     = np.reshape(xl_opt[k,:],(self.n_xl,1)) - np.reshape(scxl_opt[k,:],(self.n_xl,1))
            r_primal_w     = np.reshape(Wl_opt[k,:],(self.n_Wl,1)) - np.reshape(scWl_opt[k,:],(self.n_Wl,1))
            r_dual_x       = p1*(np.reshape(scxl_opt[k,:],(self.n_xl,1)) - np.reshape(scxl_opt_pre[k,:],(self.n_xl,1)))
            r_dual_w       = p1*(np.reshape(scWl_opt[k,:],(self.n_Wl,1)) - np.reshape(scWl_opt_pre[k,:],(self.n_Wl,1)))
            loss_rp       += r_primal_x.T@r_primal_x + r_primal_w.T@r_primal_w 
            loss_rd       += r_dual_x.T@r_dual_x + r_dual_w.T@r_dual_w
        for k in range(self.N-1):
            wl_k  = np.reshape(scWl_opt[k,:],(6,1)) # 6-D wrench at the kth step
            nv_k  = np.reshape(Tl_opt[k,:],(3*self.nq-6,1)) # 6-D null-space vector at the kth step
            t_k   = self.P_pinv@wl_k + self.P_ns@nv_k # 12-D tension vector at the kth step in the load's body frame
            wl_k1  = np.reshape(scWl_opt[k+1,:],(6,1)) # 6-D wrench at the kth step
            nv_k1  = np.reshape(Tl_opt[k+1,:],(3*self.nq-6,1)) # 6-D null-space vector at the kth step
            t_k1   = self.P_pinv@wl_k1 + self.P_ns@nv_k1 # 12-D tension vector at the kth step in the load's body frame
            for i in range(self.nq):
                ti_k  = t_k[(3*i):(3*(i+1))] # ith tension, a 3-by-1 vector, in the load's body frame
                ti_k1 = t_k1[(3*i):(3*(i+1))]
                pi_k  = self.cl0*ti_k/LA.norm(ti_k)
                pi_k1 = self.cl0*ti_k1/LA.norm(ti_k1)
                position_difference = pi_k1 - pi_k
                loss_smooth        += position_difference.T@position_difference
            
        loss = self.w_track*loss_track + self.w_rp*loss_rp + self.w_rd*loss_rd + self.w_smooth*loss_smooth
        return loss
    
    
    def ChainRule(self,Opt_Sol1,Opt_Sol2,Ref_xl,Grad_Out1,Grad_Out2,p1):
        dltdxl          = jacobian(self.loss_track,self.xl)
        dltdxl_fn       = Function('dltdxl',[self.xl,self.xl_ref],[dltdxl],['xl0','refxl0'],['dltdxl_f'])
        dlrpdxl         = jacobian(self.loss_rp,self.xl)
        dlrpdxl_fn      = Function('dlrpdxl',[self.xl,self.scxl],[dlrpdxl],['xl0','scxl0'],['dlrpdxl_f'])
        dlrpdscxl       = jacobian(self.loss_rp,self.scxl)
        dlrpdscxl_fn    = Function('dlrpdscxl',[self.xl,self.scxl],[dlrpdscxl],['xl0','scxl0'],['dlrpdscxl_f'])
        dlrpdWl         = jacobian(self.loss_rp,self.Wl)
        dlrpdWl_fn      = Function('dlrpdWl',[self.Wl,self.scWl],[dlrpdWl],['Wl0','scWl0'],['dlrpdWl_f'])
        dlrpdscWl       = jacobian(self.loss_rp,self.scWl)
        dlrpdscWl_fn    = Function('dlrpdscWl',[self.Wl,self.scWl],[dlrpdscWl],['Wl0','scWl0'],['dlrpdscWl_f'])
        dlrddscxl       = jacobian(self.loss_rd,self.scxl)
        dlrddscxl_fn    = Function('dlrddscxl',[self.scxl,self.scxl_pre,self.p],[dlrddscxl],['scxl0','scxlpre0','p0'],['dlrddscxl_f'])
        dlrddscxl_pre   = jacobian(self.loss_rd,self.scxl_pre)
        dlrddscxl_pre_fn= Function('dlrddscxl_pre',[self.scxl,self.scxl_pre,self.p],[dlrddscxl_pre],['scxl0','scxlpre0','p0'],['dlrddscxl_pre_f'])
        dlrddscWl       = jacobian(self.loss_rd,self.scWl)
        dlrddscWl_fn    = Function('dlrddscWl',[self.scWl,self.scWl_pre,self.p],[dlrddscWl],['scWl0','scWlpre0','p0'],['dlrddscWl_f'])
        dlrddscWl_pre   = jacobian(self.loss_rd,self.scWl_pre)
        dlrddscWl_pre_fn= Function('dlrddscWl_pre',[self.scWl,self.scWl_pre,self.p],[dlrddscWl_pre],['scWl0','scWlpre0','p0'],['dlrddscWl_pre_f'])
        dlrddp          = jacobian(self.loss_rd,self.P_auto)
        dlrddp_fn       = Function('dlrddp',[self.scxl,self.scxl_pre,self.scWl,self.scWl_pre],[dlrddp],['scxl0','scxlpre0','scWl0','scWlpre0'],['dlrddp_f'])
        dlsdWl0   = jacobian(self.loss_smooth_0,self.scWl)
        dlsdWl0_fn= Function('dlsdWl0',[self.scWl,self.nv,self.Wl_next,self.nv_next],[dlsdWl0],['scWl0','nv0','Wl0_next','nv0_next'],['dlsdWl0_f'])
        dlsdWlk   = jacobian(self.loss_smooth_k,self.scWl)
        dlsdWlk_fn= Function('dlsdWlk',[self.scWl,self.nv,self.Wl_next,self.nv_next,self.Wl_prev,self.nv_prev],[dlsdWlk],['scWl0','nv0','Wl0_next','nv0_next','Wl0_prev','nv0_prev'],['dlsdWlk_f'])
        dlsdWlN   = jacobian(self.loss_smooth_N2,self.scWl)
        dlsdWlN_fn= Function('dlsdWlN',[self.scWl,self.nv,self.Wl_prev,self.nv_prev],[dlsdWlN],['scWl0','nv0','Wl0_prev','nv0_prev'],['dlsdWlN_f'])
        dlsdnv0   = jacobian(self.loss_smooth_0,self.nv)
        dlsdnv0_fn= Function('dlsdnv0',[self.scWl,self.nv,self.Wl_next,self.nv_next],[dlsdnv0],['scWl0','nv0','Wl0_next','nv0_next'],['dlsdnv0_f'])
        dlsdnvk   = jacobian(self.loss_smooth_k,self.nv)
        dlsdnvk_fn= Function('dlsdnvk',[self.scWl,self.nv,self.Wl_next,self.nv_next,self.Wl_prev,self.nv_prev],[dlsdnvk],['scWl0','nv0','Wl0_next','nv0_next','Wl0_prev','nv0_prev'],['dlsdnvk_f'])
        dlsdnvN   = jacobian(self.loss_smooth_N2,self.nv)
        dlsdnvN_fn= Function('dlsdnvN',[self.scWl,self.nv,self.Wl_prev,self.nv_prev],[dlsdnvN],['scWl0','nv0','Wl0_prev','nv0_prev'],['dlsdnvN_f'])
        dltdw     = 0
        dlrpdw    = 0
        dlrddw    = 0
        dlsdw     = 0
        xl_opt    = Opt_Sol1[1]['xl_opt']
        Wl_opt    = Opt_Sol1[1]['Wl_opt']
        scxl_opt  = Opt_Sol2[1]['scxl_opt']
        scWl_opt  = Opt_Sol2[1]['scWl_opt']
        Tl_opt    = Opt_Sol2[1]['Tl_opt']
        scxl_opt_pre = Opt_Sol2[0]['scxl_opt']
        scWl_opt_pre = Opt_Sol2[0]['scWl_opt']
        xl_grad   = Grad_Out1[1]['xl_grad']
        Wl_grad   = Grad_Out1[1]['Wl_grad']
        scxl_grad = Grad_Out2[1]['scxl_grad']
        scWl_grad = Grad_Out2[1]['scWl_grad']
        nv_grad   = Grad_Out2[1]['nv_grad']
        scxl_grad_pre = Grad_Out2[0]['scxl_grad']
        scWl_grad_pre = Grad_Out2[0]['scWl_grad']
        loss      = self.loss(Opt_Sol1,Opt_Sol2,Ref_xl,p1)
        for k in range(self.N):
            # gradient of the tracking errors
            dltdxl_k       = dltdxl_fn(xl0=xl_opt[k,:],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl])['dltdxl_f'].full()
            dltdw         += dltdxl_k@xl_grad[k]
            # gradient of the primal residuals
            dlrpdxl_k      = dlrpdxl_fn(xl0=xl_opt[k,:],scxl0=scxl_opt[k,:])['dlrpdxl_f'].full()
            dlrpdscxl_k    = dlrpdscxl_fn(xl0=xl_opt[k,:],scxl0=scxl_opt[k,:])['dlrpdscxl_f'].full()
            dlrpdWl_k      = dlrpdWl_fn(Wl0=Wl_opt[k,:],scWl0=scWl_opt[k,:])['dlrpdWl_f'].full()
            dlrpdscWl_k    = dlrpdscWl_fn(Wl0=Wl_opt[k,:],scWl0=scWl_opt[k,:])['dlrpdscWl_f'].full()
            dlrpdw        += dlrpdxl_k@xl_grad[k] + dlrpdscxl_k@scxl_grad[k] + dlrpdWl_k@Wl_grad[k] + dlrpdscWl_k@scWl_grad[k]
            # gradient of the dual residuals
            dlrddscxl_k    = dlrddscxl_fn(scxl0=scxl_opt[k,:],scxlpre0=scxl_opt_pre[k,:],p0=p1)['dlrddscxl_f'].full()
            dlrddscxl_pre_k= dlrddscxl_pre_fn(scxl0=scxl_opt[k,:],scxlpre0=scxl_opt_pre[k,:],p0=p1)['dlrddscxl_pre_f'].full()
            dlrddscWl_k    = dlrddscWl_fn(scWl0=scWl_opt[k,:],scWlpre0=scWl_opt_pre[k,:],p0=p1)['dlrddscWl_f'].full()
            dlrddscWl_pre_k= dlrddscWl_pre_fn(scWl0=scWl_opt[k,:],scWlpre0=scWl_opt_pre[k,:],p0=p1)['dlrddscWl_pre_f'].full()
            dlrddw_k       = dlrddp_fn(scxl0=scxl_opt[k,:],scxlpre0=scxl_opt_pre[k,:],scWl0=scWl_opt[k,:],scWlpre0=scWl_opt_pre[k,:])['dlrddp_f'].full()
            dlrddw        += dlrddscxl_k@scxl_grad[k] + dlrddscxl_pre_k@scxl_grad_pre[k] + dlrddscWl_k@scWl_grad[k] + dlrddscWl_pre_k@scWl_grad_pre[k] + dlrddw_k

            if k==0:
                dlsdWl_0 = dlsdWl0_fn(scWl0=scWl_opt[k,:],nv0=Tl_opt[k,:],Wl0_next=scWl_opt[k+1,:],nv0_next=Tl_opt[k+1,:])['dlsdWl0_f'].full()
                dlsdnv_0 = dlsdnv0_fn(scWl0=scWl_opt[k,:],nv0=Tl_opt[k,:],Wl0_next=scWl_opt[k+1,:],nv0_next=Tl_opt[k+1,:])['dlsdnv0_f'].full()
                dlsdw   += dlsdWl_0@scWl_grad[k] + dlsdnv_0@nv_grad[k]
            elif k>0 and k<self.N-1:
                dlsdWl_k = dlsdWlk_fn(scWl0=scWl_opt[k,:],nv0=Tl_opt[k,:],Wl0_next=scWl_opt[k+1,:],nv0_next=Tl_opt[k+1,:],Wl0_prev=scWl_opt[k-1,:],nv0_prev=Tl_opt[k-1,:])['dlsdWlk_f'].full()
                dlsdnv_k = dlsdnvk_fn(scWl0=scWl_opt[k,:],nv0=Tl_opt[k,:],Wl0_next=scWl_opt[k+1,:],nv0_next=Tl_opt[k+1,:],Wl0_prev=scWl_opt[k-1,:],nv0_prev=Tl_opt[k-1,:])['dlsdnvk_f'].full()
                dlsdw   += dlsdWl_k@scWl_grad[k] + dlsdnv_k@nv_grad[k]
            else:
                dlsdWl_N = dlsdWlN_fn(scWl0=scWl_opt[k,:],nv0=Tl_opt[k,:],Wl0_prev=scWl_opt[k-1,:],nv0_prev=Tl_opt[k-1,:])['dlsdWlN_f'].full()
                dlsdnv_N = dlsdnvN_fn(scWl0=scWl_opt[k,:],nv0=Tl_opt[k,:],Wl0_prev=scWl_opt[k-1,:],nv0_prev=Tl_opt[k-1,:])['dlsdnvN_f'].full()
                dlsdw   += dlsdWl_N@scWl_grad[k] + dlsdnv_N@nv_grad[k]

        dltdxl_N    = dltdxl_fn(xl0=xl_opt[-1,:],refxl0=Ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl])['dltdxl_f'].full()
        dltdw      += dltdxl_N@xl_grad[self.N]
        dldw        = self.w_track*dltdw + self.w_rp*dlrpdw + self.w_rd*dlrddw + self.w_smooth*dlsdw

        return dldw, loss
        
        
            





    
