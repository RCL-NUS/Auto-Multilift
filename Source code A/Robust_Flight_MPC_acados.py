"""
This file defines the flight controller for a multi-lifting system
-------------------------------------------------------------------------------------
Wang Bingheng, 1st version: 19 Jan. 2024 at Control and Simulation Lab, ECE Dept. NUS
               2nd version: 13 Mar. 2024 at Control and Simulation Lab, ECE Dept. NUS
"""

from casadi import *
import numpy as np
from numpy import linalg as LA
import math
from scipy.spatial.transform import Rotation as Rot
from scipy import linalg as sLA
from multiprocessing import Pool
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from os import system

class Controller:
    """
    Geometric flight controller on SE(3) [1][2]
    [1] Lee, T., Leok, M. and McClamroch, N.H., 2010. 
        Control of complex maneuvers for a quadrotor UAV using geometric methods on SE (3). 
        arXiv preprint arXiv:1003.2005.
    [2] Lee, T., Leok, M. and McClamroch, N.H., 2010, December. 
        Geometric tracking control of a quadrotor UAV on SE (3). 
        In 49th IEEE conference on decision and control (CDC) (pp. 5420-5425). IEEE.
    """
    def __init__(self, uav_para, dt_sample):
        # Quadrotor's inertial parameters (mass, rotational inertia)
        self.m    = uav_para[0] 
        self.J    = np.diag([uav_para[1], uav_para[2], uav_para[3]])
        self.nq   = int(uav_para[4]) # number of quadrotors
        # Unit direction vector free of coordinate
        self.ex   = np.array([[1, 0, 0]]).T
        self.ey   = np.array([[0, 1, 0]]).T
        self.ez   = np.array([[0, 0, 1]]).T
        # Gravitational acceleration
        self.g    = 9.81      
        self.dt   = dt_sample

    def skew_sym(self, v):
        v_cross = np.array([
            [0, -v[2, 0], v[1, 0]],
            [v[2, 0], 0, -v[0, 0]],
            [-v[1, 0], v[0, 0], 0]]
        )
        return v_cross

    def vee_map(self, v):
        vect = np.array([[v[2, 1], v[0, 2], v[1, 0]]]).T
        return vect

    def trace(self, v):
        v_trace = v[0, 0] + v[1, 1] + v[2, 2]
        return v_trace
    
    def lowpass_filter(self,time_const,curr_i,prev_i):
        alpha       = self.dt/(self.dt+time_const)
        y_filter    = (1-alpha)*prev_i + alpha*curr_i
        return y_filter
    
    def q_2_rotation(self, q): # from body frame to inertial frame
        q = q/LA.norm(q) # normalization
        q0, q1, q2, q3 = q[0,0], q[1,0], q[2,0], q[3,0] # q0 denotes a scalar while q1, q2, and q3 represent rotational axes x, y, and z, respectively
        R = np.array([
                      [2 * (q0 ** 2 + q1 ** 2) - 1, 2 * q1 * q2 - 2 * q0 * q3,   2 * q0 * q2 + 2 * q1 * q3],
                      [2 * q0 * q3 + 2 * q1 * q2,   2 * (q0 ** 2 + q2 ** 2) - 1, 2 * q2 * q3 - 2 * q0 * q1],
                      [2 * q1 * q3 - 2 * q0 * q2,   2 * q0 * q1 + 2 * q2 * q3,   2 * (q0 ** 2 + q3 ** 2) - 1]
                     ])
        return R
    
    def geometric_ctrl(self,x,v_prev,a_lpf_prev,j_lpf_prev,ref_p,ref_v,ref_a,ref_j,ref_s,b1_d,sum_e,ctrl_gain):
        # Control gain variables
        self.kp   = np.diag([ctrl_gain[0,0], ctrl_gain[0,1], ctrl_gain[0,2]])
        self.kv   = np.diag([ctrl_gain[0,3], ctrl_gain[0,4], ctrl_gain[0,5]])
        self.ki   = np.diag([ctrl_gain[0,6], ctrl_gain[0,7], ctrl_gain[0,8]])
        self.kr   = np.diag([ctrl_gain[0,9], ctrl_gain[0,10], ctrl_gain[0,11]]) # control gain for attitude tracking error
        self.kw   = np.diag([ctrl_gain[0,12], ctrl_gain[0,13], ctrl_gain[0,14]])
        # Get the system state from the feedback
        p  = np.array([[x[0,0], x[1,0], x[2,0]]]).T
        v  = np.array([[x[3,0], x[4,0], x[5,0]]]).T
        q  = np.array([[x[6,0], x[7,0], x[8,0], x[9,0]]]).T
        Rb = self.q_2_rotation(q) # rotation matrix from body frame to inertial frame
        """
        Position controller
        """
        # Trajectory tracking errors
        ep = p - ref_p
        ev = v - ref_v
        sum_e += self.dt*ep # approximation of the error integration
        # Desired force in inertial frame for the norminal dynamics
        Fd = -np.matmul(self.kp, ep) - np.matmul(self.kv, ev) -self.ki@sum_e + self.m*self.g*self.ez + self.m*ref_a
        # Desired total thruster force fd
        fd = np.inner(Fd.T, np.transpose(np.matmul(Rb, self.ez))) # norminal total thrust projected into the current body z axis
        """
        Attitude controller
        """
        # Construct the desired rotation matrix (from body frame to inertial frame)
        b3c = Fd/LA.norm(Fd) # b3c = -A/norm(A), so A = -Fd
        b2c = np.matmul(self.skew_sym(b3c), b1_d)/LA.norm(np.matmul(self.skew_sym(b3c), b1_d)) # b2c = -C/norm(C), so C = skew_sym(b1_d)@b3c
        b1c = np.matmul(self.skew_sym(b2c), b3c)
        Rbd = np.hstack((b1c, b2c, b3c))
        # Compute the desired angular velocity and angular acceleration，see Appendix F in the 2nd version of [1] for details
        a      = (v-v_prev)/self.dt # acclearation based on 1st-order backward differentiation
        time_const  = 0.025 # used in the low-pass filter for the acceleration
        a_lpf  = self.lowpass_filter(time_const,a,a_lpf_prev)
        j      = (a_lpf-a_lpf_prev)/self.dt # jerk based on 1st-order backward differentiation
        time_const  = 0.03 # used in the low-pass filter for the jerk
        j_lpf  = self.lowpass_filter(time_const,j,j_lpf_prev)
        A      = -Fd
        dA     = np.matmul(self.kp, ev) + np.matmul(self.kv, (a_lpf-ref_a)) - self.m*ref_j
        ddA    = np.matmul(self.kp, (a_lpf-ref_a)) + np.matmul(self.kv, (j_lpf-ref_j)) - self.m*ref_s
        db3c   = -dA/LA.norm(A) + np.inner(A.T,dA.T)*A/(LA.norm(A)**3)
        ddb3c  = -ddA/LA.norm(A) + 2*np.inner(A.T,dA.T)*dA/(LA.norm(A)**3) + (LA.norm(dA)**2+np.inner(A.T,ddA.T))*A/(LA.norm(A)**3) - 3*np.inner(A.T,dA.T)**2*A/(LA.norm(A)**5)
        C      = np.matmul(self.skew_sym(b1_d),b3c)
        dC     = np.matmul(self.skew_sym(b1_d),db3c)
        ddC    = np.matmul(self.skew_sym(b1_d),ddb3c)
        db2c   = -dC/LA.norm(C) + np.inner(C.T,dC.T)*C/(LA.norm(C)**3)
        ddb2c  = -ddC/LA.norm(C) + 2*np.inner(C.T,dC.T)*dC/(LA.norm(C)**3) + (LA.norm(dC)**2+np.inner(C.T,ddC.T))*C/(LA.norm(C)**3) - 3*np.inner(C.T,dC.T)**2*C/(LA.norm(C)**5)
        db1c   = np.matmul(self.skew_sym(db2c),b3c) + np.matmul(self.skew_sym(b2c),db3c)
        ddb1c  = np.matmul(self.skew_sym(ddb2c),b3c) + 2*np.matmul(self.skew_sym(db2c),db3c) + np.matmul(self.skew_sym(b2c),ddb3c)
        dRbd   = np.hstack((db1c, db2c, db3c))
        ddRbd  = np.hstack((ddb1c, ddb2c, ddb3c))
        omegad = self.vee_map(np.matmul(Rbd.T,dRbd))
        # Bound the desired angular rate for stability concern
        if LA.norm(omegad)>=10:
            omegad = omegad/LA.norm(omegad)*10
        domegad= self.vee_map(np.matmul(Rbd.T,ddRbd)-LA.matrix_power(self.skew_sym(omegad),2))
        omega  = np.array([[x[10,0], x[11,0], x[12,0]]]).T
        # attitude tracking errors
        er  = 1/2*self.vee_map(np.matmul(Rbd.T, Rb) - np.matmul(Rb.T, Rbd))
        ew  = omega - np.matmul(Rb.T, np.matmul(Rbd, omegad))
        # desired control torque
        tau = -np.matmul(self.kr, er) - np.matmul(self.kw, ew) \
            + np.matmul(self.skew_sym(omega), np.matmul(self.J, omega)) \
            - np.matmul(self.J, (np.matmul(np.matmul(self.skew_sym(omega), Rb.T), np.matmul(Rbd, omegad)) \
                - np.matmul(Rb.T, np.matmul(Rbd, domegad))))
        # control input
        u   = np.vstack((fd,tau))
        
        return u, a_lpf, j_lpf, sum_e
    
    def L1_adaptive_law(self,xi,z_hat):
        # Piecewise-constant adaptation law
        z       = np.array([[xi[3,0],xi[4,0],xi[5,0]]]).T
        q       = np.array([[xi[6,0],xi[7,0],xi[8,0],xi[9,0]]]).T
        Rb      = self.q_2_rotation(q) # rotation matrix from body frame to inertial frame
        B       = 1/self.m*Rb@self.ez
        Brp     = np.hstack((1/self.m*Rb@self.ex,1/self.m*Rb@self.ey))
        Bbar    = np.hstack((B,Brp))
        a_s     = np.array([[-5,-5,-5]]) 
        A_s     = np.diag(a_s[0]) # diagonal Hurwitz matrix
        PHI     = LA.inv(A_s)@(sLA.expm(self.dt*A_s)-np.identity(3))
        mu      = sLA.expm(self.dt*A_s)@(z_hat - z)
        d_hat   = -LA.inv(Bbar)@LA.inv(PHI)@mu
        dm_hat  = np.reshape(d_hat[0,0],(1,1))
        dum_hat = np.reshape(d_hat[1:3,0],(2,1))
        return dm_hat, dum_hat, A_s
     

    def system_ref(self, ref_a, ml, ref_al):
        # generate the reference state and control trajectories for tracking
        Fl_ref = ml*self.g*self.ez + ml*ref_al # 3-by-1 vector of the payload in {I}
        fl_ref = LA.norm(Fl_ref) 
        F_ref  = self.m*self.g*self.ez + self.m*ref_a + Fl_ref/self.nq # 3-by-1 vector in {I}. Compared to F_d, F_ref does not require any feedback information
        f_ref  = LA.norm(F_ref)  # magnitude of F_ref in ideal case when R = R_d
        # reset the desired attitude to identity matrix and the desired angular velocity to zeros (since the attitude only experiences small changes in flight)
        qd     = np.array([[1,0,0,0]]).T
        omegad = np.zeros((3,1))
        M_ref  = np.zeros((3,1))

        return qd, omegad, f_ref, fl_ref, M_ref
    


    
class MPC:
    def __init__(self, uav_para, load_para, cable_para, dt_ctrl, horizon, gamma, gamma2):
        # Quadrotor's inertial parameters (mass, rotational inertia)
        self.m     = uav_para[0] 
        self.J     = np.diag([uav_para[1], uav_para[2], uav_para[3]])
        self.nq    = int(uav_para[4])
        self.rq    = uav_para[5] # quadrotor's radius
        self.alpha = 2*np.pi/self.nq
        # Load's parameters
        self.ml    = load_para[0]
        self.rl    = load_para[1] # payload's radius
        # Cable's parameters
        self.cl0   = cable_para[3]
        # Unit direction vector free of coordinate
        self.ex    = np.array([[1, 0, 0]]).T
        self.ey    = np.array([[0, 1, 0]]).T
        self.ez    = np.array([[0, 0, 1]]).T
        # Gravitational acceleration
        self.g     = 9.81      
        self.dt    = dt_ctrl
        # MPC's horizon
        self.N     = horizon
        self.bp    = gamma
        self.bp2   = gamma2

    def SetStateVariable(self, xi, xq, xl, index_q):
        self.xi    = xi # quadrotor's state
        self.ind_q = index_q
        self.xq    = xq # matrix including all the quadrotors' states at a time step
        self.x_qi  = SX.sym('x_qi',2,self.nq-1) # all quadrotors' planar states (x and y) excluding the i-th quadrotor's state
        self.n_xi  = xi.numel()
        sc         = 1e16 # state constraint (infinity)
        wc         = 5 # angular velocity constraint for improving the stability
        self.xi_lb = self.n_xi*[-sc]
        self.xi_ub = self.n_xi*[sc]
        self.xl    = xl # payload's state
        self.n_xl  = xl.numel()
        self.xl_lb = self.n_xl*[-sc]
        self.xl_ub = self.n_xl*[sc]
    
    def SetCtrlVariable(self, ui, ul, ti):
        self.ui    = ui # quadrotor's control
        self.ti    = ti # tension magnitude for the i-th quadrotor
        self.ul    = ul # load's control
        self.n_ui  = ui.numel()
        self.n_ti  = ti.numel()
        self.n_ul  = ul.numel()
        self.ui_lb = [0.01, -0.5,-0.5,-0.5] # lower-bound
        self.ui_ub = [40, 0.5,0.5,0.5]   # upper-bound
        self.ul_lb = self.n_ul*[0.01]
        self.ul_ub = self.n_ul*[100]
        # set the soft control constraints for computing the gradient trajectories
        self.gui_lb, self.gui_ub  = 0,0
        self.gul_lb, self.gul_ub  = 0,0
        for i in range(self.n_ui):
            gui_min_i    = -self.bp2 * log(self.ui[i,0] - self.ui_lb[i])
            self.gui_lb += gui_min_i
            gui_max_i    = -self.bp2 * log(self.ui_ub[i] - self.ui[i,0])
            self.gui_ub += gui_max_i
        for i in range(self.n_ul):
            gul_min_i    = -self.bp2 * log(self.ul[i,0] - self.ul_lb[i])
            self.gul_lb += gul_min_i
            gul_max_i    = -self.bp2 * log(self.ul_ub[i] - self.ul[i,0])
            self.gul_ub += gul_max_i
    
    def SetLoadParameter(self, Jldiag, rg):
        self.Jldiag = Jldiag
        self.rg     = rg
        self.loadp  = vertcat(self.Jldiag,self.rg) # the payload's inertial parameters
        self.n_lp   = self.loadp.numel()

    def SetDyn(self, model_i, model_l, fi, fl):
        assert hasattr(self, 'xi'), "Define the quadrotor's state variable first!"
        assert hasattr(self, 'xl'), "Define the payload's state variable first!"
        assert hasattr(self, 'ui'), "Define the quadrotor's contrl variable first!"
        assert hasattr(self, 'ul'), "Define the payload's control variable first!"
        assert hasattr(self, 'Jldiag'), "Define the payload's rotation moment first!"
        assert hasattr(self, 'rg'), "Define the payload's CoM coordinate first!"
        self.Modeli   = self.xi + self.dt*model_i
        self.MDyni_fn = Function('MDyni',[self.xi, self.ui, self.xl, self.ti, self.ind_q],[self.Modeli],['xi0', 'ui0', 'xl0', 'ti0', 'i0'],['MDynif'])
        self.Modell   = self.xl + self.dt*model_l
        self.MDynl_fn = Function('MDynl',[self.xl, self.ul, self.xq, self.Jldiag, self.rg],[self.Modell],['xl0', 'ul0', 'xq0', 'Jl0', 'rg0'],['MDynlf'])
        self.f_i      = Function('fi',[self.xi, self.ui, self.xl, self.ti, self.ind_q],[fi],['xi0', 'ui0', 'xl0', 'ti0', 'i0'],['fif'])
        self.f_l      = Function('fl',[self.xl, self.ul, self.xq, self.Jldiag, self.rg],[fl],['xl0', 'ul0', 'xq0', 'Jl0', 'rg0'],['flf'])

    
    def SetLearnablePara(self):
        self.n_wsi    = 12 # dimension of the quadrotor state weightings
        self.n_wsl    = 12 # dimension of the payload state weightings
        self.para_i   = SX.sym('para_i',1,(2*self.n_wsi+self.n_ui)) 
        self.para_l   = SX.sym('para_l',1,(2*self.n_wsl+self.n_ul))
        self.n_pi     = self.para_i.numel()
        self.n_pl     = self.para_l.numel()

    
    def SetQuadrotorCostDyn(self):
        self.ref_xi   = SX.sym('ref_xi',self.n_xi,1)
        self.ref_ui   = SX.sym('ref_ui',self.n_ui,1)
        p_error_i     = self.xi[0:3,0] - self.ref_xi[0:3,0]
        v_error_i     = self.xi[3:6,0] - self.ref_xi[3:6,0]
        qi            = self.xi[6:10,0]
        ref_qi        = self.ref_xi[6:10,0]
        Ri            = self.q_2_rotation(qi)
        Rdi           = self.q_2_rotation(ref_qi)
        error_Ri      = Rdi.T@Ri - Ri.T@Rdi
        att_error_i   = 1/2*self.vee_map(error_Ri)
        w_error_i     = self.xi[10:13,0] - self.ref_xi[10:13,0]
        track_error_i = vertcat(p_error_i,v_error_i,att_error_i,w_error_i)
        ctrl_error_i  = self.ui - self.ref_ui
        self.Qi_k     = diag(self.para_i[0,0:self.n_wsi])
        self.Qi_N     = diag(self.para_i[0,self.n_wsi:2*self.n_wsi])
        self.Ri_k     = diag(self.para_i[0,2*self.n_wsi:2*self.n_wsi+self.n_ui])
        # path cost
        self.Ji_k     = 1/2 * (track_error_i.T@self.Qi_k@track_error_i + ctrl_error_i.T@self.Ri_k@ctrl_error_i)
        self.Ji_kfn   = Function('Ji_k',[self.xi, self.ui, self.ref_xi, self.ref_ui, self.para_i],[self.Ji_k],['xi0', 'ui0', 'refxi0', 'refui0', 'parai0'],['Ji_kf'])
        # terminal cost
        self.Ji_N     = 1/2 * track_error_i.T@self.Qi_N@track_error_i
        self.Ji_Nfn   = Function('Ji_N',[self.xi, self.ref_xi, self.para_i],[self.Ji_N],['xi0', 'refxi0', 'parai0'],['Ji_Nf'])
    

    def SetConstraints_Qaudrotor(self):
        # cable length constraint
        rli         = vertcat(self.rl*cos(self.ind_q*self.alpha),self.rl*sin(self.ind_q*self.alpha),0) # attachment point in {Bl}
        pik         = self.xi[0:3]
        pik_xy      = self.xi[0:2]
        plk         = self.xl[0:3]
        qlk         = self.xl[6:10]
        Rlk         = self.q_2_rotation(qlk)
        pli         = pik - plk - Rlk@rli
        self.hc_k   = 1/(2*self.bp) * (self.cl0 - norm_2(pli))**2
        self.hc_kfn = Function('hc_k',[self.xi, self.xl,self.ind_q],[self.hc_k],['xi0', 'xl0', 'i0'],['hc_kf'])
        # self.hcnb   = self.cl0 - norm_2(pli) # hard constraint, no barrier parameter
        # self.hcnb_fn= Function('hc_nbk',[self.xi, self.xl],[self.hcnb],['xi0', 'xl0'],['hc_nbkf'])
        # collision avoidance among quadrotors 
        self.gq_k   = 0
        
        for j in range(self.nq-1):
            xj      = self.x_qi[:,j]
            dis_ij  = pik_xy - xj
            gp_ijk  = -self.bp2 * log(norm_2(dis_ij) - 2*self.rq) 
            self.gq_k += gp_ijk
        self.gq_kfn = Function('gq_k',[self.xi, self.x_qi],[self.gq_k],['xi0', 'xqi0'],['gq_kf'])
        
        

    def SetPayloadCostDyn(self):
        self.ref_xl   = SX.sym('ref_xl',self.n_xl,1)
        self.ref_ul   = SX.sym('ref_ul',self.n_ul,1)
        p_error_l     = self.xl[0:3,0] - self.ref_xl[0:3,0]
        v_error_l     = self.xl[3:6,0] - self.ref_xl[3:6,0]
        ql            = self.xl[6:10,0]
        ref_ql        = self.ref_xl[6:10,0]
        Rl            = self.q_2_rotation(ql)
        Rdl           = self.q_2_rotation(ref_ql)
        error_Rl      = Rdl.T@Rl - Rl.T@Rdl
        att_error_l   = 1/2*self.vee_map(error_Rl)
        w_error_l     = self.xl[10:13,0] - self.ref_xl[10:13,0]
        track_error_l = vertcat(p_error_l,v_error_l,att_error_l,w_error_l)
        ctrl_error_l  = self.ul - self.ref_ul
        self.Ql_k     = diag(self.para_l[0,0:self.n_wsl])
        self.Ql_N     = diag(self.para_l[0,self.n_wsl:2*self.n_wsl])
        self.Rl_k     = diag(self.para_l[0,2*self.n_wsl:2*self.n_wsl+self.n_ul])
        # path cost
        self.Jl_k     = 1/2 * (track_error_l.T@self.Ql_k@track_error_l + ctrl_error_l.T@self.Rl_k@ctrl_error_l)
        self.Jl_kfn   = Function('Jl_k',[self.xl, self.ul, self.ref_xl, self.ref_ul, self.para_l],[self.Jl_k],['xl0', 'ul0', 'refxl0', 'reful0', 'paral0'],['Jl_kf'])
        # terminal cost
        self.Jl_N     = 1/2 * track_error_l.T@self.Ql_N@track_error_l
        self.Jl_Nfn   = Function('Jl_N',[self.xl, self.ref_xl, self.para_l],[self.Jl_N],['xl0', 'refxl0', 'paral0'],['Jl_Nf'])
    

    def SetConstraints_Load(self):
        # cable length constraint
        plk         = self.xl[0:3]
        qlk         = self.xl[6:10]
        Rlk         = self.q_2_rotation(qlk)
        self.hl_k   = 0
        for i in range(self.nq):
            rli         = vertcat(self.rl*cos(i*self.alpha),self.rl*sin(i*self.alpha),0) # attachment point in {Bl}
            xi          = self.xq[:,i]
            pik         = vertcat(xi[0,0],xi[1,0],xi[2,0])
            prli        = plk + Rlk@rli
            pli         = pik - prli
            self.hl_k  += 1/(2*self.bp) * (self.cl0 - norm_2(pli))**2
        self.hl_kfn = Function('hlk',[self.xl, self.xq],[self.hl_k],['xl0', 'xq0'],['hlkf'])
        # hard constraint
        # rli_s       = vertcat(self.rl*cos(self.ind_q*self.alpha),self.rl*sin(self.ind_q*self.alpha),0) # attachment point in {Bl}, symbolic expression
        # prli_s      = plk + Rlk@rli_s
        # pik_s       = self.xi[0:3,0]
        # pli_s       = pik_s - prli_s
        # self.hli_nbk = self.cl0 - norm_2(pli_s)
        # self.hli_nbk_fn = Function('hli_nbk',[self.xl, self.xi, self.ind_q],[self.hli_nbk],['xl0', 'xi0', 'i0'], ['hli_nbkf'])



    def matrix_trace(self,R): # trace of a ratation matrix
        tr = R[0,0] + R[1,1] + R[2,2]
        return tr
    
    
    def q_2_rotation(self, q): # from body frame to inertial frame
        # no normalization to avoid singularity in optimization
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3] # q0 denotes a scalar while q1, q2, and q3 represent rotational axes x, y, and z, respectively
        R = vertcat(
        horzcat( 2 * (q0 ** 2 + q1 ** 2) - 1, 2 * q1 * q2 - 2 * q0 * q3, 2 * q0 * q2 + 2 * q1 * q3),
        horzcat(2 * q0 * q3 + 2 * q1 * q2, 2 * (q0 ** 2 + q2 ** 2) - 1, 2 * q2 * q3 - 2 * q0 * q1),
        horzcat(2 * q1 * q3 - 2 * q0 * q2, 2 * q0 * q1 + 2 * q2 * q3, 2 * (q0 ** 2 + q3 ** 2) - 1)
        )
        return R
    
    def vee_map(self, v):
        vect = vertcat(v[2, 1], v[0, 2], v[1, 0])
        return vect
    
    
    def MPCsolverQuadrotorInit(self):
        assert hasattr(self, 'xi'), "Define the quadrotor's state variable first!"
        assert hasattr(self, 'ui'), "Define the quadrotor's contrl variable first!"
        self.SetConstraints_Qaudrotor()
        # start with an empty NLP
        w        = [] # optimal trajectory list
        self.w0i       = [] # initial guess list of optimal trajectory 
        self.lbwi      = [] # lower boundary list of optimal variables
        self.ubwi      = [] # upper boundary list of optimal variables
        g        = [] # equality and inequality constraint list
        self.lbgi      = [] # lower boundary list of constraints
        self.ubgi      = [] # upper boundary list of constraints
        
        # initial cost (different from MHE which has a non-zero arrival cost)
        J        = 0

        # hyperparameters + external signal (including feedback state and reference signals)
        Para     = SX.sym('Pi', (self.n_xi # current feedback
                                +self.n_xi*(self.N+1) # state reference trajectory within an MPC horizon
                                +self.n_ui*self.N # control reference trajectory within an MPC horizon
                                +self.n_pi # hyperparameters
                                +(self.nq-1)*2*(self.N+1) # all the quadrotor's x and y states, excluding the i-th quadrotor's state
                                +self.n_xl*(self.N+1) # the load state trajectory
                                +self.N # i-th cable tension trajectory
                                +1)) # index

        # initial state
        xk       = SX.sym('x0',self.n_xi)
        w       += [xk]
        # x0       = []
        xi_fb    = Para[0:self.n_xi]
        # for i in range(self.n_xi): # convert an array to a list
        #     x0    += [xi_fb[i]]
        
        self.w0i      += [0.5 * (x + y) for x,y in zip(self.xi_lb, self.xi_ub)]
        self.lbwi     += self.xi_lb
        self.ubwi     += self.xi_ub

        g            += [xk-xi_fb]
        self.lbgi     += self.n_xi*[0]
        self.ubgi     += self.n_xi*[0]

        # formulate the NLP
        Para_i   = Para[((self.N+2)*self.n_xi+self.n_ui*self.N):((self.N+2)*self.n_xi+self.n_ui*self.N+self.n_pi)]
        xqi   = SX.sym('Xqi',2,(self.nq-1))
        for k in range(self.N):
            # new control variable
            uk       = SX.sym('u_'+str(k),self.n_ui,1)
            w       += [uk]
            self.lbwi     += self.ui_lb
            self.ubwi     += self.ui_ub
            start_indx = self.n_xi+self.n_xi*(self.N+1)+self.n_ui*self.N+self.n_pi+(self.nq-1)*2*(self.N+1)+self.n_xl*(self.N+1)+k
            end_indx   = start_indx + 1
            ul_trajk   = Para[start_indx:end_indx]
            self.w0i      += [self.m*self.g + self.ml*self.g/self.n_ul,0,0,0]
            # self.w0i      += [0.5 * (x + y) for x,y in zip(self.ui_lb, self.ui_ub)]
            # integrate the cost function till the end of horizon
            Ref_xi   = Para[(k+1)*self.n_xi:(k+2)*self.n_xi]
            Ref_ui   = Para[((self.N+2)*self.n_xi+k*self.n_ui):((self.N+2)*self.n_xi+(k+1)*self.n_ui)]
            J       += self.Ji_kfn(xi0=xk,ui0=uk,refxi0=Ref_xi,refui0=Ref_ui,parai0=Para_i)['Ji_kf'] 
            start_indx = self.n_xi+self.n_xi*(self.N+1)+self.n_ui*self.N+self.n_pi+(self.nq-1)*2*(self.N+1)+self.n_xl*k
            end_indx   = start_indx + self.n_xl
            xl_trajk = Para[start_indx:end_indx]  
            index = Para[-1]
            if k>0:       
                # add a cable length equality constraint
                J       += self.hc_kfn(xi0=xk,xl0=xl_trajk,i0=index)['hc_kf']
                # add two inter-robot separation inequality constraints (considering only two neighbouring robots)
                for i in range(self.nq-1):
                    start_indx  = self.n_xi+self.n_xi*(self.N+1)+self.n_ui*self.N+self.n_pi+i*2*(self.N+1)+2*k
                    end_indx    = start_indx + 2
                    xi_opt      = Para[start_indx:end_indx]
                    xqi[:,i]    = xi_opt
                J       += self.gq_kfn(xi0=xk,xqi0=xqi)['gq_kf']

            # next state based on the discrete time model and current state
            xnext    = self.MDyni_fn(xi0=xk,ui0=uk,xl0=xl_trajk,ti0=ul_trajk,i0=index)['MDynif']
            # update the state variable
            xk       = SX.sym('x_'+str(k+1),self.n_xi,1)
            w       += [xk]
            self.lbwi     += self.xi_lb
            self.ubwi     += self.xi_ub
            self.w0i      += [0.5 * (x + y) for x,y in zip(self.xi_lb, self.xi_ub)]
            # add the equality constraint
            g       += [xnext - xk] # different from MHE which has a reversed order!
            self.lbgi     += self.n_xi*[0]
            self.ubgi     += self.n_xi*[0]
        
        # add the final cost (including the two kinds of soft constraints)
        Ref_xi   = Para[(self.N+1)*self.n_xi:(self.N+2)*self.n_xi]
        J       += self.Ji_Nfn(xi0=xk,refxi0=Ref_xi,parai0=Para_i)['Ji_Nf']
        start_indx = self.n_xi+self.n_xi*(self.N+1)+self.n_ui*self.N+self.n_pi+(self.nq-1)*2*(self.N+1) + self.n_xl*self.N
        end_indx   = start_indx + self.n_xl
        xl_trajk = Para[start_indx:end_indx]
        J       += self.hc_kfn(xi0=xk,xl0=xl_trajk,i0=index)['hc_kf']
        for i in range(self.nq-1):
            start_indx  = self.n_xi+self.n_xi*(self.N+1)+self.n_ui*self.N+self.n_pi+2*self.N+i*2*(self.N+1)
            end_indx    = start_indx + 2
            xi_opt      = Para[start_indx:end_indx]
            xqi[:,i]    = xi_opt
        J       += self.gq_kfn(xi0=xk,xqi0=xqi)['gq_kf']
        

        # create an NLP solver and solve it
        opts = {}
        opts['ipopt.tol'] = 1e-7
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
        opts['ipopt.warm_start_init_point']='yes'
        opts['ipopt.max_iter']=100
        opts['ipopt.acceptable_tol']=1e-7
        opts['ipopt.mu_strategy']='adaptive'
        # opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 
                'x': vertcat(*w),
                'p': Para, 
                'g': vertcat(*g)}
        self.solveri = nlpsol('solver', 'ipopt', prob, opts)
    

    def MPCsolverQuadrotor(self, parameters):
        # Solve the NLP
        sol = self.solveri(x0=self.w0i, 
                          lbx=self.lbwi, 
                          ubx=self.ubwi, 
                          p=parameters,
                          lbg=self.lbgi, 
                          ubg=self.ubgi)
        
        w_opt = sol['x'].full().flatten()
        lam_g = sol['lam_g'].full().flatten()
        # take the optimal control and state
        sol_traj = np.concatenate((w_opt, self.n_ui * [0]))
        sol_traj = np.reshape(sol_traj, (-1, self.n_xi + self.n_ui))
        state_traj_opt = sol_traj[:, 0:self.n_xi]
        control_traj_opt = np.delete(sol_traj[:, self.n_xi:], -1, 0)
    
        # take the optimal Lagrangian multiplier from lam_g
        costate_traj_ipopt = np.reshape(lam_g,(-1,self.n_xi))

        # output
        opt_sol = {"xi_opt":state_traj_opt,
                  "ui_opt":control_traj_opt,
                  "costate_ipopt":costate_traj_ipopt}
        
        return opt_sol
    
    
    def MPCsolverQuadrotorInit_acados(self,gazebo_sim=False):
        """
        This function is to define the optimal problem using acados
        """
        # predict horizon in seconds
        T = self.N * self.dt

        ##--------model dynamic symbolic expression------##
        Model_q = []
        OCP_q   = []
        self.OCP_q_solver = []
        for i in range(self.nq):
            Model_q += [AcadosModel()]
            ##-------build the ACADOS ocp-------##
            OCP_q += [AcadosOcp()]
            ##-------Set the environment path-------##
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            ## get the ACADOS path
            acados_source_path = os.environ['ACADOS_SOURCE_DIR']
            sys.path.insert(0,acados_source_path)
            OCP_q[i].acados_include_path = acados_source_path + '/include'
            OCP_q[i].acados_lib_path = acados_source_path + '/lib'

            ##----------mapping CasADi to ACADOS----------##
            Model_q[i].name="ACADOS_model_q_"+str(i)
        
            # parameters
            Pi = SX.sym('P_'+str(i),(self.n_xi # state reference at one step
                        +self.n_ui # control reference at one step
                        +self.n_pi # hyperparameters
                        +(self.nq-1)*2 # all the quadrotor's x and y states, excluding the i-th quadrotor's state
                        +self.n_xl # the load state at one step
                        +1 # i-th cable tension at one step
                        +1)) # index
            Model_q[i].p = Pi
        
            ###########################
            ##-------Optimizer-------##
            ###########################

            ##-------set the ocp model-------##
            # system state and control varialbes
            Model_q[i].x    = self.xi
            xi_dot      = SX.sym('x_dot_'+str(i),self.n_xi)
            Model_q[i].xdot = xi_dot
            Model_q[i].u    = self.ui

            OCP_q[i].model  = Model_q[i]
            OCP_q[i].dims.N = self.N # number of nodes
            OCP_q[i].solver_options.tf = T # horizon length T (unit: second)
            OCP_q[i].dims.np = self.n_xi + self.n_ui + self.n_pi + (self.nq-1)*2 + self.n_xl + 1 + 1 # number of parameters for the solver
            OCP_q[i].parameter_values = np.zeros(self.n_xi + self.n_ui + self.n_pi + (self.nq-1)*2 + self.n_xl + 1 + 1) 

            Ref_xi = OCP_q[i].model.p[0:self.n_xi]
            Ref_ui = OCP_q[i].model.p[self.n_xi:(self.n_xi+self.n_ui)]
            Para_i = OCP_q[i].model.p[(self.n_xi+self.n_ui):(self.n_xi+self.n_ui+self.n_pi)]
            xq_i   = OCP_q[i].model.p[(self.n_xi+self.n_ui+self.n_pi):(self.n_xi+self.n_ui+self.n_pi+(self.nq-1)*2)]
            xl     = OCP_q[i].model.p[(self.n_xi+self.n_ui+self.n_pi+(self.nq-1)*2):(self.n_xi+self.n_ui+self.n_pi+(self.nq-1)*2+self.n_xl)]
            ti     = OCP_q[i].model.p[(self.n_xi+self.n_ui+self.n_pi+(self.nq-1)*2+self.n_xl):(self.n_xi+self.n_ui+self.n_pi+(self.nq-1)*2+self.n_xl+1)]
            indexi = OCP_q[i].model.p[(self.n_xi+self.n_ui+self.n_pi+(self.nq-1)*2+self.n_xl+1):(self.n_xi+self.n_ui+self.n_pi+(self.nq-1)*2+self.n_xl+2)]

            # explicit model
            Model_q[i].f_expl_expr = self.f_i(xi0=OCP_q[i].model.x,ui0=OCP_q[i].model.u,xl0=xl,ti0=ti,i0=indexi)['fif']

            # implicit model
        
            Model_q[i].f_impl_expr = OCP_q[i].model.xdot - self.f_i(xi0=OCP_q[i].model.x,ui0=OCP_q[i].model.u,xl0=xl,ti0=ti,i0=indexi)['fif']
        
            OCP_q[i].model  = Model_q[i]
        
            #######################################
            ##-------set the cost function-------##
            #######################################

            #------------external cost------------#
            OCP_q[i].cost.cost_type   = 'EXTERNAL'
            OCP_q[i].cost.cost_type_e = 'EXTERNAL'
            Xq_i = SX.sym('Xq_'+str(i),2,(self.nq-1))
            for k in range(self.nq-1):
                xq_ik = xq_i[k*2:(k+1)*2]
                Xq_i[:,k] = xq_ik
        
            OCP_q[i].model.cost_expr_ext_cost = self.Ji_kfn(xi0=OCP_q[i].model.x,ui0=OCP_q[i].model.u,refxi0=Ref_xi,refui0=Ref_ui,parai0=Para_i)['Ji_kf'] \
                + self.hc_kfn(xi0=OCP_q[i].model.x,xl0=xl,i0=indexi)['hc_kf'] + self.gq_kfn(xi0=OCP_q[i].model.x,xqi0=Xq_i)['gq_kf']
            OCP_q[i].model.cost_expr_ext_cost_e = self.Ji_Nfn(xi0=OCP_q[i].model.x,refxi0=Ref_xi,parai0=Para_i)['Ji_Nf'] \
                + self.hc_kfn(xi0=OCP_q[i].model.x,xl0=xl,i0=indexi)['hc_kf'] + self.gq_kfn(xi0=OCP_q[i].model.x,xqi0=Xq_i)['gq_kf']
        
            #######################################
            ##-------set the constraints---------##
            #######################################

            ##-------state constraints-----------##
            x_init   = np.zeros(self.n_xi)
            x_init[6]= 1
            OCP_q[i].constraints.x0 = x_init

            OCP_q[i].constraints.lbx = np.array(self.xi_lb)
            OCP_q[i].constraints.ubx = np.array(self.xi_ub)
            OCP_q[i].constraints.idxbx = np.array([i for i in range(self.n_xi)])

            ##-------control constraints---------##
            OCP_q[i].constraints.lbu = np.array(self.ui_lb)
            OCP_q[i].constraints.ubu = np.array(self.ui_ub)
            OCP_q[i].constraints.idxbu = np.array([i for i in range(self.n_ui)])

            ##-------set the solver--------##
            OCP_q[i].solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
            OCP_q[i].solver_options.hessian_approx = 'GAUSS_NEWTON'
            OCP_q[i].solver_options.regularize_method = 'CONVEXIFY'
            OCP_q[i].solver_options.integrator_type = 'ERK'
            OCP_q[i].solver_options.sim_method_num_stages = 4 # default 4, meaning 4-th order Runge Kutta
            OCP_q[i].solver_options.print_level = 0
            OCP_q[i].solver_options.levenberg_marquardt = 1e-10 # small value for gauss newton method, large value for gradient descent method
            OCP_q[i].solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI or SQP
            # ocp.solver_options.nlp_solver_max_iter = 100

            ##-------set the code generation--------##
            # compile acados ocp
            json_file_i = os.path.join('./'+Model_q[i].name+'_acados_ocp.json')
            # load solver from json file
            build_i = True
            generate_i = True
            if gazebo_sim:
                build_i=False
                generate_i=False
            self.OCP_q_solver += [AcadosOcpSolver(OCP_q[i],generate=generate_i,build=build_i,json_file=json_file_i)]
        
        ##--------compute Lagrangian multipliers from KKT conditions---------##
        cik_aug  = self.Ji_k + self.hc_k + self.gq_k
        ciN_aug  = self.Ji_N + self.hc_k + self.gq_k
        dcik_aug = jacobian(cik_aug, self.xi)
        dciN_aug = jacobian(ciN_aug, self.xi)
        dModeli  = jacobian(self.Modeli,self.xi)
        self.dJk_fn  = Function('dJk',[self.xi, self.ui, self.xl, self.x_qi, self.ref_xi, self.ref_ui, self.para_i, self.ind_q],[dcik_aug],['xi0', 'ui0', 'xl0', 'xq0', 'refxi0', 'refui0', 'parai0','i0'],['dJkf'])
        self.dJN_fn   = Function('dJN',[self.xi, self.xl, self.x_qi, self.ref_xi, self.para_i, self.ind_q],[dciN_aug],['xi0', 'xl0', 'xq0', 'refxi0', 'parai0','i0'],['dJNf'])
        self.F_fn    = Function('F',[self.xi, self.ui, self.xl, self.ti, self.ind_q],[dModeli],['xi0', 'ui0', 'xl0', 'ti0', 'i0'],['Ff'])
        
    
    def MPCsolverQuadrotor_acados(self, xi_fb, xqi_traj, xl_traj, ul_traj, Ref_xi, Ref_ui, Para_i, index):
        """
        This function solves the optimal control problem (ocp) using ACADOS
        """
        # set the solver parameters at 0->N-1 nodes
        for k in range(self.N):
            ref_xi = Ref_xi[k*self.n_xi:(k+1)*self.n_xi]
            ref_ui = Ref_ui[k*self.n_ui:(k+1)*self.n_ui]
            xqj    = np.zeros(2*(self.nq-1))
            for j in range(self.nq-1):
                xj = xqi_traj[(j*2*(self.N+1)+2*k):(j*2*(self.N+1)+2*(k+1))]
                xqj[2*j:2*(j+1)] = xj
            xli    = xl_traj[k*self.n_xl:(k+1)*self.n_xl]
            uli    = np.reshape(ul_traj[k],1)
            indx   = np.reshape(index,1)
            self.OCP_q_solver[index].set(k,'p',np.concatenate((ref_xi,ref_ui,Para_i,xqj,xli,uli,indx)))

        # set the terminal cost
        ref_xN = Ref_xi[self.N*self.n_xi:(self.N+1)*self.n_xi]
        xqjN   = np.zeros(2*(self.nq-1))
        for j in range(self.nq-1):
            xj = xqi_traj[(j*2*(self.N+1)+2*self.N):(j*2*(self.N+1)+2*(self.N+1))]
            xqjN[2*j:2*(j+1)]=xj
        xlN    = xl_traj[self.N*self.n_xl:(self.N+1)*self.n_xl]
        self.OCP_q_solver[index].set(self.N,'p',np.concatenate((ref_xN,ref_ui,Para_i,xqjN,xlN,uli,indx)))

        # set the initial condition to be aligned with the current feedback
        self.OCP_q_solver[index].set(0,'lbx',np.array(xi_fb))
        self.OCP_q_solver[index].set(0,'ubx',np.array(xi_fb))

        # self.OCP_q_solver[index].set(0,'x',np.array(xi_fb))

        NO_SOLUTION_FLAG = False

        # solve ocp
        status = self.OCP_q_solver[index].solve()

        if status!=0:
            NO_SOLUTION_FLAG = True

        ##--------take the optimal control and state sequences-------##
        statei_traj_opt   = np.zeros((self.N+1,self.n_xi))
        controli_traj_opt = np.zeros((self.N,self.n_ui))
        for k in range(self.N):
            statei_traj_opt[k:k+1,:] = np.reshape(self.OCP_q_solver[index].get(k,'x'),(1,self.n_xi))
            controli_traj_opt[k:k+1,:]=np.reshape(self.OCP_q_solver[index].get(k,'u'),(1,self.n_ui))
        statei_traj_opt[self.N:self.N+1,:] = np.reshape(self.OCP_q_solver[index].get(self.N,'x'),(1,self.n_xi))

        ##--------compute the Lagrangian multiplier trajectory-------##
        costatei_traj_opt = numpy.zeros((self.N, self.n_xi))
        costatei_traj_opt[self.N-1:self.N,:] = np.reshape(self.dJN_fn(xi0=statei_traj_opt[-1,:],xl0=xlN,xq0=np.transpose(np.reshape(xqjN,(self.nq-1,2))),refxi0=ref_xN,parai0=Para_i,i0=index)['dJNf'].full(),(1,self.n_xi))
        for k in range(self.N-1,0,-1):
            xi_curr   = statei_traj_opt[k,:]
            ui_curr   = controli_traj_opt[k,:]
            ref_xi = Ref_xi[k*self.n_xi:(k+1)*self.n_xi]
            ref_ui = Ref_ui[k*self.n_ui:(k+1)*self.n_ui]
            lambda_c  = np.reshape(costatei_traj_opt[k,:],(self.n_xi,1))
            xqj       = np.zeros(2*(self.nq-1))
            for j in range(self.nq-1):
                xj = xqi_traj[(j*2*(self.N+1)+2*k):(j*2*(self.N+1)+2*(k+1))]
                xqj[2*j:2*(j+1)] = xj
            xli    = xl_traj[k*self.n_xl:(k+1)*self.n_xl]
            uli    = np.reshape(ul_traj[k],1)
            dcdx_k    = np.reshape(self.dJk_fn(xi0=xi_curr, ui0=ui_curr, xl0=xli, xq0=np.transpose(np.reshape(xqj,(self.nq-1,2))), refxi0=ref_xi, refui0=ref_ui, parai0=Para_i,i0=index)['dJkf'].full(),(self.n_xi,1))
            dfdx_k    = self.F_fn(xi0=xi_curr, ui0=ui_curr, xl0=xli, ti0=uli, i0=index)['Ff'].full()
            lambda_pre= dcdx_k + dfdx_k.T@lambda_c
            costatei_traj_opt[(k-1):k,:] = lambda_pre.T
        

        # output
        opt_soli = {"xi_opt": statei_traj_opt,
                    "ui_opt": controli_traj_opt,
                    "costatei_opt": costatei_traj_opt}
        
        return opt_soli
       
    
    def MPCsolverPayloadInit(self):
        assert hasattr(self, 'xl'), "Define the payload's state variable first!"
        assert hasattr(self, 'ul'), "Define the payload's contrl variable first!"
        # start with an empty NLP
        w        = [] # optimal trajectory list
        self.w0       = [] # initial guess list of optimal trajectory 
        self.lbw      = [] # lower boundary list of optimal variables
        self.ubw      = [] # upper boundary list of optimal variables
        g        = [] # equality and inequality constraint list
        self.lbg      = [] # lower boundary list of constraints
        self.ubg      = [] # upper boundary list of constraints
        
        # initial cost (different from MHE which has a non-zero arrival cost)
        J        = 0
        
        # hyperparameters + external signals
        Paral     = SX.sym('Pl', (self.n_xl # current feedback state        
                                +self.n_xl*(self.N+1) # state reference trajectory
                                +self.n_ul*self.N # control reference trajectory
                                +self.n_pl # hyperparameters
                                +self.nq*self.n_xi*(self.N+1) # all the quadrotors' states
                                +self.n_lp)) # load inertial parameters
        
        # initial state
        xk       = SX.sym('x0',self.n_xl)
        w       += [xk]
        # x0       = []
        xl_fb    = Paral[0:self.n_xl]
        # for i in range(self.n_xl): # convert an array to a list
        #     x0    += [xl_fb[i]]
        self.w0      += [0.5 * (x + y) for x,y in zip(self.xl_lb, self.xl_ub)]
        self.lbw     += self.xl_lb
        self.ubw     += self.xl_ub

        g            += [xk-xl_fb]
        self.lbg     += self.n_xl*[0]
        self.ubg     += self.n_xl*[0]
        
        # formulate the NLP
        Para_l   = Paral[((self.N+2)*self.n_xl+self.N*self.n_ul):((self.N+2)*self.n_xl+self.N*self.n_ul+self.n_pl)]
        start_index = (self.N+2)*self.n_xl+self.n_ul*self.N+self.n_pl+self.nq*self.n_xi*(self.N+1)
        end_index   = start_index+3
        Jl       = Paral[start_index:end_index]
        start_index = (self.N+2)*self.n_xl+self.n_ul*self.N+self.n_pl+self.nq*self.n_xi*(self.N+1)+3
        end_index   = start_index+3
        rg       = Paral[start_index:end_index]
        xq       = SX.sym('xq',self.n_xi,self.nq)
        for k in range(self.N):
            # new control variable
            uk       = SX.sym('u_'+str(k),self.n_ul,1)
            w       += [uk]
            self.lbw     += self.ul_lb
            self.ubw     += self.ul_ub
            self.w0      += self.n_ul*[self.ml*self.g/self.n_ul] # uniform distribution of the payload's gravity force over these cables
            # integrate the cost function till the end of horizon
            Ref_xl   = Paral[(k+1)*self.n_xl:(k+2)*self.n_xl]
            Ref_ul   = Paral[((self.N+2)*self.n_xl+k*self.n_ul):((self.N+2)*self.n_xl+(k+1)*self.n_ul)]
            J       += self.Jl_kfn(xl0=xk, ul0=uk, refxl0=Ref_xl, reful0=Ref_ul, paral0=Para_l)['Jl_kf']
            for i in range(self.nq):
                start_index = self.n_xl+self.n_xl*(self.N+1)+self.n_ul*self.N+self.n_pl+i*self.n_xi*(self.N+1)+self.n_xi*k
                end_index   = start_index + self.n_xi
                xi_opt      = Paral[start_index:end_index]
                xq[:,i]     = xi_opt
            if k > 0:
                J       += self.hl_kfn(xl0=xk, xq0=xq)['hlkf']
            # next state based on the discrete time model and current state
            xnext    = self.MDynl_fn(xl0=xk,ul0=uk,xq0=xq,Jl0=Jl,rg0=rg)['MDynlf']
            # update the state variable
            xk       = SX.sym('x_'+str(k+1),self.n_xl,1)
            w       += [xk] 
            self.lbw     += self.xl_lb
            self.ubw     += self.xl_ub
            self.w0      += [0.5 * (x + y) for x,y in zip(self.xl_lb, self.xl_ub)]
            # add equality constraint
            g       += [xnext - xk] # different from MHE which has a reversed order!
            self.lbg     += self.n_xl*[0]
            self.ubg     += self.n_xl*[0]
        
        # add the final cost
        Ref_xl = Paral[(self.N+1)*self.n_xl:(self.N+2)*self.n_xl]
        J     += self.Jl_Nfn(xl0=xk, refxl0=Ref_xl, paral0=Para_l)['Jl_Nf']
        for i in range(self.nq):
            start_index = self.n_xl+self.n_xl*(self.N+1)+self.n_ul*self.N+self.n_pl+i*self.n_xi*(self.N+1)+self.n_xi*self.N
            end_index   = start_index+self.n_xi
            xi_opt      = Paral[start_index:end_index]
            xq[:,i]     = xi_opt
        J     += self.hl_kfn(xl0=xk, xq0=xq)['hlkf']

        # create an NLP solver and solve it
        opts = {}
        opts['ipopt.tol'] = 1e-7
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
        opts['ipopt.warm_start_init_point']='yes'
        opts['ipopt.max_iter']=100
        opts['ipopt.acceptable_tol']=1e-7
        opts['ipopt.mu_strategy']='adaptive'
        # opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 
                'x': vertcat(*w), 
                'p': Paral,
                'g': vertcat(*g)}
        self.solver = nlpsol('solver', 'ipopt', prob, opts)


    def MPCsolverPayload(self,parametersl): 
        # Solve the NLP
        sol = self.solver(x0=self.w0, 
                          lbx=self.lbw, 
                          ubx=self.ubw, 
                          p=parametersl,
                          lbg=self.lbg, 
                          ubg=self.ubg)
        
        w_opt = sol['x'].full().flatten()
        # take the optimal control and state
        sol_traj = np.concatenate((w_opt, self.n_ul * [0]))
        sol_traj = np.reshape(sol_traj, (-1, self.n_xl + self.n_ul))
        state_traj_opt = sol_traj[:, 0:self.n_xl]
        control_traj_opt = np.delete(sol_traj[:, self.n_xl:], -1, 0)

        # compute the costate trajectory using two methods
        lam_g    = sol['lam_g'].full().flatten()
        costate_traj_ipopt = np.reshape(lam_g,(-1,self.n_xl))
        
        # output
        opt_sol = {"xl_opt":state_traj_opt,
                  "ul_opt":control_traj_opt,
                  "costatel_ipopt":costate_traj_ipopt}
        
        return opt_sol
    
    def MPCsolverPayloadInit_acados(self,gazebo_sim=False):
        """
        This function is to define the optimal problem using acados
        """
        # predict horizon in seconds
        T = self.N * self.dt

        ##-------model dynamic symbolic expression------##
        modell = AcadosModel()

        ##-------build the ACADOS ocp-------##
        ocpl = AcadosOcp()
        ##-------Set the environment path-------##
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        ## get the ACADOS path
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0,acados_source_path)
        ocpl.acados_include_path = acados_source_path + '/include'
        ocpl.acados_lib_path = acados_source_path + '/lib'

        ##----------mapping CasADi to ACADOS----------##
        modell.name="ACADOS_model_l"
        
        # parameters
        Pl = SX.sym('Pl',(self.n_xl # state reference at one step
                        +self.n_ul # control reference at one step
                        +self.n_pl # hyperparameters
                        +self.nq*self.n_xi # all the quadrotors' states
                        +self.n_lp)) # load inertial parameters
        modell.p = Pl

        ###########################
        ##-------Optimizer-------##
        ###########################

        ##-------set the ocp model-------##
        ocpl.dims.N = self.N # number of nodes
        ocpl.solver_options.tf = T # horizon length T (unit: second)
        ocpl.dims.np = self.n_xl + self.n_ul + self.n_pl + self.nq*self.n_xi + self.n_lp
        ocpl.parameter_values = np.zeros(self.n_xl + self.n_ul + self.n_pl + self.nq*self.n_xi + self.n_lp)
        
        # system state and control varialbes
        modell.x    = self.xl
        xl_dot      = SX.sym('xl_dot',self.n_xl)
        modell.xdot = xl_dot
        modell.u    = self.ul
        ocpl.model  = modell
        Ref_xl = ocpl.model.p[0:self.n_xl]
        Ref_ul = ocpl.model.p[self.n_xl:(self.n_xl+self.n_ul)]
        Para_l = ocpl.model.p[(self.n_xl+self.n_ul):(self.n_xl+self.n_ul+self.n_pl)]
        xq     = ocpl.model.p[(self.n_xl+self.n_ul+self.n_pl):(self.n_xl+self.n_ul+self.n_pl+self.nq*self.n_xi)]
        Jl     = ocpl.model.p[(self.n_xl+self.n_ul+self.n_pl+self.nq*self.n_xi):(self.n_xl+self.n_ul+self.n_pl+self.nq*self.n_xi+3)]
        rg     = ocpl.model.p[(self.n_xl+self.n_ul+self.n_pl+self.nq*self.n_xi+3):(self.n_xl+self.n_ul+self.n_pl+self.nq*self.n_xi+self.n_lp)]
       

        # explicit model
        Xq     = SX.sym('Xq',self.n_xi,self.nq)
        for k in range(self.nq):
            xqi = xq[k*self.n_xi:(k+1)*self.n_xi]
            Xq[:,k]=xqi
        modell.f_expl_expr = self.f_l(xl0=ocpl.model.x,ul0=ocpl.model.u,xq0=Xq,Jl0=Jl,rg0=rg)['flf']

        # implicit model
        
        modell.f_impl_expr = ocpl.model.xdot - self.f_l(xl0=ocpl.model.x,ul0=ocpl.model.u,xq0=Xq,Jl0=Jl,rg0=rg)['flf']
        
        ocpl.model  = modell
        
        #######################################
        ##-------set the cost function-------##
        #######################################

        #------------external cost------------#
        ocpl.cost.cost_type   = 'EXTERNAL'
        ocpl.cost.cost_type_e = 'EXTERNAL'

        ocpl.model.cost_expr_ext_cost = self.Jl_kfn(xl0=ocpl.model.x, ul0=ocpl.model.u, refxl0=Ref_xl, reful0=Ref_ul, paral0=Para_l)['Jl_kf']\
              + self.hl_kfn(xl0=ocpl.model.x, xq0=Xq)['hlkf']
        ocpl.model.cost_expr_ext_cost_e = self.Jl_Nfn(xl0=ocpl.model.x, refxl0=Ref_xl, paral0=Para_l)['Jl_Nf']\
              + self.hl_kfn(xl0=ocpl.model.x, xq0=Xq)['hlkf']

        #######################################
        ##-------set the constraints---------##
        #######################################

        ##-------state constraints-----------##
        x_init   = np.zeros(self.n_xl)
        x_init[6]= 1
        ocpl.constraints.x0 = x_init

        ocpl.constraints.lbx = np.array(self.xl_lb)
        ocpl.constraints.ubx = np.array(self.xl_ub)
        ocpl.constraints.idxbx = np.array([i for i in range(self.n_xl)])

        ##-------control constraints---------##
        ocpl.constraints.lbu = np.array(self.ul_lb)
        ocpl.constraints.ubu = np.array(self.ul_ub)
        ocpl.constraints.idxbu = np.array([i for i in range(self.n_ul)])

        ##-------set the solver--------##
        ocpl.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocpl.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocpl.solver_options.regularize_method = 'CONVEXIFY'
        ocpl.solver_options.integrator_type = 'ERK'
        ocpl.solver_options.sim_method_num_stages = 4 # default 4
        ocpl.solver_options.print_level = 0
        ocpl.solver_options.levenberg_marquardt = 1e-10 # small value for gauss newton method, large value for gradient descent method
        ocpl.solver_options.nlp_solver_type ='SQP_RTI' # SQP_RTI or SQP
        # ocpl.solver_options.nlp_solver_max_iter = 100

        ##-------set the code generation--------##
        # compile acados ocp
        json_file_l = os.path.join('./'+modell.name+'_acados_ocp.json')
        
        # load solver from json file
        build_l = True
        generate_l = True
        if gazebo_sim:
            build_i=False
            generate_i=False
        self.acados_solver_ql = AcadosOcpSolver(ocpl,generate=generate_l,build=build_l,json_file=json_file_l)

        ##--------compute Lagrangian multipliers from KKT conditions---------##
        clk_aug  = self.Jl_k + self.hl_k
        clN_aug  = self.Jl_N + self.hl_k
        dclk_aug = jacobian(clk_aug,self.xl)
        dclN_aug = jacobian(clN_aug,self.xl)
        dModell  = jacobian(self.Modell,self.xl)
        self.dJlk_fn  = Function('dJlk',[self.xl, self.ul, self.xq, self.ref_xl, self.ref_ul, self.para_l], [dclk_aug], ['xl0', 'ul0', 'xq0', 'refxl0', 'reful0', 'paral0'], ['dJlkf'])
        self.dJlN_fn  = Function('dJlN',[self.xl, self.xq, self.ref_xl, self.para_l],[dclN_aug],['xl0', 'xq0', 'refxl0', 'paral0'],['dJlNf'])
        self.Fcl_fn   = Function('Fl',[self.xl, self.ul, self.xq, self.Jldiag, self.rg], [dModell], ['xl0', 'ul0', 'xq0', 'Jl0', 'rg0'], ['Flf'])



    def MPCsolverPayload_acados(self, xl_fb, xq_traj, Ref_xl, Ref_ul, Para_l, Jl, rg):
        """
        This function solves the optimal control problem (ocp) using ACADOS
        """
        
        # set the solver parameters at 0->N-1 nodes
        for k in range(self.N):
            ref_xl = Ref_xl[k*self.n_xl:(k+1)*self.n_xl]
            ref_ul = Ref_ul[k*self.n_ul:(k+1)*self.n_ul]
            xqi    = np.zeros(self.n_xi*self.nq)
            for i in range(self.nq):
                xi = xq_traj[(i*self.n_xi*(self.N+1)+self.n_xi*k):(i*self.n_xi*(self.N+1)+self.n_xi*(k+1))]
                xqi[self.n_xi*i:self.n_xi*(i+1)] = xi
            self.acados_solver_ql.set(k,'p',np.concatenate((ref_xl,ref_ul,Para_l,xqi,Jl,rg)))
        
        # set the terminal cost
        ref_xlN = Ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl]
        xqiN   = np.zeros(self.n_xi*self.nq)
        for i in range(self.nq):
            xiN = xq_traj[(i*self.n_xi*(self.N+1)+self.n_xi*self.N):(i*self.n_xi*(self.N+1)+self.n_xi*(self.N+1))]
            xqiN[self.n_xi*i:self.n_xi*(i+1)]=xiN
        self.acados_solver_ql.set(self.N,'p',np.concatenate((ref_xlN,ref_ul,Para_l,xqiN,Jl,rg)))

        # set the initial condition to be aligned with the current feedback
        self.acados_solver_ql.set(0,'lbx',np.array(xl_fb))
        self.acados_solver_ql.set(0,'ubx',np.array(xl_fb))

        # self.acados_solver_ql.set(0,'x',np.array(xl_fb))

        NO_SOLUTION_FLAG = False

        # solve ocp
        status = self.acados_solver_ql.solve()

        if status!=0:
            NO_SOLUTION_FLAG = True
        
        ##--------take the optimal control and state sequences-------##
        statel_traj_opt   = np.zeros((self.N+1,self.n_xl))
        controll_traj_opt = np.zeros((self.N,self.n_ul))
        for k in range(self.N):

            statel_traj_opt[k:k+1,:] = np.reshape(self.acados_solver_ql.get(k,'x'),(1,self.n_xl))
            controll_traj_opt[k:k+1,:]=np.reshape(self.acados_solver_ql.get(k,'u'),(1,self.n_ul))
        statel_traj_opt[self.N:self.N+1,:] = np.reshape(self.acados_solver_ql.get(self.N,'x'),(1,self.n_xl))

        ##--------compute Lagrangian multipliers from KKT conditions---------##
        costatel_traj_opt = numpy.zeros((self.N, self.n_xl))
        costatel_traj_opt[self.N-1:self.N,:] = np.reshape(self.dJlN_fn(xl0=statel_traj_opt[-1,:], xq0=np.transpose(np.reshape(xqiN,(self.nq,self.n_xi))), refxl0=ref_xlN, paral0=Para_l)['dJlNf'].full(),(1,self.n_xl))
        for k in range(self.N-1,0,-1):
            xl_curr   = statel_traj_opt[k,:]
            ul_curr   = controll_traj_opt[k,:]
            lambda_c  = np.reshape(costatel_traj_opt[k,:],(self.n_xl,1))
            ref_xl = Ref_xl[k*self.n_xl:(k+1)*self.n_xl]
            ref_ul = Ref_ul[k*self.n_ul:(k+1)*self.n_ul]
            xqi    = np.zeros(self.n_xi*self.nq)
            for i in range(self.nq):
                xi = xq_traj[(i*self.n_xi*(self.N+1)+self.n_xi*k):(i*self.n_xi*(self.N+1)+self.n_xi*(k+1))]
                xqi[self.n_xi*i:self.n_xi*(i+1)] = xi
            dcdx_k    = np.reshape(self.dJlk_fn(xl0=xl_curr, ul0=ul_curr, xq0=np.transpose(np.reshape(xqi,(self.nq,self.n_xi))), refxl0=ref_xl, reful0=ref_ul, paral0=Para_l)['dJlkf'].full(),(self.n_xl,1))
            dfdx_k    = self.Fcl_fn(xl0=xl_curr, ul0=ul_curr, xq0=np.transpose(np.reshape(xqi,(self.nq,self.n_xi))), Jl0=Jl,rg0=rg)['Flf'].full()
            lambda_pre= dcdx_k + dfdx_k.T@lambda_c
            costatel_traj_opt[(k-1):k,:] = lambda_pre.T

        # output
        opt_soll = {"xl_opt": statel_traj_opt,
                    "ul_opt": controll_traj_opt,
                    "costatel_opt": costatel_traj_opt}
        
        return opt_soll

    
    def Distributed_forwardMPC(self, xq_fb, xl_fb, xq_traj_prev, uq_traj_prev, xl_traj_prev, ul_traj_prev, Ref_xq, Ref_uq, Ref_xl, Ref_ul, Para_q, Para_l, Jl, rg):
        epsilon = 1e-2 # threshold for stopping the iteration
        k_max   = 5 # maximum number of iterations
        max_violation = 5 # initial value of max_violation, defined as the maximum value of the differences between two trajectories in successive iterations for all quadrotors
        ke       = 1
        xq_traj = []
        uq_traj = []
        for iq in range(self.nq):
            xiq_traj = np.zeros((self.N+1,self.n_xi))
            uiq_traj = np.zeros((self.N,self.n_ui))
            xi_prev  = xq_traj_prev[iq]
            ui_prev  = uq_traj_prev[iq]
            for iqk in range(self.N):
                xiq_traj[iqk,:] = xi_prev[iqk+1,:] # note that we have moved forward by one time-step, so we take elements from [1,:]
                if iqk <self.N-1:
                    uiq_traj[iqk,:] = ui_prev[iqk+1,:]
                else:
                    uiq_traj[-1,:] = ui_prev[-1,:]
            xiq_traj[-1,:] = xi_prev[-1,:]
            xq_traj += [xiq_traj]
            uq_traj += [uiq_traj]
        xl_traj = np.zeros((self.N+1,self.n_xl))
        ul_traj = np.zeros((self.N,self.n_ul))
        for il in range(self.N):
            xl_traj[il,:] = xl_traj_prev[il+1,:]
            if il <self.N-1:
                ul_traj[il,:] = ul_traj_prev[il+1,:]
            else:
                ul_traj[-1,:] = ul_traj_prev[-1,:]
        xl_traj[-1,:] = xl_traj_prev[-1,:]
        
        while max_violation>=epsilon and ke<=k_max:
            viol_x_list = []
            viol_u_list = []
            xq_temp     = [] # temporary list for saving the updated state trajectories during the 'for' loop
            uq_temp     = [] # temporary list for saving the updated control trajectories during the 'for' loop
            cx_quad     = []
            
            for i in range(self.nq): # later, we will upgrade these iterations to parallel computing
                xi_opt      = np.zeros((self.N+1,self.n_xi))
                ui_opt      = np.zeros((self.N,self.n_ui))
                xi_traj     = xq_traj[i] # this trajectory list should be updated after each iteration
                ui_traj     = uq_traj[i] # this trajectory list should also be updated after each iteration
                xi_fb       = np.reshape(xq_fb[i],self.n_xi)
                Ref_xi      = Ref_xq[i]
                Ref_ui      = Ref_uq[i]
                ref_xi      = np.zeros(self.n_xi*(self.N+1))
                ref_ui      = np.zeros(self.n_ui*self.N)
                xl_trajh    = np.zeros(self.n_xl*(self.N+1))
                for k in range(self.N):
                    ref_xik = np.reshape(Ref_xi[:,k],self.n_xi)
                    ref_xi[k*self.n_xi:(k+1)*self.n_xi]=ref_xik
                    ref_uik = np.reshape(Ref_ui[:,k],self.n_ui)
                    ref_ui[k*self.n_ui:(k+1)*self.n_ui]=ref_uik
                    xl_k    = np.reshape(xl_traj[k,:],self.n_xl)
                    xl_trajh[k*self.n_xl:(k+1)*self.n_xl]=xl_k
                ref_xi[self.N*self.n_xi:(self.N+1)*self.n_xi]=np.reshape(Ref_xi[:,self.N],self.n_xi)
                xl_trajh[self.N*self.n_xl:(self.N+1)*self.n_xl]=np.reshape(xl_traj[self.N,:],self.n_xl)
                Para_i      = np.reshape(Para_q[i],self.n_pi)
                xq_i        = np.zeros((self.nq-1)*2*(self.N+1))
                # del xq_traj_i[i] # delete the i-th quadrotor's state from xq_traj
                kj = 0
                for j in range(self.nq):
                    if j!=i:
                        xqj     = xq_traj[j]
                        xqj_xy  = np.reshape(xqj[:,0:2],2*(self.N+1))
                        xq_i[kj*2*(self.N+1):(kj+1)*2*(self.N+1)] = xqj_xy
                        kj += 1
                uli_traj    = np.reshape(ul_traj[:,i],self.N)
                opt_sol_i   = self.MPCsolverQuadrotor_acados(xi_fb, xq_i, xl_trajh, uli_traj, ref_xi, ref_ui, Para_i, i)
                xi_opt      = np.array(opt_sol_i['xi_opt'])
                ui_opt      = np.array(opt_sol_i['ui_opt'])
                cox_opt_i   = opt_sol_i['costatei_opt']
                # cox_ipopt_i = opt_sol_i['costate_ipopt']
                sum_viol_xi = 0
                sum_viol_ui = 0
                sum_viol_cxi= 0
                for ki in range(len(ui_traj)):
                    sum_viol_xi  += LA.norm(xi_opt[ki,:]-xi_traj[ki,:])
                    sum_viol_ui  += LA.norm(ui_opt[ki,:]-ui_traj[ki,:])
                    # sum_viol_cxi += LA.norm(cox_opt_i[ki,:]-cox_ipopt_i[ki,:])
                sum_viol_xi  += LA.norm(xi_opt[-1,:]-xi_traj[-1,:])
                viol_xi  = sum_viol_xi/len(xi_opt)
                viol_ui  = sum_viol_ui/len(ui_opt)
                # viol_cxi = sum_viol_cxi/len(cox_opt_i)
                viol_x_list += [viol_xi]
                viol_u_list += [viol_ui]
                initial_error = LA.norm(np.reshape(xi_opt[0,:],(self.n_xi,1))-xi_fb)
                print('iteration=',ke,'quadrotor_ID=',i,'viol_xi=',format(viol_xi,'.5f'),'viol_ui=',format(viol_ui,'.5f'))
                # print('iteration=',k,'quadrotor_ID=',i,'x0_error=',initial_error)
                # save trajectories to the temporary lists
                xq_temp  += [xi_opt]
                uq_temp  += [ui_opt]
                # xq_temp.append(xi_opt)
                # uq_temp.append(ui_opt)
                # xq_traj[i] = xi_opt
                # uq_traj[i] = ui_opt
                # save costate trajectories
                cx_quad += [cox_opt_i]
                
            # update the quadrotors' trajectories
            xq_traj  = xq_temp
            uq_traj  = uq_temp

            #----------The above code will be in parallel, followed by the following code for computing the payload's trajectory--------#
            #----------In other words, the computations of the quadrotors' MPC and the payload's MPC is sequential----------#
            #----------The reason is that the payload does not have the compuation capability and its MPC is solved by the leader agent--------#
            
            # solve the MPC of the payload using the updated quadrotor trajectories xq_traj
            xl_fbh      = np.reshape(xl_fb,self.n_xl)
            ref_xl      = np.zeros(self.n_xl*(self.N+1))
            ref_ul      = np.zeros(self.n_ul*self.N)
            for k in range(self.N):
                ref_xlk = np.reshape(Ref_xl[:,k],self.n_xl)
                ref_xl[k*self.n_xl:(k+1)*self.n_xl] = ref_xlk
                ref_ulk = np.reshape(Ref_ul[:,k],self.n_ul)
                ref_ul[k*self.n_ul:(k+1)*self.n_ul] = ref_ulk
            ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl]=np.reshape(Ref_xl[:,self.N],self.n_xl)
            Para_lh     = np.reshape(Para_l,self.n_pl)
            xq_h        = np.zeros(self.nq*self.n_xi*(self.N+1))
            for j in range(self.nq):
                xq_j    = xq_traj[j]
                xq_jp   = np.reshape(xq_j,self.n_xi*(self.N+1)) # row-by-row
                xq_h[j*self.n_xi*(self.N+1):(j+1)*self.n_xi*(self.N+1)]=xq_jp
            Jlh         = np.reshape(Jl,3)
            rgh         = np.reshape(rg,3)
            # Parameter_l = np.hstack((xl_fbh,ref_xl,ref_ul,Para_lh,xq_h,Jlh,rgh))
            # Parameter_l = np.reshape(parameterl,(1,len(parameterl)))
            opt_sol_l   = self.MPCsolverPayload_acados(xl_fbh, xq_h, ref_xl, ref_ul, Para_lh, Jlh, rgh)
            xl_opt      = np.array(opt_sol_l['xl_opt'])
            ul_opt      = np.array(opt_sol_l['ul_opt'])
            cox_opt_l   = opt_sol_l['costatel_opt']
            # cox_ipopt_l = opt_sol_l['costatel_ipopt']
            # Ref_add     = ul_opt # update the addition using the optimized cable force
            sum_viol_xl = 0
            sum_viol_ul = 0
            sum_viol_cxl= 0
            for kl in range(len(ul_traj)):
                sum_viol_xl  += LA.norm(xl_opt[kl,:]-xl_traj[kl,:])
                sum_viol_ul  += LA.norm(ul_opt[kl,:]-ul_traj[kl,:])
                # sum_viol_cxl += LA.norm(cox_opt_l[kl,:]-cox_ipopt_l[kl,:])
            sum_viol_xl  += LA.norm(xl_opt[-1,:]-xl_traj[-1,:])
            viol_xl  = sum_viol_xl/len(xl_opt)
            viol_ul  = sum_viol_ul/len(ul_opt)
            # viol_cxl = sum_viol_cxl/len(cox_opt_l)
            viol_x_list += [viol_xl]
            viol_u_list += [viol_ul]
            initial_error = LA.norm(np.reshape(xl_opt[0,:],(self.n_xl,1))-xl_fb)
            print('iteration=',ke,'payload:','viol_xl=',format(viol_xl,'.5f'),'viol_ul=',format(viol_ul,'.5f'),'viol_x0l=',initial_error)
            # update the payload's trajectories
            xl_traj  = xl_opt
            ul_traj  = ul_opt

            # compute the maximum violation value
            viol  = np.concatenate((viol_x_list,viol_u_list))
            if ke>1:
                max_violation = np.max(viol)
            print('iteration=',ke,'max_violation=',format(max_violation,'.5f'))
            # update the iteration number
            ke += 1

        # output
        opt_system = {"xq_traj":xq_traj,
                      "uq_traj":uq_traj,
                      "xl_traj":xl_traj,
                      "ul_traj":ul_traj,
                      "cx_quad":cx_quad,
                      "cx_load":cox_opt_l
                      }
        
        return opt_system
    
        
    def DiffKKT_quadrotor(self):
        assert hasattr(self, 'xi'), "Define the quadrotor's state variable first!"
        assert hasattr(self, 'ui'), "Define the quadrotor's contrl variable first!"
        
        # define co-state variables
        self.costate    = SX.sym('cs', self.n_xi,1) # defined for completeness, not used in actual computation of all coefficient matrices
        self.next_cs    = SX.sym('cs_next', self.n_xi,1)

        # define the initial state (i.e., the current feedback)
        self.x_init     = SX.sym('x_init', self.n_xi,1)
        
        # define the Lagrangain
        self.L0         = self.Ji_k + self.gui_lb + self.gui_ub + self.hc_k + self.gq_k + self.next_cs.T@self.Modeli + self.costate.T@(self.x_init - self.xi)         # k=0
        self.Lk         = self.Ji_k + self.gui_lb + self.gui_ub + self.hc_k + self.gq_k + self.next_cs.T@self.Modeli - self.costate.T@self.xi  # k=1,...,N-1 
        self.LN         = self.Ji_N + self.hc_k + self.gq_k - self.costate.T@self.xi                                                        # k=N

        # differentiate the dynamics to get the system jacobian
        self.F          = jacobian(self.Modeli, self.xi)
        self.F_fn       = Function('F',[self.xi, self.ui, self.xl, self.ti, self.ind_q],[self.F],['xi0', 'ui0', 'xl0', 'ti0', 'i0'],['Ff'])
        self.G          = jacobian(self.Modeli, self.ui)
        self.G_fn       = Function('G',[self.xi, self.ui, self.xl, self.ti, self.ind_q],[self.G],['xi0', 'ui0', 'xl0', 'ti0', 'i0'],['Gf'])
        self.E_p        = np.zeros((self.n_xi, self.n_pi)) # used at k=0,...,N-1 for beta = weighting matrices (theta)
        self.E_xi0      = np.zeros((self.n_xi, self.n_xi)) # used at k=0,...,N-1 for beta = xi_init
        self.E_xl0      = jacobian(self.Modeli, self.xl) # used at k=0 only for beta = xl_init
        self.E_xl0_fn   = Function('E_xl',[self.xi, self.ui, self.xl, self.ti, self.ind_q],[self.E_xl0],['xi0', 'ui0', 'xl0', 'ti0', 'i0'],['E_xlf'])
        self.E_xlk      = np.zeros((self.n_xi, self.n_xl)) # used at k=1,...,N-1 for beta = xl_init
        
        self.E_ulk      = np.zeros((self.n_xi, self.n_ul)) # used at k=1,...,N-1 for beta = ul_0

        # first-order derivative of L0 (initial Lagrangain) k=0
        self.dL0x       = jacobian(self.L0, self.xi)
        self.dL0u       = jacobian(self.L0, self.ui)

        # first-order derivative of Lk (path Lagrangian) k=1,...,N-1
        self.dLkx       = jacobian(self.Lk, self.xi)
        self.dLku       = jacobian(self.Lk, self.ui)

        # first-order derivative of LN (terminal Lagrangain) k=N
        self.dLNx       = jacobian(self.LN, self.xi)

        """
        second-order derivative of L0
        """
        self.ddL0xx     = jacobian(self.dL0x, self.xi)
        self.ddL0xx_fn  = Function('ddL0xx',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddL0xx], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddL0xxf'])
        self.ddL0xu     = jacobian(self.dL0x, self.ui)
        self.ddL0xu_fn  = Function('ddL0xu',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddL0xu], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddL0xuf'])
        self.ddL0ux     = jacobian(self.dL0u, self.xi)
        self.ddL0ux_fn  = Function('ddL0ux',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddL0ux], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddL0uxf'])
        self.ddL0uu     = jacobian(self.dL0u, self.ui)
        self.ddL0uu_fn  = Function('ddL0uu',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddL0uu], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddL0uuf'])
        # when beta = weighting matrices (theta), denoted by 'p' in the code
        self.ddL0xp     = jacobian(self.dL0x, self.para_i)  
        self.ddL0xp_fn  = Function('ddL0xp',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddL0xp], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddL0xpf'])
        self.ddL0up     = jacobian(self.dL0u, self.para_i)
        self.ddL0up_fn  = Function('ddL0up',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddL0up], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddL0upf'])
        # when beta = xi_init, denoted by 'xi' in the code
        self.ddL0xxi    = np.zeros((self.n_xi, self.n_xi)) 
        self.ddL0uxi    = np.zeros((self.n_ui, self.n_xi)) 
        # when beta = xl_init, denoted by 'xl' in the code
        self.ddL0xxl    = jacobian(self.dL0x, self.xl) 
        self.ddL0xxl_fn = Function('ddL0xxl',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddL0xxl], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddL0xxlf'])
        self.ddL0uxl    = jacobian(self.dL0u, self.xl)
        self.ddL0uxl_fn = Function('ddL0uxl',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddL0uxl], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddL0uxlf'])
        
        
        """
        second-order derivative of Lk, k=1,...,N-1
        """
        self.ddLkxx     = jacobian(self.dLkx, self.xi)
        self.ddLkxx_fn  = Function('ddLkxx',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddLkxx], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddLkxxf'])
        self.ddLkxu     = jacobian(self.dLkx, self.ui)
        self.ddLkxu_fn  = Function('ddL0xu',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddLkxu], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddLkxuf'])
        self.ddLkux     = jacobian(self.dLku, self.xi)
        self.ddLkux_fn  = Function('ddL0ux',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddLkux], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddLkuxf'])
        self.ddLkuu     = jacobian(self.dLku, self.ui)
        self.ddLkuu_fn  = Function('ddL0uu',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddLkuu], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddLkuuf'])
        # when beta = weighting matrices (theta), denoted by 'p' in the code
        self.ddLkxp     = jacobian(self.dLkx, self.para_i)  
        self.ddLkxp_fn  = Function('ddLkxp',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddLkxp], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddLkxpf'])
        self.ddLkup     = jacobian(self.dLku, self.para_i)
        self.ddLkup_fn  = Function('ddLkup',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddLkup], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddLkupf'])
        # when beta = xi_init, denoted by 'xi' in the code
        self.ddLkxxi    = np.zeros((self.n_xi, self.n_xi))
        self.ddLkuxi    = np.zeros((self.n_ui, self.n_xi))
        # when beta = xl_init, denoted by 'xl' in the code
        self.ddLkxxl    = np.zeros((self.n_xi, self.n_xl))
        self.ddLkuxl    = np.zeros((self.n_ui, self.n_xl))
        # when beta = ul_0, denoted by 'ul' in the code
        self.ddLkxul    = np.zeros((self.n_xi, self.n_ul))
        self.ddLkuul    = np.zeros((self.n_ui, self.n_ul))

        """
        second-order derivative of LN
        """ 
        self.ddLNxx     = jacobian(self.dLNx, self.xi)
        self.ddLNxx_fn  = Function('ddLNxx',[self.xi, self.xl, self.x_qi, self.ref_xi, self.para_i, self.ind_q], [self.ddLNxx],
                                   ['xi0', 'xl0', 'xq0', 'refxi0', 'parai0', 'i0'], ['ddLNxxf'])
        # when beta = weighting matrices (theta), denoted by 'p' in the code
        self.ddLNxp     = jacobian(self.dLNx, self.para_i)
        self.ddLNxp_fn  = Function('ddLNxp',[self.xi, self.xl, self.x_qi, self.ref_xi, self.para_i, self.ind_q],  [self.ddLNxp],
                                   ['xi0', 'xl0', 'xq0', 'refxi0', 'parai0', 'i0'], ['ddLNxpf'])
        # when beta = xi_init, denoted by 'xi' in the code
        self.ddLNxxi    = np.zeros((self.n_xi, self.n_xi))
        # when beta = xl_init, denoted by 'xl' in the code
        self.ddLNxxl    = np.zeros((self.n_xi, self.n_xl))
        # when beta = (ul_0)i, denoted by 'ti' in the code
        self.ddLNxul    = np.zeros((self.n_xi, self.n_ul))
        

    def GetAuxSys_quadrotor(self, index, xq_traj, uq_traj, xl_traj, ul_traj, cx_quad_traj, Ref_xq, Ref_uq, CTRL_gain):
        self.ei         = jacobian(self.ul[index,0],self.ul) 
        self.E_ul0      = jacobian(self.Modeli, self.ti)@self.ei # used at k=0 only for beta = ul_0
        self.E_ul0_fn   = Function('E_ul',[self.xi, self.ui, self.xl, self.ti, self.ind_q],[self.E_ul0],['xi0', 'ui0', 'xl0', 'ti0', 'i0'],['E_ulf'])
        # when beta = ul_0, denoted by 'ul' in the code
        self.ddL0xul    = jacobian(self.dL0x, self.ti)@self.ei
        self.ddL0xul_fn = Function('ddL0xti',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddL0xul], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddL0xulf'])
        self.ddL0uul    = jacobian(self.dL0u, self.ti)@self.ei 
        self.ddL0uul_fn = Function('ddL0uti',[self.xi, self.ui, self.xl, self.ti, self.ind_q, self.next_cs, self.x_qi, self.ref_xi, self.ref_ui, self.para_i], [self.ddL0uul], 
                                   ['xi0', 'ui0', 'xl0', 'ti0', 'i0', 'cs_n', 'xq0', 'refxi0', 'refui0', 'parai0'], ['ddL0uulf'])
        # initialize the coefficient matrices of the auxiliary MPC system
        matF, matG = [], []
        matE_p, matE_xi, matE_xl, matE_ul = [], [], [], []
        matLxx, matLxu,  matLuu,  matLux  = [], [], [], []
        matLxp, matLxxi, matLxxl, matLxul = [], [], [], []
        matLup, matLuxi, matLuxl, matLuul = [], [], [], []
        # the corresponding weightings for the i-th quadrotor
        Para_i = CTRL_gain[index]
        # solve for the above coefficient matrices
        for k in range(self.N):
            curr_x    = xq_traj[index][k,:]
            curr_u    = uq_traj[index][k,:]
            next_cs   = cx_quad_traj[index][k,:] # we store the costate from k=1 to k=N, excluding the first costate at k=0. So, costate[k,:] is the next costate relative to the current timestep
            curr_xl   = xl_traj[k,:]
            curr_ti   = ul_traj[k,index]
            curr_xq   = np.zeros((2,self.nq-1))
            kj        = 0
            for j in range(self.nq):
                if j!=index:
                    xj_opt      = xq_traj[j]
                    curr_xq[:,kj:kj+1] = np.reshape(xj_opt[k,0:2],(2,1))
                    kj         += 1
            curr_refx = Ref_xq[index][:,k]
            curr_refu = Ref_uq[index][:,k]
            matF      += [self.F_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index)['Ff'].full()] # the first element will be used in the sensitivity propagation
            matG      += [self.G_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index)['Gf'].full()]
            matE_p    += [self.E_p]
            matE_xi   += [self.E_xi0]
            if k == 0:
                matE_xl     += [self.E_xl0_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index)['E_xlf'].full()]
                matE_ul     += [self.E_ul0_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index)['E_ulf'].full()]
                matLxx      += [self.ddL0xx_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddL0xxf'].full()]
                matLxu      += [self.ddL0xu_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddL0xuf'].full()]
                matLux      += [self.ddL0ux_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddL0uxf'].full()]
                matLuu      += [self.ddL0uu_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddL0uuf'].full()]
                matLxp      += [self.ddL0xp_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddL0xpf'].full()]            
                matLup      += [self.ddL0up_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddL0upf'].full()]
                matLxxi     += [self.ddL0xxi]
                matLuxi     += [self.ddL0uxi]
                matLxxl     += [self.ddL0xxl_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddL0xxlf'].full()]
                matLuxl     += [self.ddL0uxl_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddL0uxlf'].full()]
                matLxul     += [self.ddL0xul_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddL0xulf'].full()]
                matLuul     += [self.ddL0uul_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddL0uulf'].full()]
            else:
                matE_xl     += [self.E_xlk]
                matE_ul     += [self.E_ulk]
                matLxx      += [self.ddLkxx_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddLkxxf'].full()]
                matLxu      += [self.ddLkxu_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddLkxuf'].full()]
                matLux      += [self.ddLkux_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddLkuxf'].full()]
                matLuu      += [self.ddLkuu_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddLkuuf'].full()]
                matLxp      += [self.ddLkxp_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddLkxpf'].full()]
                matLup      += [self.ddLkup_fn(xi0=curr_x,ui0=curr_u,xl0=curr_xl,ti0=curr_ti,i0=index,cs_n=next_cs,xq0=curr_xq,refxi0=curr_refx,refui0=curr_refu,parai0=Para_i)['ddLkupf'].full()]
                matLxxi     += [self.ddLkxxi]
                matLuxi     += [self.ddLkuxi]
                matLxxl     += [self.ddLkxxl]
                matLuxl     += [self.ddLkuxl]
                matLxul     += [self.ddLkxul]
                matLuul     += [self.ddLkuul]
        curr_x   = xq_traj[index][-1,:]
        curr_xl  = xl_traj[-1,:]
        curr_xq  = np.zeros((2,self.nq-1))
        kj       = 0
        for j in range(self.nq):
            if j!=index:
                xj_opt      = xq_traj[j]
                curr_xq[:,kj:kj+1] = np.reshape(xj_opt[-1,0:2],(2,1))
                kj         += 1
        curr_refx = Ref_xq[index][:,-1]
        matLxx     += [self.ddLNxx_fn(xi0=curr_x,xl0=curr_xl,xq0=curr_xq,refxi0=curr_refx,parai0=Para_i,i0=index)['ddLNxxf'].full()]
        matLxp     += [self.ddLNxp_fn(xi0=curr_x,xl0=curr_xl,xq0=curr_xq,refxi0=curr_refx,parai0=Para_i,i0=index)['ddLNxpf'].full()]
        matLxxi    += [self.ddLNxxi]
        matLxxl    += [self.ddLNxxl]
        matLxul    += [self.ddLNxul]

        auxSys = {"matF"   : matF,
                  "matG"   : matG,
                  "matE_p" : matE_p,
                  "matE_xi": matE_xi,
                  "matE_xl": matE_xl,
                  "matE_ul": matE_ul,
                  "matLxx" : matLxx,
                  "matLxu" : matLxu,
                  "matLux" : matLux,
                  "matLuu" : matLuu,
                  "matLxp" : matLxp,
                  "matLup" : matLup,
                  "matLxxi": matLxxi,
                  "matLuxi": matLuxi,
                  "matLxxl": matLxxl,
                  "matLuxl": matLuxl,
                  "matLxul": matLxul,
                  "matLuul": matLuul
                  }
        return auxSys
        
        

    def DiffKKT_payload(self):
        assert hasattr(self, 'xl'), "Define the payload's state variable first!"
        assert hasattr(self, 'ul'), "Define the payload's contrl variable first!"
        # define co-state variables
        self.costatel   = SX.sym('csl', self.n_xl,1) # defined for completeness, not used in actual computation of all coefficient matrices
        self.next_csl   = SX.sym('csl_next', self.n_xl,1)

        # define the initial state (i.e., the current feedback)
        self.xl_init    = SX.sym('xl_init', self.n_xl,1)

        # define the Lagrangain
        self.Ll0        = self.Jl_k + self.gul_lb + self.gul_ub + self.hl_k + self.next_csl.T@self.Modell + self.costatel.T@(self.xl_init - self.xl)  # k=0
        self.Llk        = self.Jl_k + self.gul_lb + self.gul_ub + self.hl_k + self.next_csl.T@self.Modell - self.costatel.T@self.xl       # k=1,...,N-1
        self.LlN        = self.Jl_N + self.hl_k - self.costatel.T@self.xl 

        # differentiate the dynamics to get the system jacobian
        self.Fl         = jacobian(self.Modell, self.xl)
        self.Fl_fn      = Function('Fl',[self.xl, self.ul, self.xq, self.loadp], [self.Fl], ['xl0', 'ul0', 'xq0', 'loadp0'], ['Flf'])
        self.Gl         = jacobian(self.Modell, self.ul)
        self.Gl_fn      = Function('Gl',[self.xl, self.ul, self.xq, self.loadp], [self.Gl], ['xl0', 'ul0', 'xq0', 'loadp0'], ['Glf'])
        self.El_p       = np.zeros((self.n_xl, self.n_pl)) # used at k=0,...,N-1 for beta = weighting matrices (thetal)
        self.El_xl0     = np.zeros((self.n_xl, self.n_xl)) # used at k=0,...,N-1 for beta = xl_init
        
        self.El_xik     = np.zeros((self.n_xl, self.n_xi)) # used at k=1,...,N-1 for beta = xi_init
        self.El_lp      = jacobian(self.Modell, self.loadp) # used at k=0,...,N-1 for beta = load's inertial parameters (thetap)
        self.El_lp_fn   = Function('El_lp',[self.xl, self.ul, self.xq, self.loadp], [self.El_lp], ['xl0', 'ul0', 'xq0', 'loadp0'], ['El_lpf'])

        # first-order derivative of Ll0 (initial Lagrangain) k=0
        self.dLl0x      = jacobian(self.Ll0, self.xl)
        self.dLl0u      = jacobian(self.Ll0, self.ul)

        # first-order derivative of Llk (path Lagrangian) k=1,...,N-1
        self.dLlkx      = jacobian(self.Llk, self.xl)
        self.dLlku      = jacobian(self.Llk, self.ul)

        # first-order derivative of LlN (terminal Lagrangain) k=N
        self.dLlNx      = jacobian(self.LlN, self.xl)

        """
        second-order derivative of Ll0
        """
        self.ddLl0xx    = jacobian(self.dLl0x, self.xl)
        self.ddLl0xx_fn = Function('ddLl0xx',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLl0xx],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLl0xxf'])
        self.ddLl0xu    = jacobian(self.dLl0x, self.ul)
        self.ddLl0xu_fn = Function('ddLl0xu',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLl0xu],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLl0xuf'])
        self.ddLl0ux    = jacobian(self.dLl0u, self.xl)
        self.ddLl0ux_fn = Function('ddLl0ux',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLl0ux],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLl0uxf'])
        self.ddLl0uu    = jacobian(self.dLl0u, self.ul)
        self.ddLl0uu_fn = Function('ddLl0uu',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLl0uu],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLl0uuf'])
        # when beta = weighting matrices (thetal), denoted by 'p' in the code
        self.ddLl0xp    = jacobian(self.dLl0x, self.para_l)
        self.ddLl0xp_fn = Function('ddLl0xp',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLl0xp],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLl0xpf'])
        self.ddLl0up    = jacobian(self.dLl0u, self.para_l)
        self.ddLl0up_fn = Function('ddLl0up',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLl0up],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLl0upf'])
        # when beta = xl_init, denoted by 'xl' in the code
        self.ddLl0xxl   = np.zeros((self.n_xl, self.n_xl))
        self.ddLl0uxl   = np.zeros((self.n_ul, self.n_xl))
        
        # when beta = load's inertial parameters, denoted by 'lp' in the code
        self.ddLl0xlp   = jacobian(self.dLl0x, self.loadp)
        self.ddLl0xlp_fn= Function('ddLl0xlp',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLl0xlp],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLl0xlpf'])
        self.ddLl0ulp   = jacobian(self.dLl0u, self.loadp)
        self.ddLl0ulp_fn= Function('ddLl0ulp',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLl0ulp],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLl0ulpf'])
        
        """
        second-order derivative of Llk, k=1,...,N-1
        """
        self.ddLlkxx    = jacobian(self.dLlkx, self.xl)
        self.ddLlkxx_fn = Function('ddLlkxx',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLlkxx],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLlkxxf'])
        self.ddLlkxu    = jacobian(self.dLlkx, self.ul)
        self.ddLlkxu_fn = Function('ddLlkxu',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLlkxu],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLlkxuf'])
        self.ddLlkux    = jacobian(self.dLlku, self.xl)
        self.ddLlkux_fn = Function('ddLlkux',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLlkux],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLlkuxf'])
        self.ddLlkuu    = jacobian(self.dLlku, self.ul)
        self.ddLlkuu_fn = Function('ddLlkuu',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLlkuu],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLlkuuf'])
        # when beta = weighting matrices (thetal), denoted by 'p' in the code
        self.ddLlkxp    = jacobian(self.dLlkx, self.para_l)
        self.ddLlkxp_fn = Function('ddLlkxp',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLlkxp],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLlkxpf'])
        self.ddLlkup    = jacobian(self.dLlku, self.para_l)
        self.ddLlkup_fn = Function('ddLlkup',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLlkup],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLlkupf'])
        # when beta = xl_init, denoted by 'xl' in the code
        self.ddLlkxxl   = np.zeros((self.n_xl, self.n_xl))
        self.ddLlkuxl   = np.zeros((self.n_ul, self.n_xl))
        # when beta = xi_init, denoted by 'xi' in the code
        self.ddLlkxxi   = np.zeros((self.n_xl, self.n_xi))
        self.ddLlkuxi   = np.zeros((self.n_ul, self.n_xi))
        # when beta = load's inertial parameters, denoted by 'lp' in the code
        self.ddLlkxlp   = jacobian(self.dLlkx, self.loadp)
        self.ddLlkxlp_fn= Function('ddLlkxlp',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLlkxlp],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLlkxlpf'])
        self.ddLlkulp   = jacobian(self.dLlku, self.loadp)
        self.ddLlkulp_fn= Function('ddLlkulp',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLlkulp],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLlkulpf'])
        """
        second-order derivative of LlN
        """
        self.ddLlNxx    = jacobian(self.dLlNx, self.xl)
        self.ddLlNxx_fn = Function('ddLlNxx',[self.xl, self.xq, self.ref_xl, self.para_l], [self.ddLlNxx], ['xl0', 'xq0', 'refxl0', 'paral0'], ['ddLlNxxf'])
        # when beta = weighting matrices (thetal), denoted by 'p' in the code
        self.ddLlNxp    = jacobian(self.dLlNx, self.para_l)
        self.ddLlNxp_fn = Function('ddLlNxp',[self.xl, self.xq, self.ref_xl, self.para_l], [self.ddLlNxp], ['xl0', 'xq0', 'refxl0', 'paral0'], ['ddLlNxpf'])
        # when beta = xl_init, denoted by 'xl' in the code
        self.ddLlNxxl   = np.zeros((self.n_xl, self.n_xl))
        # when beta = xi_init, denoted by 'xi' in the code
        self.ddLlNxxi   = np.zeros((self.n_xl, self.n_xi))
        # when beta = load's inertial parameters, denoted by 'lp' in the code
        self.ddLlNxlp   = np.zeros((self.n_xl, self.n_lp))


    def GetAuxSys_payload(self, index, xl_traj, ul_traj, xq_traj, cx_load_traj, Ref_xl, Ref_ul, Para_l, loadp):

        self.El_xi0     = jacobian(self.Modell, self.xq[:,index]) # used at k=0 only for beta = xi_init
        self.El_xi0_fn  = Function('El_xi0',[self.xl, self.ul, self.xq, self.loadp], [self.El_xi0], ['xl0', 'ul0', 'xq0', 'loadp0'], ['El_xi0f'])
        
        # when beta = xi_init, denoted by 'xi' in the code
        self.ddLl0xxi   = jacobian(self.dLl0x, self.xq[:,index])
        self.ddLl0xxi_fn= Function('ddLl0xxi',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLl0xxi],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLl0xxif'])
        self.ddLl0uxi   = jacobian(self.dLl0u, self.xq[:,index])
        self.ddLl0uxi_fn= Function('ddLl0uxi',[self.xl, self.ul, self.xq, self.next_csl, self.ref_xl, self.ref_ul, self.loadp, self.para_l], [self.ddLl0uxi],
                                   ['xl0', 'ul0', 'xq0', 'csl_n', 'refxl0', 'reful0', 'loadp0', 'paral0'], ['ddLl0uxif'])
        # initialize the coefficient matrices of the payload's auxiliary MPC system
        matFl, matGl = [], []
        matEl_p, matEl_xl, matEl_xi, matEl_lp = [], [], [], []
        matLlxx, matLlxu,  matLluu,  matLlux  = [], [], [], []
        matLlxp, matLlxxl, matLlxxi, matLlxlp = [], [], [], []
        matLlup, matLluxl, matLluxi, matLlulp = [], [], [], []

        # solve for the above coefficient matrices
        for k in range(self.N):
            curr_xl     = xl_traj[k,:]
            curr_ul     = ul_traj[k,:]
            next_csl    = cx_load_traj[k,:] # we store the costate from k=1 to k=N, excluding the first costate at k=0. So, costate[k,:] is the next costate relative to the current timestep
            curr_xq     = np.zeros((self.n_xi,self.nq))
            for i in range(self.nq):
                xi_opt      = xq_traj[i]
                curr_xq[:,i:i+1] = np.reshape(xi_opt[k,:],(self.n_xi,1))
            curr_refxl  = Ref_xl[:,k]
            curr_reful  = Ref_ul[:,k]
            matFl      += [self.Fl_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,loadp0=loadp)['Flf'].full()] # the first element will be used in the sensitivity propagation
            matGl      += [self.Gl_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,loadp0=loadp)['Glf'].full()]
            matEl_p    += [self.El_p]
            matEl_xl   += [self.El_xl0]
            matEl_lp   += [self.El_lp_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,loadp0=loadp)['El_lpf'].full()]
            if k==0:
                matEl_xi   += [self.El_xi0_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,loadp0=loadp)['El_xi0f'].full()]
                matLlxx    += [self.ddLl0xx_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLl0xxf'].full()]
                matLlxu    += [self.ddLl0xu_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLl0xuf'].full()]
                matLlux    += [self.ddLl0ux_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLl0uxf'].full()]
                matLluu    += [self.ddLl0uu_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLl0uuf'].full()]
                matLlxp    += [self.ddLl0xp_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLl0xpf'].full()]
                matLlup    += [self.ddLl0up_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLl0upf'].full()]
                matLlxxl   += [self.ddLl0xxl]
                matLluxl   += [self.ddLl0uxl]
                matLlxxi   += [self.ddLl0xxi_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLl0xxif'].full()]
                matLluxi   += [self.ddLl0uxi_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLl0uxif'].full()]
                matLlxlp   += [self.ddLl0xlp_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLl0xlpf'].full()]
                matLlulp   += [self.ddLl0ulp_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLl0ulpf'].full()]
            else:
                matEl_xi   += [self.El_xik]
                matLlxx    += [self.ddLlkxx_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLlkxxf'].full()]
                matLlxu    += [self.ddLlkxu_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLlkxuf'].full()]
                matLlux    += [self.ddLlkux_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLlkuxf'].full()]
                matLluu    += [self.ddLlkuu_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLlkuuf'].full()]
                matLlxp    += [self.ddLlkxp_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLlkxpf'].full()]
                matLlup    += [self.ddLlkup_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLlkupf'].full()]
                matLlxxl   += [self.ddLlkxxl]
                matLluxl   += [self.ddLlkuxl]
                matLlxxi   += [self.ddLlkxxi]
                matLluxi   += [self.ddLlkuxi]
                matLlxlp   += [self.ddLlkxlp_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLlkxlpf'].full()]
                matLlulp   += [self.ddLlkulp_fn(xl0=curr_xl,ul0=curr_ul,xq0=curr_xq,csl_n=next_csl,refxl0=curr_refxl,reful0=curr_reful,loadp0=loadp,paral0=Para_l)['ddLlkulpf'].full()]
        curr_xl    = xl_traj[-1,:]
        curr_xq    = np.zeros((self.n_xi,self.nq))
        for i in range(self.nq):
            xi_opt      = xq_traj[i]
            curr_xq[:,i:i+1] = np.reshape(xi_opt[-1,:],(self.n_xi,1))
        curr_refxl = Ref_xl[:,-1]
        matLlxx     += [self.ddLlNxx_fn(xl0=curr_xl,xq0=curr_xq,refxl0=curr_refxl,paral0=Para_l)['ddLlNxxf'].full()]
        matLlxp     += [self.ddLlNxp_fn(xl0=curr_xl,xq0=curr_xq,refxl0=curr_refxl,paral0=Para_l)['ddLlNxpf'].full()]
        matLlxxl    += [self.ddLlNxxl]
        matLlxxi    += [self.ddLlNxxi]
        matLlxlp    += [self.ddLlNxlp]

        auxSysl = {"matFl"   : matFl,
                   "matGl"   : matGl,
                   "matEl_p" : matEl_p,
                   "matEl_xl": matEl_xl,
                   "matEl_xi": matEl_xi,
                   "matEl_lp": matEl_lp,
                   "matLlxx" : matLlxx,
                   "matLlxu" : matLlxu,
                   "matLlux" : matLlux,
                   "matLluu" : matLluu,
                   "matLlxp" : matLlxp,
                   "matLlup" : matLlup,
                   "matLlxxl": matLlxxl,
                   "matLluxl": matLluxl,
                   "matLlxxi": matLlxxi,
                   "matLluxi": matLluxi,
                   "matLlxlp": matLlxlp,
                   "matLlulp": matLlulp
                   }
        return auxSysl

    



class MPC_gradient: # compute the gradients of the MPC's solution trajectories w.r.t the interested hyperparameters (using the PDP method)
    def __init__(self, xi, xl, ul, para_i, para_l, loadp, horizon):
        self.n_xi = xi.numel()
        self.n_xl = xl.numel()
        self.n_ul = ul.numel()
        self.n_pi = para_i.numel()
        self.n_pl = para_l.numel()
        self.n_lp = loadp.numel()
        self.N    = horizon

    def Gradient_solver_quadrotor(self,auxSys):
        matF, matG = auxSys['matF'], auxSys['matG']
        matE_p, matE_xi, matE_xl, matE_ul = auxSys['matE_p'], auxSys['matE_xi'], auxSys['matE_xl'], auxSys['matE_ul']
        matLxx, matLxu,  matLux,  matLuu  = auxSys['matLxx'], auxSys['matLxu'],  auxSys['matLux'],  auxSys['matLuu']
        matLxp, matLup, matLxxi, matLuxi  = auxSys['matLxp'], auxSys['matLup'],  auxSys['matLxxi'], auxSys['matLuxi']
        matLxxl,matLuxl,matLxul, matLuul  = auxSys['matLxxl'],auxSys['matLuxl'], auxSys['matLxul'], auxSys['matLuul']
        
        # solve the gradient trajectory using the PDP method
        I        = np.identity(self.n_xi)
        P        = self.N * [np.zeros((self.n_xi, self.n_xi))]
        W_p      = self.N * [np.zeros((self.n_xi, self.n_pi))] # for beta = weighting matrices (denoted as 'p')
        W_xi     = self.N * [np.zeros((self.n_xi, self.n_xi))] # for beta = xi_init (denoted as 'xi')
        W_xl     = self.N * [np.zeros((self.n_xi, self.n_xl))] # for beta = xl_init (denoted as 'xl')
        W_ul     = self.N * [np.zeros((self.n_xi, self.n_ul))] # for beta = (ul_0) (denoted as 'ul')
        P[-1]    = matLxx[-1]
        W_p[-1]  = matLxp[-1]
        W_xi[-1] = matLxxi[-1]
        W_xl[-1] = matLxxl[-1]
        W_ul[-1] = matLxul[-1]
        for k in range(self.N-1,0,-1):
            P_next    = P[k]
            W_p_next  = W_p[k]
            W_xi_next = W_xi[k]
            W_xl_next = W_xl[k]
            W_ul_next = W_ul[k]
            invLuu    = LA.inv(matLuu[k])
            GinvLuu   = matG[k]@invLuu
            LxuinvLuu = matLxu[k]@invLuu
            A_k       = matF[k] - GinvLuu@matLxu[k].T
            R_k       = GinvLuu@matG[k].T
            M_p_k     = matE_p[k] - GinvLuu@matLup[k]     # for beta = weighulng matrices (denoted as 'p')
            M_xi_k    = matE_xi[k] - GinvLuu@matLuxi[k]   # for beta = xi_init (denoted as 'xi')
            M_xl_k    = matE_xl[k] - GinvLuu@matLuxl[k]   # for beta = xl_init (denoted as 'xl')
            M_ul_k    = matE_ul[k] - GinvLuu@matLuul[k]   # for beta = (ul_0) (denoted as 'ul')
            Q_k       = matLxx[k] - LxuinvLuu@matLxu[k].T
            N_p_k     = matLxp[k] - LxuinvLuu@matLup[k]   # for beta = weighting matrices (denoted as 'p'), matLxp[0] is not used!
            N_xi_k    = matLxxi[k] - LxuinvLuu@matLuxi[k] # for beta = xi_init (denoted as 'xi')
            N_xl_k    = matLxxl[k] - LxuinvLuu@matLuxl[k] # for beta = xl_init (denoted as 'xl')
            N_ul_k    = matLxul[k] - LxuinvLuu@matLuul[k] # for beta = (ul_0) (denoted as 'ul')
            temp_mat  = A_k.T@P_next@LA.inv(I + R_k@P_next)
            P_curr    = Q_k + temp_mat@A_k
            W_p_curr  = temp_mat@(M_p_k - R_k@W_p_next) + A_k.T@W_p_next + N_p_k
            W_xi_curr = temp_mat@(M_xi_k - R_k@W_xi_next) + A_k.T@W_xi_next + N_xi_k
            W_xl_curr = temp_mat@(M_xl_k - R_k@W_xl_next) + A_k.T@W_xl_next + N_xl_k
            W_ul_curr = temp_mat@(M_ul_k - R_k@W_ul_next) + A_k.T@W_ul_next + N_ul_k

            P[k-1]    = P_curr
            W_p[k-1]  = W_p_curr
            W_xi[k-1] = W_xi_curr
            W_xl[k-1] = W_xl_curr
            W_ul[k-1] = W_ul_curr
        
        # compute the gradient of the first optimal control command u*i_0 w.r.t the interested hyperparameters
        invLuu    = LA.inv(matLuu[0])
        GinvLuu   = matG[0]@invLuu
        R_0       = GinvLuu@matG[0].T
        A_0       = matF[0] - GinvLuu@matLxu[0].T
        temp_mat2 = matG[0].T@P[0]@LA.inv(I + R_0@P[0])
        # when beta = weighting matrices (denoted as 'p')
        X_p_0     = np.zeros((self.n_xi, self.n_pi))
        M_p_0     = matE_p[0] - GinvLuu@matLup[0]
        U_p_0     = -invLuu@((matLxu[0].T + temp_mat2@A_0)@X_p_0 + temp_mat2@(M_p_0 - R_0@W_p[0]) + matG[0].T@W_p[0] + matLup[0])
        # when beta = xi_init (denoted as 'xi')
        X_xi_0    = np.identity(self.n_xi)
        M_xi_0    = matE_xi[0] - GinvLuu@matLuxi[0]
        U_xi_0    = -invLuu@((matLxu[0].T + temp_mat2@A_0)@X_xi_0 + temp_mat2@(M_xi_0 - R_0@W_xi[0]) + matG[0].T@W_xi[0] + matLuxi[0])
        # when beta = xl_init (denoted as 'xl')
        X_xl_0    = np.zeros((self.n_xi, self.n_xl))
        M_xl_0    = matE_xl[0] - GinvLuu@matLuxl[0]
        U_xl_0    = -invLuu@((matLxu[0].T + temp_mat2@A_0)@X_xl_0 + temp_mat2@(M_xl_0 - R_0@W_xl[0]) + matG[0].T@W_xl[0] + matLuxl[0])
        # when beta = (ul_0)i (denoted as 'ul')
        X_ul_0    = np.zeros((self.n_xi, self.n_ul))
        M_ul_0    = matE_ul[0] - GinvLuu@matLuul[0]
        U_ul_0    = -invLuu@((matLxu[0].T + temp_mat2@A_0)@X_ul_0 + temp_mat2@(M_ul_0 - R_0@W_ul[0]) + matG[0].T@W_ul[0] + matLuul[0])

        # select the first element (xxx_0) from the list, as it is generated using the current feedback from the actual systems
        quad_mat = {"Fi_0"   : matF[0],
                    "Gi_0"   : matG[0],
                    "Ei_xl_0": matE_xl[0], 
                    "Ei_ul_0": matE_ul[0],
                    "Ui_pi_0": U_p_0,
                    "Ui_xi_0": U_xi_0,
                    "Ui_xl_0": U_xl_0,
                    "Ui_ul_0": U_ul_0
                    }
        return quad_mat
    
    def Gradient_solver_quadrotor_openloop(self,auxSys):
        matF, matG = auxSys['matF'], auxSys['matG']
        matE_p, matE_xi, matE_xl, matE_ul = auxSys['matE_p'], auxSys['matE_xi'], auxSys['matE_xl'], auxSys['matE_ul']
        matLxx, matLxu,  matLux,  matLuu  = auxSys['matLxx'], auxSys['matLxu'],  auxSys['matLux'],  auxSys['matLuu']
        matLxp, matLup, matLxxi, matLuxi  = auxSys['matLxp'], auxSys['matLup'],  auxSys['matLxxi'], auxSys['matLuxi']
        matLxxl,matLuxl,matLxul, matLuul  = auxSys['matLxxl'],auxSys['matLuxl'], auxSys['matLxul'], auxSys['matLuul']
        
        # solve the gradient trajectory using the PDP method
        I        = np.identity(self.n_xi)
        P        = self.N * [np.zeros((self.n_xi, self.n_xi))]
        W_p      = self.N * [np.zeros((self.n_xi, self.n_pi))] # for beta = weighting matrices (denoted as 'p')
        X_p      = (self.N + 1) * [np.zeros((self.n_xi, self.n_pi))] # for beta = weighting matrices (denoted as 'p')

        P[-1]    = matLxx[-1]
        W_p[-1]  = matLxp[-1]
        for k in range(self.N-1,0,-1):
            P_next    = P[k]
            W_p_next  = W_p[k]
            invLuu    = LA.inv(matLuu[k])
            GinvLuu   = matG[k]@invLuu
            LxuinvLuu = matLxu[k]@invLuu
            A_k       = matF[k] - GinvLuu@matLxu[k].T
            R_k       = GinvLuu@matG[k].T
            M_p_k     = matE_p[k] - GinvLuu@matLup[k]     # for beta = weighulng matrices (denoted as 'p')
            Q_k       = matLxx[k] - LxuinvLuu@matLxu[k].T
            N_p_k     = matLxp[k] - LxuinvLuu@matLup[k]   # for beta = weighting matrices (denoted as 'p')
            temp_mat  = A_k.T@P_next@LA.inv(I + R_k@P_next)
            P_curr    = Q_k + temp_mat@A_k
            W_p_curr  = temp_mat@(M_p_k - R_k@W_p_next) + A_k.T@W_p_next + N_p_k
            P[k-1]    = P_curr
            W_p[k-1]  = W_p_curr
        
        # compute the gradient of the optimal state trajectory w.r.t the weightings
        for k in range(self.N):
            invLuu    = LA.inv(matLuu[k])
            GinvLuu   = matG[k]@invLuu
            R_k       = GinvLuu@matG[k].T
            A_k       = matF[k] - GinvLuu@matLxu[k].T
            temp_mat2 = matG[k].T@P[k]@LA.inv(I + R_k@P[k])
            M_p_k     = matE_p[k] - GinvLuu@matLup[k]
            U_p_k     = -invLuu@((matLxu[k].T + temp_mat2@A_k)@X_p[k] + temp_mat2@(M_p_k - R_k@W_p[k]) + matG[k].T@W_p[k] + matLup[k])
            X_p[k+1]  = matF[k]@X_p[k] + matG[k]@U_p_k + matE_p[k]
       
        return X_p
    

    def Gradient_solver_payload_2_quadi(self,auxSysl): # relative to the ith quadrotor
        matFl, matGl = auxSysl['matFl'], auxSysl['matGl']
        matEl_p, matEl_xi, matEl_xl, matEl_lp = auxSysl['matEl_p'], auxSysl['matEl_xi'], auxSysl['matEl_xl'], auxSysl['matEl_lp']
        matLlxx, matLlxu,  matLlux,  matLluu  = auxSysl['matLlxx'], auxSysl['matLlxu'],  auxSysl['matLlux'],  auxSysl['matLluu']
        matLlxp, matLlup, matLlxxi, matLluxi  = auxSysl['matLlxp'], auxSysl['matLlup'],  auxSysl['matLlxxi'], auxSysl['matLluxi']
        matLlxxl,matLluxl,matLlxlp, matLlulp  = auxSysl['matLlxxl'],auxSysl['matLluxl'], auxSysl['matLlxlp'], auxSysl['matLlulp']

        # solve the gradient trajectory using the PDP method
        I         = np.identity(self.n_xl)
        Pl        = self.N * [np.zeros((self.n_xl, self.n_xl))]
        Wl_p      = self.N * [np.zeros((self.n_xl, self.n_pl))] # for beta = weighting matrices (denoted as 'p')
        Wl_xi     = self.N * [np.zeros((self.n_xl, self.n_xi))] # for beta = xi_init (denoted as 'xi')
        Wl_xl     = self.N * [np.zeros((self.n_xl, self.n_xl))] # for beta = xl_init (denoted as 'xl')
        Wl_lp     = self.N * [np.zeros((self.n_xl, self.n_lp))] # for beta = payload inertial parameters (denoted as 'lp')
        Pl[-1]    = matLlxx[-1]
        Wl_p[-1]  = matLlxp[-1]
        Wl_xi[-1] = matLlxxi[-1]
        Wl_xl[-1] = matLlxxl[-1]
        Wl_lp[-1] = matLlxlp[-1]
        for k in range(self.N-1,0,-1):
            Pl_next    = Pl[k]
            Wl_p_next  = Wl_p[k]
            Wl_xi_next = Wl_xi[k]
            Wl_xl_next = Wl_xl[k]
            Wl_lp_next = Wl_lp[k]
            invLluu    = LA.inv(matLluu[k])
            GinvLluu   = matGl[k]@invLluu
            LxuinvLluu = matLlxu[k]@invLluu
            Al_k       = matFl[k] - GinvLluu@matLlxu[k].T
            Rl_k       = GinvLluu@matGl[k].T
            Ml_p_k     = matEl_p[k] - GinvLluu@matLlup[k]     # for beta = weighting matrices (denoted as 'p')
            Ml_xi_k    = matEl_xi[k] - GinvLluu@matLluxi[k]   # for beta = xi_init (denoted as 'xi')
            Ml_xl_k    = matEl_xl[k] - GinvLluu@matLluxl[k]   # for beta = xl_init (denoted as 'xl')
            Ml_lp_k    = matEl_lp[k] - GinvLluu@matLlulp[k]   # for beta = payload inertial parameters (denoted as 'lp')
            Ql_k       = matLlxx[k] - LxuinvLluu@matLlxu[k].T
            Nl_p_k     = matLlxp[k] - LxuinvLluu@matLlup[k]   # for beta = weighting matrices (denoted as 'p')
            Nl_xi_k    = matLlxxi[k] - LxuinvLluu@matLluxi[k] # for beta = xi_init (denoted as 'xi')
            Nl_xl_k    = matLlxxl[k] - LxuinvLluu@matLluxl[k] # for beta = xl_init (denoted as 'xl')
            Nl_lp_k    = matLlxlp[k] - LxuinvLluu@matLlulp[k] # for beta = payload inertial parameters (denoted as 'lp')
            templ_mat  = Al_k.T@Pl_next@LA.inv(I + Rl_k@Pl_next)
            Pl_curr    = Ql_k + templ_mat@Al_k
            Wl_p_curr  = templ_mat@(Ml_p_k - Rl_k@Wl_p_next) + Al_k.T@Wl_p_next + Nl_p_k
            Wl_xi_curr = templ_mat@(Ml_xi_k - Rl_k@Wl_xi_next) + Al_k.T@Wl_xi_next + Nl_xi_k
            Wl_xl_curr = templ_mat@(Ml_xl_k - Rl_k@Wl_xl_next) + Al_k.T@Wl_xl_next + Nl_xl_k
            Wl_lp_curr = templ_mat@(Ml_lp_k - Rl_k@Wl_lp_next) + Al_k.T@Wl_lp_next + Nl_lp_k

            Pl[k-1]    = Pl_curr
            Wl_p[k-1]  = Wl_p_curr
            Wl_xi[k-1] = Wl_xi_curr
            Wl_xl[k-1] = Wl_xl_curr
            Wl_lp[k-1] = Wl_lp_curr
        
        # compute the gradient of the first optimal control command u*i_0 w.r.t the interested hyperparameters
        invLluu    = LA.inv(matLluu[0])
        GinvLluu   = matGl[0]@invLluu
        Rl_0       = GinvLluu@matGl[0].T
        Al_0       = matFl[0] - GinvLluu@matLlxu[0].T
        templ_mat2 = matGl[0].T@Pl[0]@LA.inv(I + Rl_0@Pl[0])
        # when beta = weighting matrices (denoted as 'p')
        Xl_p_0     = np.zeros((self.n_xl, self.n_pl))
        Ml_p_0     = matEl_p[0] - GinvLluu@matLlup[0]
        Ul_p_0     = -invLluu@((matLlxu[0].T + templ_mat2@Al_0)@Xl_p_0 + templ_mat2@(Ml_p_0 - Rl_0@Wl_p[0]) + matGl[0].T@Wl_p[0] + matLlup[0])
        # when beta = xi_init (denoted as 'xi')
        Xl_xi_0    = np.zeros((self.n_xl, self.n_xi))
        Ml_xi_0    = matEl_xi[0] - GinvLluu@matLluxi[0]
        Ul_xi_0    = -invLluu@((matLlxu[0].T + templ_mat2@Al_0)@Xl_xi_0 + templ_mat2@(Ml_xi_0 - Rl_0@Wl_xi[0]) + matGl[0].T@Wl_xi[0] + matLluxi[0])
        # when beta = xl_init (denoted as 'xl')
        Xl_xl_0    = np.identity(self.n_xl)
        Ml_xl_0    = matEl_xl[0] - GinvLluu@matLluxl[0]
        Ul_xl_0    = -invLluu@((matLlxu[0].T + templ_mat2@Al_0)@Xl_xl_0 + templ_mat2@(Ml_xl_0 - Rl_0@Wl_xl[0]) + matGl[0].T@Wl_xl[0] + matLluxl[0])
        # when beta = payload inertial parameters (denoted as 'lp')
        Xl_lp_0    = np.zeros((self.n_xl, self.n_lp))
        Ml_lp_0    = matEl_lp[0] - GinvLluu@matLlulp[0]
        Ul_lp_0    = -invLluu@((matLlxu[0].T + templ_mat2@Al_0)@Xl_lp_0 + templ_mat2@(Ml_lp_0 - Rl_0@Wl_lp[0]) + matGl[0].T@Wl_lp[0] + matLlulp[0])
        
        # select the first element (xxx_0) from the list, as it is generated using the current feedback from the actual systems
        load_mat = {"Fl_0"   : matFl[0],
                    "Gl_0"   : matGl[0],
                    "El_xi_0": matEl_xi[0],
                    "El_lp_0": matEl_lp[0],
                    "Ul_pl_0": Ul_p_0,
                    "Ul_xi_0": Ul_xi_0,
                    "Ul_xl_0": Ul_xl_0,
                    "Ul_lp_0": Ul_lp_0
                   }
        return load_mat
    

    def Gradient_solver_payload_openloop(self,auxSysl): # relative to the ith quadrotor
        matFl, matGl = auxSysl['matFl'], auxSysl['matGl']
        matEl_p, matEl_xi, matEl_xl, matEl_lp = auxSysl['matEl_p'], auxSysl['matEl_xi'], auxSysl['matEl_xl'], auxSysl['matEl_lp']
        matLlxx, matLlxu,  matLlux,  matLluu  = auxSysl['matLlxx'], auxSysl['matLlxu'],  auxSysl['matLlux'],  auxSysl['matLluu']
        matLlxp, matLlup, matLlxxi, matLluxi  = auxSysl['matLlxp'], auxSysl['matLlup'],  auxSysl['matLlxxi'], auxSysl['matLluxi']
        matLlxxl,matLluxl,matLlxlp, matLlulp  = auxSysl['matLlxxl'],auxSysl['matLluxl'], auxSysl['matLlxlp'], auxSysl['matLlulp']

        # solve the gradient trajectory using the PDP method
        I         = np.identity(self.n_xl)
        Pl        = self.N * [np.zeros((self.n_xl, self.n_xl))]
        Wl_p      = self.N * [np.zeros((self.n_xl, self.n_pl))] # for beta = weighting matrices (denoted as 'p')
        Xl_p      = (self.N+1) * [np.zeros((self.n_xl, self.n_pl))] # for beta = weighting matrices (denoted as 'p')
        Pl[-1]    = matLlxx[-1]
        Wl_p[-1]  = matLlxp[-1]
        
        for k in range(self.N-1,0,-1):
            Pl_next    = Pl[k]
            Wl_p_next  = Wl_p[k]
            invLluu    = LA.inv(matLluu[k])
            GinvLluu   = matGl[k]@invLluu
            LxuinvLluu = matLlxu[k]@invLluu
            Al_k       = matFl[k] - GinvLluu@matLlxu[k].T
            Rl_k       = GinvLluu@matGl[k].T
            Ml_p_k     = matEl_p[k] - GinvLluu@matLlup[k]     # for beta = weighting matrices (denoted as 'p')
            Ql_k       = matLlxx[k] - LxuinvLluu@matLlxu[k].T
            Nl_p_k     = matLlxp[k] - LxuinvLluu@matLlup[k]   # for beta = weighting matrices (denoted as 'p')
            templ_mat  = Al_k.T@Pl_next@LA.inv(I + Rl_k@Pl_next)
            Pl_curr    = Ql_k + templ_mat@Al_k
            Wl_p_curr  = templ_mat@(Ml_p_k - Rl_k@Wl_p_next) + Al_k.T@Wl_p_next + Nl_p_k
            Pl[k-1]    = Pl_curr
            Wl_p[k-1]  = Wl_p_curr
           
        
        # compute the gradient of the optimal state trajectory w.r.t the weightings
        for k in range(self.N):
            invLluu    = LA.inv(matLluu[k])
            GinvLluu   = matGl[k]@invLluu
            Rl_k       = GinvLluu@matGl[k].T
            Al_k       = matFl[k] - GinvLluu@matLlxu[k].T
            templ_mat2 = matGl[k].T@Pl[k]@LA.inv(I + Rl_k@Pl[k])
            Ml_p_k     = matEl_p[k] - GinvLluu@matLlup[k]
            Ul_p_k     = -invLluu@((matLlxu[0].T + templ_mat2@Al_k)@Xl_p[k] + templ_mat2@(Ml_p_k - Rl_k@Wl_p[k]) + matGl[k].T@Wl_p[k] + matLlup[k])
            Xl_p[k+1]  = matFl[k]@Xl_p[k] + matGl[k]@Ul_p_k + matEl_p[k]
        
        return Xl_p
    
    

class Sensitivity_propagation: # compute the sensitivities (gradients) of the actual system states w.r.t the interested hyperparameters (online training)
    def __init__(self, uav_para, xi, xl, ul, para_i, para_l, loadp, horizon_loss, horizon):
        self.nq      = int(uav_para[4])
        self.n_xi    = xi.numel()
        self.n_xl    = xl.numel()
        self.n_ul    = ul.numel()
        self.n_pi    = para_i.numel()
        self.n_pl    = para_l.numel()
        self.n_lp    = loadp.numel()
        self.N_loss  = horizon_loss # horizon_loss can be longer than the MPC's horizon
        self.N       = horizon # MPC horizon
        self.xi      = xi
        self.xl      = xl
        self.pvi     = self.xi[0:6,:] # pvi: position and velocity of the i-th quadrotor
        self.pvl     = self.xl[0:6,:] # pvl: position and velocity of the load
        self.refxi   = SX.sym('refxi',6,1) # reference position and velocity of the i-th quadrotor
        self.refxl   = SX.sym('refxl',self.n_xl,1) # reference state of the load
        weight_hl_i  = 5e3*np.array([1,1,1, 1,1,1]) # weight for the high-level loss function for the i-th quadrotor
        weight_hl_l  = 2e4*np.array([1,1,2, 1,1,1.25, 5,5,5, 10,10,1]) # weight for the high-level loss function for the load
        self.error_i = self.pvi - self.refxi
        p_error_l    = self.xl[0:3,0] - self.refxl[0:3,0]
        v_error_l    = self.xl[3:6,0] - self.refxl[3:6,0]
        ql           = self.xl[6:10,0]
        refql        = self.refxl[6:10,0]
        Rl           = self.q_2_rotation(ql)
        Rdl          = self.q_2_rotation(refql)
        error_Rl     = Rdl.T@Rl - Rl.T@Rdl
        att_error_l  = 1/2*self.vee_map(error_Rl)
        w_error_l    = self.xl[10:13,0] - self.refxl[10:13,0]
        self.error_l = vertcat(p_error_l,v_error_l,att_error_l,w_error_l)
        self.loss_i  = self.error_i.T@np.diag(weight_hl_i)@self.error_i
        self.loss_l  = self.error_l.T@np.diag(weight_hl_l)@self.error_l

    def q_2_rotation(self, q): # from body frame to inertial frame
        # no normalization to avoid singularity in optimization
        q0, q1, q2, q3 = q[0,0], q[1,0], q[2,0], q[3,0] # q0 denotes a scalar while q1, q2, and q3 represent rotational axes x, y, and z, respectively
        R = vertcat(
        horzcat( 2 * (q0 ** 2 + q1 ** 2) - 1, 2 * q1 * q2 - 2 * q0 * q3, 2 * q0 * q2 + 2 * q1 * q3),
        horzcat(2 * q0 * q3 + 2 * q1 * q2, 2 * (q0 ** 2 + q2 ** 2) - 1, 2 * q2 * q3 - 2 * q0 * q1),
        horzcat(2 * q1 * q3 - 2 * q0 * q2, 2 * q0 * q1 + 2 * q2 * q3, 2 * (q0 ** 2 + q3 ** 2) - 1)
        )
        return R
    
    def vee_map(self, v):
        vect = vertcat(v[2, 1], v[0, 2], v[1, 0])
        return vect
    
        
    def quadrotor_sensitivity(self, Quads_Mat, Load_Mat, Xl_pl, sumXi_pl, index):
        # define the sensitivity trajectories, starting with zero matrices (the sensitivities of the actual states w.r.t the parameters are all zeros!)
        Xi_pi       = (self.N_loss+1)*[np.zeros((self.n_xi, self.n_pi))] # X^i_pi: the gradient of the i-th quadrotor's state w.r.t its MPC's weighting matrices
        Xi_pl       = (self.N_loss+1)*[np.zeros((self.n_xi, self.n_pl))] # X^i_pl: the gradient of the i-th quadrotor's state w.r.t the payload MPC's weighting matrices
        Xl_pi       = (self.N_loss+1)*[np.zeros((self.n_xl, self.n_pi))] # X^l_pi: the gradient of the payload's state w.r.t the i-th quadrotor MPC's weighting matrices
        

        for t in range(self.N_loss): # in a receding horizon manner, selecting the latest N_loss elements
            # select the first element (xxx_0) from the list, as it is generated using the current feedback from the actual systems
            Fi_t    = Quads_Mat[len(Quads_Mat)-self.N_loss+t][index]['Fi_0']
            Gi_t    = Quads_Mat[len(Quads_Mat)-self.N_loss+t][index]['Gi_0']
            Ei_xl_t = Quads_Mat[len(Quads_Mat)-self.N_loss+t][index]['Ei_xl_0']
            Ei_ul_t = Quads_Mat[len(Quads_Mat)-self.N_loss+t][index]['Ei_ul_0']
            Ui_xi_t = Quads_Mat[len(Quads_Mat)-self.N_loss+t][index]['Ui_xi_0']
            Ui_xl_t = Quads_Mat[len(Quads_Mat)-self.N_loss+t][index]['Ui_xl_0']
            Ui_ul_t = Quads_Mat[len(Quads_Mat)-self.N_loss+t][index]['Ui_ul_0']
            Ui_pi_t = Quads_Mat[len(Quads_Mat)-self.N_loss+t][index]['Ui_pi_0']
            Fl_t    = Load_Mat[len(Load_Mat)-self.N_loss+t][index]['Fl_0']
            Gl_t    = Load_Mat[len(Load_Mat)-self.N_loss+t][index]['Gl_0']
            El_xi_t = Load_Mat[len(Load_Mat)-self.N_loss+t][index]['El_xi_0']
            Ul_xi_t = Load_Mat[len(Load_Mat)-self.N_loss+t][index]['Ul_xi_0']
            Ul_xl_t = Load_Mat[len(Load_Mat)-self.N_loss+t][index]['Ul_xl_0']
            Ul_pl_t = Load_Mat[len(Load_Mat)-self.N_loss+t][index]['Ul_pl_0']
            
            # compute the system matrices for the sensitivity propagation
            dfidxi  = Fi_t + Gi_t@(Ui_xi_t + Ui_ul_t@Ul_xi_t) + Ei_ul_t@Ul_xi_t
            dfidxl  = Ei_xl_t + Gi_t@(Ui_xl_t + Ui_ul_t@Ul_xl_t) + Ei_ul_t@Ul_xl_t
            dfidul  = Gi_t@Ui_ul_t + Ei_ul_t
            dfldxl  = Fl_t + Gl_t@Ul_xl_t
            dfldxi  = El_xi_t + Gl_t@Ul_xi_t
            # sensitivity propagation
            Xi_pi[t+1] = dfidxi@Xi_pi[t] + dfidxl@Xl_pi[t] + Gi_t@Ui_pi_t
            Xi_pl[t+1] = dfidxi@Xi_pl[t] + dfidxl@Xl_pl[t] + dfidul@Ul_pl_t
            Xl_pi[t+1] = dfldxl@Xl_pi[t] + dfldxi@Xi_pi[t]
            sumXi_pl[t] = sumXi_pl[t] + dfldxi@Xi_pl[t]
            
        
        quad_sensitivity = {"Xi_pi": Xi_pi,
                            "Xi_pl": Xi_pl,
                            "Xl_pi": Xl_pi
                            }
        return quad_sensitivity, sumXi_pl
    
    
    def quadrotor_sensitivity_nocoupling(self, Quads_Mat, Load_Mat, index):
        # define the sensitivity trajectories, starting with zero matrices (the sensitivities of the actual states w.r.t the parameters are all zeros!)
        Xi_pi       = (self.N_loss+1)*[np.zeros((self.n_xi, self.n_pi))] # X^i_pi: the gradient of the i-th quadrotor's state w.r.t its MPC's weighting matrices

        for t in range(self.N_loss): # in a receding horizon manner, selecting the latest N_loss elements
            # select the first element (xxx_0) from the list, as it is generated using the current feedback from the actual systems
            Fi_t    = Quads_Mat[len(Quads_Mat)-self.N_loss+t][index]['Fi_0']
            Gi_t    = Quads_Mat[len(Quads_Mat)-self.N_loss+t][index]['Gi_0']
            Ui_xi_t = Quads_Mat[len(Quads_Mat)-self.N_loss+t][index]['Ui_xi_0']
            Ui_pi_t = Quads_Mat[len(Quads_Mat)-self.N_loss+t][index]['Ui_pi_0']
            
            # compute the system matrices for the sensitivity propagation
            dfidxi  = Fi_t + Gi_t@Ui_xi_t
            # sensitivity propagation
            Xi_pi[t+1] = dfidxi@Xi_pi[t] + Gi_t@Ui_pi_t
                            
        return Xi_pi
    
    
    def payload_sensitivity(self, Load_Mat, sumXi_pl):
        # define the sensitivity trajectory, starting with zero matrices (the sensitivities of the actual states w.r.t the parameters are all zeros!)
        Xl_pl       = (self.N_loss+1)*[np.zeros((self.n_xl, self.n_pl))] # X^l_pl: the gradient of the payload's state w.r.t its MPC's weighting matrices
        
        for t in range(self.N_loss):
            # the following jacobian matrices are independent of the quadrotor's index, so simply "0" is chosen
            Fl_t    = Load_Mat[len(Load_Mat)-self.N_loss+t][0]['Fl_0']
            Gl_t    = Load_Mat[len(Load_Mat)-self.N_loss+t][0]['Gl_0']
            Ul_xl_t = Load_Mat[len(Load_Mat)-self.N_loss+t][0]['Ul_xl_0']
            Ul_pl_t = Load_Mat[len(Load_Mat)-self.N_loss+t][0]['Ul_pl_0']
            # compute the system matrices for the sensitivity propagation
            dfldxl  = Fl_t + Gl_t@Ul_xl_t
            # sensitivity propagation
            Xl_pl[t+1] = dfldxl@Xl_pl[t] + sumXi_pl[t] + Gl_t@Ul_pl_t
            
        return Xl_pl
    
    
    def payload_sensitivity_nocoupling(self, Load_Mat):
        # define the sensitivity trajectory, starting with zero matrices (the sensitivities of the actual states w.r.t the parameters are all zeros!)
        Xl_pl       = (self.N_loss+1)*[np.zeros((self.n_xl, self.n_pl))] # X^l_pl: the gradient of the payload's state w.r.t its MPC's weighting matrices
        
        for t in range(self.N_loss):
            # the following jacobian matrices are independent of the quadrotor's index, so simply "0" is chosen
            Fl_t    = Load_Mat[len(Load_Mat)-self.N_loss+t][0]['Fl_0']
            Gl_t    = Load_Mat[len(Load_Mat)-self.N_loss+t][0]['Gl_0']
            Ul_xl_t = Load_Mat[len(Load_Mat)-self.N_loss+t][0]['Ul_xl_0']
            Ul_pl_t = Load_Mat[len(Load_Mat)-self.N_loss+t][0]['Ul_pl_0']
            # compute the system matrices for the sensitivity propagation
            dfldxl  = Fl_t + Gl_t@Ul_xl_t
            # sensitivity propagation
            Xl_pl[t+1] = dfldxl@Xl_pl[t] + Gl_t@Ul_pl_t
            
        return Xl_pl
    
   
    def Distributed_sensitivity(self, Quads_Mat, Load_Mat):
        epsilon = 1e-3    # threshold for stopping the iteration, 1e-4 for testing the convergence step, 1e-3 is used in training
        k_max   = 5      # maximum number of iterations
        max_violation = 5 # initial value of max_violation
        k = 1             # iteration index
        # initialization
        Xi_Pi_Prev    = []
        Xi_Pl_Prev    = []
        Xl_Pi_Prev    = []
        for i in range(self.nq):
            Xi_pi_prev    = (self.N_loss+1)*[np.zeros((self.n_xi, self.n_pi))]
            Xi_pl_prev    = (self.N_loss+1)*[np.zeros((self.n_xi, self.n_pl))]
            Xl_pi_prev    = (self.N_loss+1)*[np.zeros((self.n_xl, self.n_pi))]
            Xi_Pi_Prev   += [Xi_pi_prev]
            Xi_Pl_Prev   += [Xi_pl_prev]
            Xl_Pi_Prev   += [Xl_pi_prev]
        
        Xl_pl         = (self.N_loss+1)*[np.zeros((self.n_xl, self.n_pl))] 
        Xl_pl_prev    = (self.N_loss+1)*[np.zeros((self.n_xl, self.n_pl))]
        k_step = 0

        while max_violation>=epsilon and k<=k_max:
            sumXi_pl         = (self.N_loss)*[np.zeros((self.n_xl, self.n_pl))] # initialize sumXi_pl
            
            viol_X_list      = []
            All_quad_sens_pi = [] # list for saving all the updated sensitivity trajectories during the 'for' loop
            All_quad_sens_il = []
            All_quad_sens_li = []
          
            for i in range(self.nq): # later, we will upgrade these iterations to parallel computing
                quad_sensitivity, sumXi_pl = self.quadrotor_sensitivity(Quads_Mat, Load_Mat, Xl_pl, sumXi_pl,i)
                Xi_pi  = quad_sensitivity['Xi_pi']
                Xi_pl  = quad_sensitivity['Xi_pl']
                Xl_pi  = quad_sensitivity['Xl_pi']
                
                All_quad_sens_pi += [Xi_pi]
                All_quad_sens_il += [Xi_pl]
                All_quad_sens_li += [Xl_pi]

                sum_Xi_pi_viol = 0
                sum_Xi_pl_viol = 0
                sum_Xl_pi_viol = 0
                for t in range(self.N_loss+1):
                    sum_Xi_pi_viol += LA.norm(Xi_pi[t] - Xi_Pi_Prev[i][t], ord=2)
                    sum_Xi_pl_viol += LA.norm(Xi_pl[t] - Xi_Pl_Prev[i][t], ord=2)
                    sum_Xl_pi_viol += LA.norm(Xl_pi[t] - Xl_Pi_Prev[i][t], ord=2)
                    
                viol_Xi_pi = sum_Xi_pi_viol/(self.N_loss+1)
                viol_Xi_pl = sum_Xi_pl_viol/(self.N_loss+1)
                viol_Xl_pi = sum_Xl_pi_viol/(self.N_loss+1)   
                viol_X_list += [viol_Xi_pi]
                viol_X_list += [viol_Xi_pl]
                viol_X_list += [viol_Xl_pi]      
                print('sensitivity: iteration:',k,'quadrotor:',i,'viol_Xi_pi=',format(viol_Xi_pi,'.5f'),'viol_Xi_pl=',format(viol_Xi_pl,'.5f'),'viol_Xl_pi=',format(viol_Xl_pi,'.5f'))
                # update
                Xi_Pi_Prev[i] = Xi_pi
                Xi_Pl_Prev[i] = Xi_pl
                Xl_Pi_Prev[i] = Xl_pi
            # solve the sensitivity of the payload using the updated quadrotor trajectories sumXi_pl
            Xl_pl  = self.payload_sensitivity(Load_Mat, sumXi_pl)
            sum_Xl_pl_viol = 0
           
            for t in range(self.N_loss+1):
                sum_Xl_pl_viol += LA.norm(Xl_pl[t] - Xl_pl_prev[t], ord=2)
              
            viol_Xl_pl = sum_Xl_pl_viol/(self.N_loss+1)
            print('sensitivity: iteration-',k,'payload:','viol_Xl_pl=',format(viol_Xl_pl,'.5f'))
            viol_X_list += [viol_Xl_pl]
            # compute the maximum violation value
            if k>1:
                max_violation = np.max(viol_X_list)
            print('sensitivity: iteration-',k,'max_violation=',format(max_violation,'.5f'))
            # update
            Xl_pl_prev = Xl_pl
            # update the iteration number
            k += 1

        return All_quad_sens_pi, All_quad_sens_il, All_quad_sens_li, Xl_pl
    

    def Distributed_sensitivity_nocoupling(self, Quads_Mat, Load_Mat):
        
        All_quad_sens_pi = [] # list for saving all the updated sensitivity trajectories during the 'for' loop
          
          
        for i in range(self.nq): # later, we will upgrade these iterations to parallel computing
            Xi_pi = self.quadrotor_sensitivity_nocoupling(Quads_Mat, Load_Mat,i)     
            All_quad_sens_pi += [Xi_pi]           
            
        # solve the sensitivity of the payload using the updated quadrotor trajectories sumXi_pl
        Xl_pl  = self.payload_sensitivity_nocoupling(Load_Mat)
            

        return All_quad_sens_pi, Xl_pl
    
    
    def loss_quadrotor(self, xi, refxi):
        # define the quadrotor's high-level loss function
        lossi_fn  = Function('lossi',[self.xi, self.refxi], [self.loss_i], ['xi0', 'refxi0'], ['lossif'])
        loss_quad_i = lossi_fn(xi0=xi,refxi0=refxi)['lossif'].full()
        return loss_quad_i
    
    def loss_load(self, xl, refxl):
        # define the load's high-level loss function
        lossl_fn  = Function('lossl',[self.xl, self.refxl], [self.loss_l], ['xl0', 'refxl0'], ['losslf'])
        loss_load = lossl_fn(xl0=xl,refxl0=refxl)['losslf'].full()
        return loss_load
    
    def loss_i_horizon(self, xi_traj, Refxi):
        loss_i   = 0
        for t in range(self.N_loss):
            xi      = xi_traj[len(xi_traj)-self.N_loss+t]
            refxi   = Refxi[len(Refxi)-self.N_loss+t]
            lossi_t = self.loss_quadrotor(xi,refxi) 
            loss_i += lossi_t
        return loss_i
    
    def loss_l_horizon(self, xl_traj, Refxl):
        loss_l   = 0
        for t in range(self.N_loss):
            xl      = xl_traj[len(xl_traj)-self.N_loss+t]
            refxl   = Refxl[len(Refxl)-self.N_loss+t]
            lossl_t = self.loss_load(xl,refxl) 
            loss_l += lossl_t
        return loss_l

    
    def ChainRule_quadrotor_i(self, index, Quad_State, Ref_Quad, All_quad_sens_pi, All_quad_sens_il):
        # define the gradient of loss_i w.r.t the state xi
        dlids     = jacobian(self.loss_i, self.xi)
        dlids_fn  = Function('dlids',[self.xi, self.refxi], [dlids], ['xi0','refxi0'], ['dlidsf'])
        # initialize the parameter gradient
        dpi       = np.zeros((1,self.n_pi))
        dpil      = np.zeros((1,self.n_pl))
        # initialize the high-level loss_i
        loss_i    = 0
        # select the corresponding sensitivity trajectroy from the list 'All_quad_sens'
        Xi_pi     = All_quad_sens_pi[index] # for the i-th quadrotor
        Xi_pl     = All_quad_sens_il[index]

        for t in range(self.N_loss):
            xi         = Quad_State[len(Quad_State)-self.N_loss+t][index]
            refxi      = Ref_Quad[len(Ref_Quad)-self.N_loss+t][index]
            loss_i_t   = self.loss_quadrotor(xi,refxi)
            loss_i    += loss_i_t
            Dlids      = dlids_fn(xi0=xi,refxi0=refxi)['dlidsf'].full()
            Xi_pi_t1   = Xi_pi[t+1] # we skip the initial state at t=0, as its gradient w.r.t the weightings is zero!
            dpi       += Dlids@Xi_pi_t1
            Xi_pl_t1   = Xi_pl[t+1]
            dpil      += Dlids@Xi_pl_t1
           
        return dpi, dpil, loss_i
    
    
    def ChainRule_quadrotor_i_nocoupling(self, index, Quad_State, Ref_Quad, All_quad_sens_pi):
        # define the gradient of loss_i w.r.t the state xi
        dlids     = jacobian(self.loss_i, self.xi)
        dlids_fn  = Function('dlids',[self.xi, self.refxi], [dlids], ['xi0','refxi0'], ['dlidsf'])
        # initialize the parameter gradient
        dpi       = np.zeros((1,self.n_pi))
        # initialize the high-level loss_i
        loss_i    = 0
        # select the corresponding sensitivity trajectroy from the list 'All_quad_sens'
        Xi_pi     = All_quad_sens_pi[index] # for the i-th quadrotor

        for t in range(self.N_loss):
            xi         = Quad_State[len(Quad_State)-self.N_loss+t][index]
            refxi      = Ref_Quad[len(Ref_Quad)-self.N_loss+t][index]
            loss_i_t   = self.loss_quadrotor(xi,refxi)
            loss_i    += loss_i_t
            Dlids      = dlids_fn(xi0=xi,refxi0=refxi)['dlidsf'].full()
            Xi_pi_t1   = Xi_pi[t+1] # we skip the initial state at t=0, as its gradient w.r.t the weightings is zero!
            dpi       += Dlids@Xi_pi_t1
           
        return dpi, loss_i
    
    
    def ChainRule_quadrotor_i_openloop(self, index, Quad_State, Ref_Quad, xq_traj, Ref_xq, Xq_p):
        # define the gradient of loss_i w.r.t the state xi
        dlids     = jacobian(self.loss_i, self.xi)
        dlids_fn  = Function('dlids',[self.xi, self.refxi], [dlids], ['xi0','refxi0'], ['dlidsf'])
        # initialize the parameter gradient
        dpi       = np.zeros((1,self.n_pi))
        # initialize the high-level loss_i
        loss_i    = 0
        # select the corresponding sensitivity trajectroy from the list 'All_quad_sens'
        Xi_pi     = Xq_p[index] # for the i-th quadrotor
        for t in range(self.N_loss):
            xi         = Quad_State[len(Quad_State)-self.N_loss+t][index]
            refxi      = Ref_Quad[len(Ref_Quad)-self.N_loss+t][index]
            loss_i_t   = self.loss_quadrotor(xi,refxi)
            loss_i    += loss_i_t
        for t in range(self.N):
            xi         = xq_traj[index][t+1,:]
            refxi      = Ref_xq[index][0:6,t+1]
            Dlids      = dlids_fn(xi0=xi,refxi0=refxi)['dlidsf'].full()
            Xi_pi_t1   = Xi_pi[t+1] # we skip the initial state at t=0, as its gradient w.r.t the weightings is zero!
            dpi       += Dlids@Xi_pi_t1
        return dpi, loss_i
    

    def ChainRule_load(self, Load_State, Ref_Load, Xl_pl, All_quad_sens_li):
        # define the gradient of loss_l w.r.t the state xl
        dllds     = jacobian(self.loss_l, self.xl)
        dllds_fn  = Function('dllds',[self.xl, self.refxl], [dllds], ['xl0','refxl0'], ['dlldsf'])
        # initialize the parameter gradient
        dpl       = np.zeros((1,self.n_pl))
        Dli       = self.nq*[np.zeros((1,self.n_pi))]
        # initialize the high-level loss_l
        loss_l    = 0
        for t in range(self.N_loss):
            xl         = Load_State[len(Load_State)-self.N_loss+t] 
            refxl      = Ref_Load[len(Ref_Load)-self.N_loss+t]
            loss_l_t   = self.loss_load(xl,refxl)
            loss_l    += loss_l_t
            Dllds      = dllds_fn(xl0=xl,refxl0=refxl)['dlldsf'].full()
            Xl_pl_t1   = Xl_pl[t+1] # we skip the initial state at t=0, as its gradient w.r.t the weightings is zero!
            dpl       += Dllds@Xl_pl_t1
            for i in range(self.nq):
                Xl_pi  = All_quad_sens_li[i]
                Xl_pi_t1 = Xl_pi[t+1]
                Dli[i] = Dli[i] + Dllds@Xl_pi_t1
           
        return Dli, dpl,loss_l
    

    def ChainRule_load_nocoupling(self, Load_State, Ref_Load, Xl_pl):
        # define the gradient of loss_l w.r.t the state xl
        dllds     = jacobian(self.loss_l, self.xl)
        dllds_fn  = Function('dllds',[self.xl, self.refxl], [dllds], ['xl0','refxl0'], ['dlldsf'])
        # initialize the parameter gradient
        dpl       = np.zeros((1,self.n_pl))
        # initialize the high-level loss_l
        loss_l    = 0
        for t in range(self.N_loss):
            xl         = Load_State[len(Load_State)-self.N_loss+t] 
            refxl      = Ref_Load[len(Ref_Load)-self.N_loss+t]
            loss_l_t   = self.loss_load(xl,refxl)
            loss_l    += loss_l_t
            Dllds      = dllds_fn(xl0=xl,refxl0=refxl)['dlldsf'].full()
            Xl_pl_t1   = Xl_pl[t+1] # we skip the initial state at t=0, as its gradient w.r.t the weightings is zero!
            dpl       += Dllds@Xl_pl_t1
            
        return dpl,loss_l
    
    
    def ChainRule_load_openloop(self, Load_State, Ref_Load, xl_traj, Ref_xl, Xl_pl):
        # define the gradient of loss_l w.r.t the state xl
        dllds     = jacobian(self.loss_l, self.xl)
        dllds_fn  = Function('dllds',[self.xl, self.refxl], [dllds], ['xl0','refxl0'], ['dlldsf'])
        # initialize the parameter gradient
        dpl       = np.zeros((1,self.n_pl))
        # initialize the high-level loss_l
        loss_l    = 0
        for t in range(self.N_loss):
            xl         = Load_State[len(Load_State)-self.N_loss+t] 
            refxl      = Ref_Load[len(Ref_Load)-self.N_loss+t]
            loss_l_t   = self.loss_load(xl,refxl)
            loss_l    += loss_l_t
        for t in range(self.N):
            xl         = xl_traj[t+1,:]
            refxl      = Ref_xl[:,t+1]
            Dllds      = dllds_fn(xl0=xl,refxl0=refxl)['dlldsf'].full()
            Xl_pl_t1   = Xl_pl[t+1] # we skip the initial state at t=0, as its gradient w.r.t the weightings is zero!
            dpl       += Dllds@Xl_pl_t1
        return dpl, loss_l
    

    
  

            
    
    
    








    

    
    


        




            







        







        





        






        

                    












        




    
