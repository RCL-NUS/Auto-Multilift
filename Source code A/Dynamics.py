"""
This file defines the simulation environment for a multi-lifting system
-------------------------------------------------------------
Wang Bingheng, 10 Jan. 2024 at Control and Simulation Lab, ECE Dept. NUS
----------------
1st version
"""

from casadi import *
import numpy as np
from numpy import linalg as LA
import math
from multiprocessing import Process, Array

class multilifting:
    def __init__(self, uav_para, load_para, cable_para, dt_sample):
        # Quadrotor's inertial parameters (mass, rotational inertia)
        # In the initial stage, we assume that all quadrotors share the same inertial parameters, i.g., homogeneous multi-lifting systems.
        # We assume such a scenario where n homogeneous quadrotors work together to transport a round payload with n cables.
        self.mi     = uav_para[0] # the quadrotor's mass
        self.Ji     = np.diag([uav_para[1], uav_para[2], uav_para[3]]) # rotational inertia in the quadrotor body frame {Bi}
        self.nq     = int(uav_para[4]) # number of quadrotors
        # Payload's known parameters
        self.ml     = load_para[0] # the payload's mass
        self.rl     = load_para[1] # radius of the payload
        # Payload's unknown parameters
        self.Jldiag = SX.sym('Jl',3,1) # rotational inertia about CoM
        self.rg     = SX.sym('rg',3,1) # location of the center-of-mass (CoM) in {Bl}
        self.S_rg   = self.skew_sym(self.rg)
        self.Jl     = diag(self.Jldiag) + self.ml* self.S_rg@self.S_rg.T # rotational inertia in the payload body frame {Bl}, obtained by the parallel axis theorem
        # Cable's parameters
        self.E      = cable_para[0] # Young's modulus
        self.A      = cable_para[1] # cross-section area
        # self.K      = cable_para[0] # stiffness
        self.ct     = cable_para[2] # damping coefficient
        self.cl0    = cable_para[3] # cable natural length
        # Quadrotor's state and control
        self.pi     = SX.sym('pi',3,1) # position of CoM in the world frame {I}
        self.vi     = SX.sym('vi',3,1) # velocity of CoM in {I}
        self.qi     = SX.sym('qi',4,1) # quaternion of quadrotor
        self.wi     = SX.sym('wi',3,1) # angular velocity in the quadrotor's body frame {Bi}
        self.fi     = SX.sym('fi',1,1) # total thrust in {Bi}
        self.Mi     = SX.sym('Mi',3,1) # 3-by-1 control torque in {Bi}
        self.xi     = vertcat(self.pi, self.vi, self.qi, self.wi) # 13-by-1 vector
        self.nxi    = self.xi.numel()
        self.ui     = vertcat(self.fi, self.Mi) # 4-by-1 vector
        self.ti     = SX.sym('ti')     # tension magnitude for the i-th quadrotor
        self.nui    = self.ui.numel()
        # Payload's state and control
        self.pl     = SX.sym('pl',3,1) # position of center-of-origin (CO) of the payload's body frame {Bl} in {I}
        self.vl     = SX.sym('vl',3,1) # velocity of CO in {Bl}
        self.ql     = SX.sym('ql',4,1) # quaternion of payload
        self.wl     = SX.sym('wl',3,1) # angular velocity in {Bl}
        self.xl     = vertcat(self.pl, self.vl, self.ql, self.wl) # 13-by-1 vector
        self.nxl    = self.xl.numel()
        self.ul     = SX.sym('ul',self.nq,1) # magnitudes of n cable tensions used in MPC only
        self.nul    = self.ul.numel()
        # Gravitational acceleration 
        self.g      = 9.81
        # Simulation time step used in MPC
        self.dt     = dt_sample
        # Unit direction vectors free of coordinate
        self.ex     = vertcat(1, 0, 0)
        self.ey     = vertcat(0, 1, 0)
        self.ez     = vertcat(0, 0, 1)
        # Polynomial coefficients of reference trajectory
        self.polyc = SX.sym('c',1,8)
        # Time in polynomial
        self.time  = SX.sym('t')
        # Initial time in polynomial
        self.time0 = SX.sym('t0')
        #-----------variables used in L1-AC-----------#
        # Predicted state z_hat
        self.z_hat = SX.sym('zhat',3,1)
        # Matched disturbance in body frame
        self.dm    = SX.sym('dm',1,1)
        # Unmatched disturbance in body frame
        self.dum   = SX.sym('dum',2,1)
        # Hurwitz matrix
        self.As    = SX.sym('As',3,3)
        #-----------Reference parameters--------------#
        self.rc    = 3 # radius of the circle, 5 for testing
        self.Tc    = 20 # 15 for training, 20 for evaluation, fig-8
        self.hc    = 2  # height of the circle
        self.wc    = 2*np.pi/self.Tc # desired angular velocity of the circle reference

    def dir_cosine(self, Euler):
        # Euler angles for roll, pitch and yaw
        gamma, theta, psi = Euler[0,0], Euler[1,0], Euler[2,0]
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(gamma), math.sin(gamma)],
                        [0, -math.sin(gamma),  math.cos(gamma)]])
        R_y = np.array([[ math.cos(theta), 0, -math.sin(theta)],
                        [0,                1,              0],
                        [math.sin(theta), 0, math.cos(theta)]])
        R_z = np.array([[math.cos(psi), math.sin(psi), 0],
                        [-math.sin(psi),  math.cos(psi), 0],
                        [0,                          0, 1]])
        # Rotation matrix from world frame to body frame, X->Z->Y
        R_wb= np.matmul(np.matmul(R_y, R_z), R_x)
        # Rotation matrix from body frame to world frame, Y->Z->X
        R_bw= np.transpose(R_wb)
        return R_bw
    
    def skew_sym(self, v): # skew-symmetric operator
        v_cross = vertcat(
            horzcat(0, -v[2,0], v[1,0]),
            horzcat(v[2,0], 0, -v[0,0]),
            horzcat(-v[1,0], v[0,0], 0)
        )
        return v_cross
    
    def skew_sym_numpy(self, v):
        v_cross = np.array([
            [0, -v[2, 0], v[1, 0]],
            [v[2, 0], 0, -v[0, 0]],
            [-v[1, 0], v[0, 0], 0]]
        )
        return v_cross

    def q_2_rotation(self, q, normal): # from body frame to inertial frame
        if normal == 1:
            q = q/norm_2(q) # normalization
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
    
    def lowpass_filter(self, time_const, curr_i, prev_i, dt):
        alpha       = dt/(dt+time_const)
        y_filter    = (1-alpha)*prev_i + alpha*curr_i
        return y_filter
    
    def tension_magnitude(self, L, dL):
        # L: stretched cable length 
        # dL: changing rate of the cable length
        f_tc = self.E*self.A/self.cl0*(L-self.cl0)+ self.ct*self.E*self.A/self.cl0*dL # computed tension magnitude
        # f_tc = self.K*(L-self.cl0)+ self.ct*self.K*dL # computed tension magnitude
        if f_tc < 0:
            f_t = 0
        elif f_tc > 2e2: # avoid any potentially unstable performance caused by an extremely large tension force
            f_t = 2e2
        else:
            f_t = f_tc
        return f_t
    
    def ith_cable_force(self, xi, xl, index):
        pi      = vertcat(xi[0,0], xi[1,0], xi[2,0])
        vi      = vertcat(xi[3,0], xi[4,0], xi[5,0])
        pl      = vertcat(xl[0,0], xl[1,0], xl[2,0])
        vl      = vertcat(xl[3,0], xl[4,0], xl[5,0]) # in {Bl}
        ql      = vertcat(xl[6,0], xl[7,0], xl[8,0], xl[9,0])
        wl      = vertcat(xl[10,0], xl[11,0], xl[12,0])
        Rl      = self.q_2_rotation(ql, 1)
        rli     = np.array([[self.rl*cos(index*self.alpha),self.rl*sin(index*self.alpha),0]]).T
        pli     = pi - pl - Rl@rli
        L       = LA.norm(pli) # length of the i-th cable
        # dL      = (L-L_prev)/dt
        # time_ct = 2*dt
        # dL_lpf  = self.lowpass_filter(time_ct, dL, dL_lpf_prev, dt)
        # analytical time-derivative of the cable length, Feb 20, 2024
        S_wl    = self.skew_sym(wl)
        vli     = vi - Rl@vl - Rl@S_wl@rli
        dL      = vli.T@pli/L
        f_t     = self.tension_magnitude(L, dL)
        return f_t

    def payload_dyn(self, xl, ul, xq, normal):
        """
        configuration of the payload tether attachment points:
        start from the first point [rl;0;0]
        allocate the remaining attachment points anticlockwisely
        the separation angle between any two neighbouring attachment points is defined by alpha = 2*pi/nq
        the coordinate of the n-th attachment point is [rl*cos((n-1)*alpha);rl*sin((n-1)*alpha);0]
        ----------------
        For the payload, xq={xi} (with i from 1 to nq) are regarded as the external signals.
        """
        # Position in {I}
        pl      = vertcat(xl[0,0], xl[1,0], xl[2,0])
        # Velocity in {Bl}
        vl      = vertcat(xl[3,0], xl[4,0], xl[5,0])
        # Quaternion
        ql      = vertcat(xl[6,0], xl[7,0], xl[8,0], xl[9,0])
        # Angular velocity in {Bl}
        wl      = vertcat(xl[10,0], xl[11,0], xl[12,0])
        # Rotation matrix from {Bl} to {I}
        Rl      = self.q_2_rotation(ql,normal)
        # Cable forces and torques
        self.alpha   = 2*np.pi/self.nq
        sum_Tl  = np.zeros((3,1)) # total cable forces acting on the payload in {Bl}
        sum_Ttl = np.zeros((3,1)) # total cable torques acting on the payload in {Bl}
        for i in range(self.nq):
            rli      = vertcat(self.rl*cos(i*self.alpha),self.rl*sin(i*self.alpha),0) # attachment point in {Bl}
            xi       = xq[:,i] # the state of the i-th quadrotor
            pi       = vertcat(xi[0,0],xi[1,0],xi[2,0])
            pli      = pi - pl - Rl@rli
            Ti       = ul[i,0]*pli/norm_2(pli) # tension vector in {I} from the attachment point to the CoM of the i-th quadrotor
            Tli      = Rl.T@Ti # express the tension vector in {Bl}
            sum_Tl  += Tli
            S_rli    = self.skew_sym(rli)
            Ttli     = S_rli@Tli # tension torque of the i-th cable in {Bl}
            sum_Ttl += Ttli

        # Kinematics
        dpl     = Rl@vl
        S_wl    = self.skew_sym(wl)
        omega   = vertcat(
            horzcat(0, -wl.T),
            horzcat(wl,-S_wl)
        )
        dql     = 1/2*omega@ql

        # Dynamics (both the translational and rotational dynamics are in {Bl})
        
        Ml      = vertcat(
            horzcat(self.ml*np.identity(3),-self.ml*self.S_rg),
            horzcat(self.ml*self.S_rg, self.Jl) 
        )
        Gl      = -self.ml*self.g*self.ez # the payload's gravity in {I}
        Cl      = vertcat(
            sum_Tl + Rl.T@Gl - self.ml*S_wl@vl - self.ml*S_wl@(S_wl@self.rg),
            sum_Ttl + self.S_rg@Rl.T@Gl - S_wl@(self.Jl@wl) - self.ml*self.S_rg@S_wl@vl
        )
        dVl     = inv(Ml)@Cl
        dvl     = vertcat(dVl[0,0],dVl[1,0],dVl[2,0])
        dwl     = vertcat(dVl[3,0],dVl[4,0],dVl[5,0])

        # Model
        model_l = vertcat(dpl, dvl, dql, dwl)

        return model_l
    
    
    def quadrotor_dyn(self, xi, ui, xl, ti, i, normal):
        assert hasattr(self, 'alpha'), "Define the separation angle first!"
        """
        For i-th quadrotor ('i' is the index), xl (the payload's state) is regarded as the external signal.
        """
        # Position in {I}
        pi      = vertcat(xi[0,0], xi[1,0], xi[2,0])
        # Velocity in {I}
        vi      = vertcat(xi[3,0], xi[4,0], xi[5,0])
        # Quaternion
        qi      = vertcat(xi[6,0], xi[7,0], xi[8,0], xi[9,0])
        # Angular velocity in {Bi}
        wi      = vertcat(xi[10,0], xi[11,0], xi[12,0])
        # Rotation matrix from {Bi} to {I}
        Ri      = self.q_2_rotation(qi,normal)
        # The payload's position in {I}
        pl      = vertcat(xl[0,0], xl[1,0], xl[2,0])
        # The payload's quaternion
        ql      = vertcat(xl[6,0], xl[7,0], xl[8,0], xl[9,0])
        # Rotation matrix from {Bl} to {I}
        Rl      = self.q_2_rotation(ql,normal)
        # Cable force in {I}
        rli     = vertcat(self.rl*cos(i*self.alpha),self.rl*sin(i*self.alpha),0) # attachment point in {Bl}
        pli     = pi - pl - Rl@rli
        Til     = -ti*pli/norm_2(pli)
        # Total thrust
        f       = ui[0,0]
        # Control torque in {Bi}
        tau     = vertcat(ui[1,0], ui[2,0], ui[3,0])

        # Kinematics
        dpi     = vi
        S_wi    = self.skew_sym(wi)
        omega   = vertcat(
            horzcat(0, -wi.T),
            horzcat(wi,-S_wi)
        )
        dqi     = 1/2*omega@qi

        # Dynamics
        dvi     = -self.g*self.ez + 1/self.mi*(Ri@(f*self.ez) + Til) # in {I}
        dwi     = LA.inv(self.Ji)@(-S_wi@(self.Ji@wi) + tau) # in {Bi}

        # Model
        model_i = vertcat(dpi, dvi, dqi, dwi)

        return model_i
    
    def L1_AC_quadrotor_model(self):
        assert hasattr(self, 'alpha'), "Define the separation angle first!"
        # Rotation matrix from {Bi} to {I}
        Ri      = self.q_2_rotation(self.qi,1)
        # Rotation matrix from {Bl} to {I}
        Rl      = self.q_2_rotation(self.ql,1)
        # Cable force in {I}
        rli     = vertcat(self.rl*cos(self.index_q*self.alpha),self.rl*sin(self.index_q*self.alpha),0) # attachment point in {Bl}
        pli     = self.pi - self.pl - Rl@rli
        Til     = -self.ti*pli/norm_2(pli)
        # Dynamics model used in the L1-AC state predictor
        f_z     = -self.g*self.ez + 1/self.mi*Til
        B       = 1/self.mi*Ri@self.ez
        Brp     = horzcat(1/self.mi*Ri@self.ex,1/self.mi*Ri@self.ey)
        self.z  = self.vi
        dz_hat  = f_z + B@(self.fi + self.dm) + Brp@self.dum + self.As@(self.z_hat - self.z)
        self.dzhat_fn= Function('dz_hat',[self.z_hat,self.xi,self.ui,self.xl,self.ti,self.dm,self.dum,self.As,self.index_q],[dz_hat],['zhat0','xi0','ui0','xl0','ti0','dm0','dum0','As0','i0'],['dz_hatf'])

    
    def model(self):
        # Payload: 4-th order Runge-Kutta discretization of the model used in MPC (symbolic computation)
        self.xq      = SX.sym('xq',self.nxi, self.nq)
        k1l_normal   = self.payload_dyn(self.xl, self.ul, self.xq, 1)
        k1l     = self.payload_dyn(self.xl, self.ul, self.xq, 0) # no normalization for quaternion to avoid singularity in optimization
        k2l     = self.payload_dyn(self.xl + self.dt/2*k1l, self.ul, self.xq, 0)
        k3l     = self.payload_dyn(self.xl + self.dt/2*k2l, self.ul, self.xq, 0)
        k4l     = self.payload_dyn(self.xl + self.dt*k3l, self.ul, self.xq, 0)
        self.model_l = (k1l + 2*k2l + 2*k3l + k4l)/6
        self.dynl      = k1l # load dynamics model used in Acados
        # Payload: dynamic model used in the step function (the time-step in simulation may be different from that used in MPC)
        self.dyn_l_fn = Function('k1l',[self.xl, self.ul, self.xq, self.Jldiag, self.rg], [k1l_normal], ['xl0', 'ul0', 'xq0', 'Jldiag0', 'rg0'], ['k1lf'])

        # Quadrotor: 4-th order Runge Kutta discretization of the model used in MPC (symbolic computation)
        self.index_q = SX.sym('iq') # the index of the quadrotor (e.g., 1-st, 2-nd, 3-rd...i-th)
        k1i_normal   = self.quadrotor_dyn(self.xi, self.ui, self.xl, self.ti, self.index_q, 1)
        k1i     = self.quadrotor_dyn(self.xi, self.ui, self.xl, self.ti, self.index_q, 0)
        k2i     = self.quadrotor_dyn(self.xi + self.dt/2*k1i, self.ui, self.xl, self.ti, self.index_q, 0)
        k3i     = self.quadrotor_dyn(self.xi + self.dt/2*k2i, self.ui, self.xl, self.ti, self.index_q, 0)
        k4i     = self.quadrotor_dyn(self.xi + self.dt*k3i, self.ui, self.xl, self.ti, self.index_q, 0)
        self.model_i = (k1i + 2*k2i + 2*k3i + k4i)/6
        self.dyni      = k1i # i-th quadrotor dynamics model used in Acados
        # Quadrotor: dynamic model used in the step function (the time-step in simulation may be different from that used in MPC)
        self.dyn_i_fn = Function('k1i',[self.xi, self.ui, self.xl, self.ti, self.index_q], [k1i_normal], ['xi0', 'ui0', 'xl0', 'ti0', 'index0'], ['k1if'])

        self.L1_AC_quadrotor_model()

    def step_load(self, xl, ul, xq, Jl, rg, dt):
        # Payload: 4-th order Runge-Kutta discrete-time model
        k1l     = self.dyn_l_fn(xl0=xl, ul0=ul, xq0=xq, Jldiag0=Jl, rg0=rg)['k1lf'].full()
        k2l     = self.dyn_l_fn(xl0=xl+dt/2*k1l, ul0=ul, xq0=xq, Jldiag0=Jl, rg0=rg)['k1lf'].full()
        k3l     = self.dyn_l_fn(xl0=xl+dt/2*k2l, ul0=ul, xq0=xq, Jldiag0=Jl, rg0=rg)['k1lf'].full()
        k4l     = self.dyn_l_fn(xl0=xl+dt*k3l, ul0=ul, xq0=xq, Jldiag0=Jl, rg0=rg)['k1lf'].full()
        xldot   = (k1l + 2*k2l + 2*k3l + k4l)/6
        xl_new  = xl + dt*xldot
        # components
        pl_new     = np.array([[xl_new[0,0], xl_new[1,0], xl_new[2,0]]]).T
        vl_new     = np.array([[xl_new[3,0], xl_new[4,0], xl_new[5,0]]]).T
        ql_new     = np.array([[xl_new[6,0], xl_new[7,0], xl_new[8,0], xl_new[9,0]]]).T
        ql_new     = ql_new/LA.norm(ql_new) # normalization
        wl_new     = np.array([[xl_new[10,0], xl_new[11,0], xl_new[12,0]]]).T
        Rl_new     = self.q_2_rotation(ql_new,1)
        # Y->Z->X rotation from {b} to {I}
        gamma_l   = np.arctan(Rl_new[2, 1]/Rl_new[1, 1])
        theta_l   = np.arctan(Rl_new[0, 2]/Rl_new[0, 0])
        psi_l     = np.arcsin(-Rl_new[0, 1])
        Euler_l_new = np.array([[gamma_l, theta_l, psi_l]]).T
        output = {"pl_new":pl_new,
                  "vl_new":vl_new,
                  "ql_new":ql_new,
                  "wl_new":wl_new,
                  "Euler_l_new":Euler_l_new
                 }
        return output

    
    def step_quadrotor(self, xi, ui, xl, ti, index, dt):
        # Quadrotor: 4-th order Runge-Kutta discrete-time model
        k1i     = self.dyn_i_fn(xi0=xi, ui0=ui, xl0=xl, ti0=ti, index0=index)['k1if'].full()
        k2i     = self.dyn_i_fn(xi0=xi+dt/2*k1i, ui0=ui, xl0=xl, ti0=ti, index0=index)['k1if'].full()
        k3i     = self.dyn_i_fn(xi0=xi+dt/2*k2i, ui0=ui, xl0=xl, ti0=ti, index0=index)['k1if'].full()
        k4i     = self.dyn_i_fn(xi0=xi+dt*k3i, ui0=ui, xl0=xl, ti0=ti, index0=index)['k1if'].full()
        xidot   = (k1i + 2*k2i + 2*k3i + k4i)/6
        xi_new  = xi + dt*xidot
        # components
        pi_new     = np.array([[xi_new[0,0], xi_new[1,0], xi_new[2,0]]]).T
        vi_new     = np.array([[xi_new[3,0], xi_new[4,0], xi_new[5,0]]]).T
        qi_new     = np.array([[xi_new[6,0], xi_new[7,0], xi_new[8,0], xi_new[9,0]]]).T
        qi_new     = qi_new/LA.norm(qi_new) # normalization
        wi_new     = np.array([[xi_new[10,0], xi_new[11,0], xi_new[12,0]]]).T
        Ri_new     = self.q_2_rotation(qi_new,1)
        # Y->Z->X rotation from {b} to {I}
        gamma_i   = np.arctan(Ri_new[2, 1]/Ri_new[1, 1])
        theta_i   = np.arctan(Ri_new[0, 2]/Ri_new[0, 0])
        psi_i     = np.arcsin(-Ri_new[0, 1])
        Euler_i_new = np.array([[gamma_i, theta_i, psi_i]]).T
        output = {"pi_new":pi_new,
                  "vi_new":vi_new,
                  "qi_new":qi_new,
                  "wi_new":wi_new,
                  "Euler_i_new":Euler_i_new
                 }
        return output
    

    """
    Predictor function for the L1-AC state predictor
    """
    def predictor_L1(self, z_hat, xi, ui, xl, ti, dm, dum, As, index, dt):
        # define discrete-time dynamics using 4-th order Runge-Kutta
        k1z    = self.dzhat_fn(zhat0=z_hat,xi0=xi,ui0=ui,xl0=xl,ti0=ti,dm0=dm,dum0=dum,As0=As,i0=index)['dz_hatf'].full()
        k2z    = self.dzhat_fn(zhat0=z_hat+dt/2*k1z,xi0=xi,ui0=ui,xl0=xl,ti0=ti,dm0=dm,dum0=dum,As0=As,i0=index)['dz_hatf'].full()
        k3z    = self.dzhat_fn(zhat0=z_hat+dt/2*k2z,xi0=xi,ui0=ui,xl0=xl,ti0=ti,dm0=dm,dum0=dum,As0=As,i0=index)['dz_hatf'].full()
        k4z    = self.dzhat_fn(zhat0=z_hat+dt*k3z,xi0=xi,ui0=ui,xl0=xl,ti0=ti,dm0=dm,dum0=dum,As0=As,i0=index)['dz_hatf'].full()
        dz_hat = (k1z + 2*k2z + 2*k3z + k4z)/6 
        zhat_new = z_hat + dt*dz_hat
        return zhat_new
    
    # polynomial trajectories
    def polytraj(self,coeff,time,time0):
        time_vec   = vertcat(1,
                             self.time-self.time0,
                             (self.time-self.time0)**2,
                             (self.time-self.time0)**3,
                             (self.time-self.time0)**4,
                             (self.time-self.time0)**5,
                             (self.time-self.time0)**6,
                             (self.time-self.time0)**7)
        polyp      = mtimes(self.polyc, time_vec)
        polyp_fn   = Function('ref_p',[self.polyc,self.time,self.time0],[polyp],['pc0','t0','ti0'],['ref_pf'])
        ref_p      = polyp_fn(pc0=coeff,t0=time,ti0=time0)['ref_pf'].full()
        polyv      = jacobian(polyp, self.time)
        polyv_fn   = Function('ref_v',[self.polyc,self.time,self.time0],[polyv],['pc0','t0','ti0'],['ref_vf'])
        ref_v      = polyv_fn(pc0=coeff,t0=time,ti0=time0)['ref_vf'].full()
        polya      = jacobian(polyv, self.time)
        polya_fn   = Function('ref_a',[self.polyc,self.time,self.time0],[polya],['pc0','t0','ti0'],['ref_af'])
        ref_a      = polya_fn(pc0=coeff,t0=time,ti0=time0)['ref_af'].full()
        polyj      = jacobian(polya, self.time)
        polyj_fn   = Function('ref_j',[self.polyc,self.time,self.time0],[polyj],['pc0','t0','ti0'],['ref_jf'])
        ref_j      = polyj_fn(pc0=coeff,t0=time,ti0=time0)['ref_jf'].full()
        polys      = jacobian(polyj, self.time)
        polys_fn   = Function('ref_s',[self.polyc,self.time,self.time0],[polys],['pc0','t0','ti0'],['ref_sf'])
        ref_s      = polys_fn(pc0=coeff,t0=time,ti0=time0)['ref_sf'].full()

        return ref_p, ref_v, ref_a, ref_j, ref_s
    
    def reference_circle(self, Coeffx_evaluation, Coeffy_evaluation, Coeffz_evaluation, time, t_switch): 
        if time <6.5+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_evaluation[0,:],time,t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_evaluation[0,:],time,t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_evaluation[0,:],time,t_switch)
        elif time >=6.5+t_switch and time <9.5+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_evaluation[1,:],time,6.5+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_evaluation[1,:],time,6.5+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_evaluation[1,:],time,6.5+t_switch)
        elif time >=9.5+t_switch and time <12.5+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_evaluation[2,:],time,9.5+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_evaluation[2,:],time,9.5+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_evaluation[2,:],time,9.5+t_switch)
        else:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_evaluation[3,:],time,12.5+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_evaluation[3,:],time,12.5+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_evaluation[3,:],time,12.5+t_switch)
        ref_p = np.reshape(np.vstack((ref_px, ref_py, ref_pz)), (3,1))
        ref_v = np.reshape(np.vstack((ref_vx, ref_vy, ref_vz)), (3,1))
        ref_a = np.reshape(np.vstack((ref_ax, ref_ay, ref_az)), (3,1))
        ref_j = np.reshape(np.vstack((ref_jx, ref_jy, ref_jz)), (3,1))
        ref_s = np.reshape(np.vstack((ref_sx, ref_sy, ref_sz)), (3,1))
        return ref_p, ref_v, ref_a, ref_j, ref_s
    
    def new_circle_load(self,coeffa,time):
        ref_an, ref_wn, ref_awn, ref_jx, ref_sx = self.polytraj(coeffa,time,0)
        ref_p = np.array([[self.rc*math.cos(ref_an), self.rc*math.sin(ref_an), self.hc]]).T
        ref_v = np.vstack((-self.rc * math.sin(ref_an) * ref_wn,self.rc * math.cos(ref_an) * ref_wn,0))
        ref_a = np.vstack((-self.rc*math.cos(ref_an)*ref_wn**2-self.rc*math.sin(ref_an)*ref_awn,
                           -self.rc*math.sin(ref_an)*ref_wn**2+self.rc*math.cos(ref_an)*ref_awn,
                           0.0))
        return ref_p, ref_v, ref_a
    
    def new_circle_quadrotor(self,coeffa,time,angle_t,index):
        pil   = np.array([[(self.rl+self.cl0*math.sin(angle_t))*math.cos(index*self.alpha),(self.rl+self.cl0*math.sin(angle_t))*math.sin(index*self.alpha),self.cl0*math.cos(angle_t)]]).T # relative position of the i-th quadrotor in the desired {Bl_d} which is parallel to {I}
        ref_p, ref_v, ref_a = self.new_circle_load(coeffa,time)
        ref_pi = ref_p + pil
        return ref_pi, ref_v, ref_a
    
    def minisnap_load_circle(self,Coeffx,Coeffy,Coeffz,time):
        t_switch = 0
        t1,t2,t3 = 5, 2.4, 2.4
        if time <t1:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx[0,:],time,t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy[0,:],time,t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz[0,:],time,t_switch)
        elif time>=t1 and time<(t1+t2):
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx[1,:],time,t1+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy[1,:],time,t1+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz[1,:],time,t1+t_switch)
        elif time>=(t1+t2) and time<(t1+t2+t3):
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx[2,:],time,t1+t2+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy[2,:],time,t1+t2+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz[2,:],time,t1+t2+t_switch)
        else:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx[3,:],time,t1+t2+t3+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy[3,:],time,t1+t2+t3+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz[3,:],time,t1+t2+t3+t_switch)
        ref_p = np.reshape(np.vstack((ref_px, ref_py, ref_pz)), (3,1))
        ref_v = np.reshape(np.vstack((ref_vx, ref_vy, ref_vz)), (3,1))
        ref_a = np.reshape(np.vstack((ref_ax, ref_ay, ref_az)), (3,1))
        # ref_j = np.reshape(np.vstack((ref_jx, ref_jy, ref_jz)), (3,1))
        # ref_s = np.reshape(np.vstack((ref_sx, ref_sy, ref_sz)), (3,1))
        return ref_p, ref_v, ref_a
    
    def minisnap_quadrotor_circle(self,Coeffx,Coeffy,Coeffz,time,angle_t,index):
        pil   = np.array([[(self.rl+self.cl0*math.sin(angle_t))*math.cos(index*self.alpha),(self.rl+self.cl0*math.sin(angle_t))*math.sin(index*self.alpha),self.cl0*math.cos(angle_t)]]).T # relative position of the i-th quadrotor in the desired {Bl_d} which is parallel to {I}
        ref_p, ref_v, ref_a = self.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time)
        ref_pi = ref_p + pil
        return ref_pi, ref_v, ref_a
    

    def minisnap_load_fig8(self,Coeffx,Coeffy,Coeffz,time):
        t_switch = 0
        t1,t2,t3,t4,t5,t6,t7 = 4,2,2,2,2,2,2
        if time<t1:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx[0,:],time,t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy[0,:],time,t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz[0,:],time,t_switch)
        elif time>=t1 and time<(t1+t2):
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx[1,:],time,t1+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy[1,:],time,t1+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz[1,:],time,t1+t_switch)
        elif time>=(t1+t2) and time<(t1+t2+t3):
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx[2,:],time,t1+t2+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy[2,:],time,t1+t2+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz[2,:],time,t1+t2+t_switch)
        elif time>=(t1+t2+t3) and time<(t1+t2+t3+t4):
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx[3,:],time,t1+t2+t3+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy[3,:],time,t1+t2+t3+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz[3,:],time,t1+t2+t3+t_switch)
        elif time>=(t1+t2+t3+t4) and time<(t1+t2+t3+t4+t5):
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx[4,:],time,t1+t2+t3+t4+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy[4,:],time,t1+t2+t3+t4+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz[4,:],time,t1+t2+t3+t4+t_switch)
        elif time>=(t1+t2+t3+t4+t5) and time<(t1+t2+t3+t4+t5+t6):
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx[5,:],time,t1+t2+t3+t4+t5+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy[5,:],time,t1+t2+t3+t4+t5+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz[5,:],time,t1+t2+t3+t4+t5+t_switch)
        elif time>=(t1+t2+t3+t4+t5+t6) and time<(t1+t2+t3+t4+t5+t6+t7):
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx[6,:],time,t1+t2+t3+t4+t5+t6+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy[6,:],time,t1+t2+t3+t4+t5+t6+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz[6,:],time,t1+t2+t3+t4+t5+t6+t_switch)
        else:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx[7,:],time,t1+t2+t3+t4+t5+t6+t7+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy[7,:],time,t1+t2+t3+t4+t5+t6+t7+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz[7,:],time,t1+t2+t3+t4+t5+t6+t7+t_switch)
        ref_p = np.reshape(np.vstack((ref_px, ref_py, ref_pz)), (3,1))
        ref_v = np.reshape(np.vstack((ref_vx, ref_vy, ref_vz)), (3,1))
        ref_a = np.reshape(np.vstack((ref_ax, ref_ay, ref_az)), (3,1))
        return ref_p, ref_v, ref_a
    
    def minisnap_quadrotor_fig8(self,Coeffx,Coeffy,Coeffz,time,angle_t,index):
        pil   = np.array([[(self.rl+self.cl0*math.sin(angle_t))*math.cos(index*self.alpha),(self.rl+self.cl0*math.sin(angle_t))*math.sin(index*self.alpha),self.cl0*math.cos(angle_t)]]).T # relative position of the i-th quadrotor in the desired {Bl_d} which is parallel to {I}
        ref_p, ref_v, ref_a = self.minisnap_load_fig8(Coeffx,Coeffy,Coeffz,time)
        ref_pi = ref_p + pil
        return ref_pi, ref_v, ref_a
    
    def hovering_load(self,h):
        ref_p = np.array([[0,0,h]]).T
        ref_v = np.zeros((3,1))
        ref_a = np.zeros((3,1))
        return ref_p, ref_v, ref_a
    
    def hovering_quadrotor(self,h,angle_t,index):
        pil   = np.array([[(self.rl+self.cl0*math.sin(angle_t))*math.cos(index*self.alpha),(self.rl+self.cl0*math.sin(angle_t))*math.sin(index*self.alpha),self.cl0*math.cos(angle_t)]]).T # relative position of the i-th quadrotor in the desired {Bl_d} which is parallel to {I}
        ref_p, ref_v, ref_a = self.hovering_load(h)
        ref_pi = ref_p + pil
        return ref_pi, ref_v, ref_a
    

    # below functions are for demo
    # get the position of the center and the four vertexes of the robot within a trajectory
    def get_quadrotor_position(self, wing_len, state_traj):

        # thrust_position in body frame
        r1 = vertcat(wing_len*0.5 / sqrt(2), wing_len*0.5 / sqrt(2), 0)
        r2 = vertcat(-wing_len*0.5 / sqrt(2), wing_len*0.5 / sqrt(2), 0)
        r3 = vertcat(-wing_len*0.5 / sqrt(2), -wing_len*0.5 / sqrt(2), 0)
        r4 = vertcat(wing_len*0.5 / sqrt(2), -wing_len*0.5 / sqrt(2), 0)

        # horizon
        horizon = np.size(state_traj, 1)
        position = np.zeros((15,horizon))
        for t in range(horizon):
            # position of COM
            rc = state_traj[0:3, t]
            # rotation matrix
            q = state_traj[6:10, t]
            CIB = self.q_2_rotation(q,1)
           
            # position of each rotor in inertial frame
            r1_pos = rc + mtimes(CIB, r1).full().flatten()
            r2_pos = rc + mtimes(CIB, r2).full().flatten()
            r3_pos = rc + mtimes(CIB, r3).full().flatten()
            r4_pos = rc + mtimes(CIB, r4).full().flatten()

            # store
            position[0:3,t] = rc
            position[3:6,t] = r1_pos
            position[6:9,t] = r2_pos
            position[9:12,t] = r3_pos
            position[12:15,t] = r4_pos

        return position
    
    def get_payload_position(self, load_len, state_traj):

        # thrust_position in body frame
        r1 = vertcat(load_len*0.5 , 0, 0)
        r2 = vertcat(0, -load_len*0.5 , 0)
        r3 = vertcat(-load_len*0.5 , 0, 0)
        r4 = vertcat(0, load_len*0.5, 0)

        # horizon
        horizon = np.size(state_traj, 1)
        position = np.zeros((15,horizon))
        for t in range(horizon):
            # position of COM
            rc = state_traj[0:3, t]
            # rotation matrix
            q = state_traj[6:10, t]
            CIB = self.q_2_rotation(q,1)
           
            # position of each rotor in inertial frame
            r1_pos = rc + mtimes(CIB, r1).full().flatten()
            r2_pos = rc + mtimes(CIB, r2).full().flatten()
            r3_pos = rc + mtimes(CIB, r3).full().flatten()
            r4_pos = rc + mtimes(CIB, r4).full().flatten()

            # store
            position[0:3,t] = rc
            position[3:6,t] = r1_pos
            position[6:9,t] = r2_pos
            position[9:12,t] = r3_pos
            position[12:15,t] = r4_pos

        return position


        






    

    



