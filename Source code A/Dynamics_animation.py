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
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        self.K      = cable_para[0] # stiffness
        self.ct     = cable_para[1] # damping coefficient
        self.cl0    = cable_para[2] # cable natural length
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
        self.rc    = 3 # radius of the circle
        self.Tc    = 20 # period of the circle
        self.hc    = 1  # height of the circle
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
        f_tc = self.K*(L-self.cl0) + self.ct*self.K*dL # computed tension magnitude
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
    
    def get_payload_position3(self, load_len, state_traj):

        # thrust_position in body frame
        # r1 = vertcat(load_len*0.5 , 0, 0)
        # r2 = vertcat(0, -load_len*0.5 , 0)
        # r3 = vertcat(-load_len*0.5 , 0, 0)
        # r4 = vertcat(0, load_len*0.5, 0)

        r1 = vertcat(load_len*math.cos(0*self.alpha),load_len*math.sin(0*self.alpha),0)
        r2 = vertcat(load_len*math.cos(1*self.alpha),load_len*math.sin(1*self.alpha),0)
        r3 = vertcat(load_len*math.cos(2*self.alpha),load_len*math.sin(2*self.alpha),0)
        r4 = vertcat(load_len*math.cos(3*self.alpha),load_len*math.sin(3*self.alpha),0)
        r5 = vertcat(load_len*math.cos(4*self.alpha),load_len*math.sin(4*self.alpha),0)
        r6 = vertcat(load_len*math.cos(5*self.alpha),load_len*math.sin(5*self.alpha),0)

        # horizon
        horizon = np.size(state_traj, 1)
        position = np.zeros((3*7,horizon))
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
            r5_pos = rc + mtimes(CIB, r5).full().flatten()
            r6_pos = rc + mtimes(CIB, r6).full().flatten()
            # store
            position[0:3,t] = rc
            position[3:6,t] = r1_pos
            position[6:9,t] = r2_pos
            position[9:12,t] = r3_pos
            # position[12:15,t] = r4_pos
            # position[15:18,t] = r5_pos
            # position[18:21,t] = r6_pos

        return position
    
    
    def get_payload_position6(self, load_len, state_traj):

        # thrust_position in body frame
        # r1 = vertcat(load_len*0.5 , 0, 0)
        # r2 = vertcat(0, -load_len*0.5 , 0)
        # r3 = vertcat(-load_len*0.5 , 0, 0)
        # r4 = vertcat(0, load_len*0.5, 0)

        r1 = vertcat(load_len*math.cos(0*self.alpha),load_len*math.sin(0*self.alpha),0)
        r2 = vertcat(load_len*math.cos(1*self.alpha),load_len*math.sin(1*self.alpha),0)
        r3 = vertcat(load_len*math.cos(2*self.alpha),load_len*math.sin(2*self.alpha),0)
        r4 = vertcat(load_len*math.cos(3*self.alpha),load_len*math.sin(3*self.alpha),0)
        r5 = vertcat(load_len*math.cos(4*self.alpha),load_len*math.sin(4*self.alpha),0)
        r6 = vertcat(load_len*math.cos(5*self.alpha),load_len*math.sin(5*self.alpha),0)

        # horizon
        horizon = np.size(state_traj, 1)
        position = np.zeros((3*7,horizon))
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
            r5_pos = rc + mtimes(CIB, r5).full().flatten()
            r6_pos = rc + mtimes(CIB, r6).full().flatten()
            # store
            position[0:3,t] = rc
            position[3:6,t] = r1_pos
            position[6:9,t] = r2_pos
            position[9:12,t] = r3_pos
            position[12:15,t] = r4_pos
            position[15:18,t] = r5_pos
            position[18:21,t] = r6_pos

        return position
    

    def set_axes_equal(self, ax):

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def play_animation(self, k, wing_len, load_len, STATE_0, STATE_1, STATE_2, STATE_3, STATE_4, STATE_5, STATE_l, STATE_l0,STATE_l1, STATE_l9, position_ref, dt ,save_option=0):
        cm_2_inch = 2.54
        font1 = {'family':'Times New Roman',
         'weight':'normal',
         'style':'normal', 'size':5}
        fig=plt.figure(1, figsize=(10/cm_2_inch,10/cm_2_inch),dpi=600)
        ax = plt.axes(projection="3d")
        ax.xaxis._axinfo["grid"].update({"linewidth":0.5})
        ax.yaxis._axinfo["grid"].update({"linewidth":0.5})
        ax.zaxis._axinfo["grid"].update({"linewidth":0.5})
        ax.xaxis._axinfo["grid"].update({"linestyle":'--'})
        ax.yaxis._axinfo["grid"].update({"linestyle":'--'})
        ax.zaxis._axinfo["grid"].update({"linestyle":'--'})
        ax.tick_params(axis='x',which='major',pad=-6, length=1)
        ax.set_xlabel('x [m]', labelpad=-14, **font1)
        ax.tick_params(axis='y',which='major',pad=-6, length=1)
        ax.set_ylabel('y [m]', labelpad=-14, **font1)
        ax.tick_params(axis='z',which='major',pad=-5, length=1)
        ax.set_zlabel('z [m]', labelpad=-15, **font1)
        ax.set_xlim(-4,4)
        ax.set_ylim(-4,4)
        ax.set_zlim(-5,5)
        self.set_axes_equal(ax)
        # ax.set_zticks([0,2,4,6])
        for t in ax.xaxis.get_major_ticks(): 
            t.label.set_font('Times New Roman') 
            t.label.set_fontsize(6)
        for t in ax.yaxis.get_major_ticks(): 
            t.label.set_font('Times New Roman') 
            t.label.set_fontsize(6)
        for t in ax.zaxis.get_major_ticks(): 
            t.label.set_font('Times New Roman') 
            t.label.set_fontsize(6)
        for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
            axis.line.set_linewidth(0.5)
        # ax.view_init(15,-160)
        # ax.set_zlim(-0.5, 8)
        # ax.set_ylim(-6, 6)
        # ax.set_xlim(-4, 4)

        # time label
        time_template = 'Time: %.2f [s]'
        time_text = ax.text2D(0.5, 0.9, "Time", transform=ax.transAxes,**font1)
       
        # data
        position_0 = self.get_quadrotor_position(wing_len,STATE_0)
        position_1 = self.get_quadrotor_position(wing_len,STATE_1)
        position_2 = self.get_quadrotor_position(wing_len,STATE_2)
        position_3 = self.get_quadrotor_position(wing_len,STATE_3)
        position_4 = self.get_quadrotor_position(wing_len,STATE_4)
        position_5 = self.get_quadrotor_position(wing_len,STATE_5)
        position_l = self.get_payload_position6(load_len,STATE_l)
        position_l0 = self.get_payload_position6(load_len,STATE_l0)
        position_l1 = self.get_payload_position6(load_len,STATE_l1)
        position_l9 = self.get_payload_position6(load_len,STATE_l9)
        sim_horizon = np.size(position_l, 1)

        # animation
        line_traj_ref, = ax.plot(position_ref[0, :1], position_ref[1, :1], position_ref[2, :1], linestyle='--', linewidth=0.5, color='orange')
        if k==0:
            line_traj_l, = ax.plot(position_l[0,:1], position_l[1,:1], position_l[2,:1],linewidth=1,color='red')
        elif k==1:
            line_traj_l, = ax.plot(position_l[0,:1], position_l[1,:1], position_l[2,:1],linewidth=1,color='pink')
            line_traj_l0, = ax.plot(position_l0[0,:1], position_l0[1,:1], position_l0[2,:1],linewidth=1,color='red')
        elif k==4:
            line_traj_l, = ax.plot(position_l[0,:1], position_l[1,:1], position_l[2,:1],linewidth=1,color='blue')
            line_traj_l1, = ax.plot(position_l1[0,:1], position_l1[1,:1], position_l1[2,:1],linewidth=1,color='pink')
            line_traj_l0, = ax.plot(position_l0[0,:1], position_l0[1,:1], position_l0[2,:1],linewidth=1,color='red')
        else:
            line_traj_l, = ax.plot(position_l[0,:1], position_l[1,:1], position_l[2,:1],linewidth=1,color='green')
            line_traj_l9, = ax.plot(position_l9[0,:1], position_l9[1,:1], position_l9[2,:1],linewidth=1,color='blue')
            line_traj_l1, = ax.plot(position_l1[0,:1], position_l1[1,:1], position_l1[2,:1],linewidth=1,color='pink')
            line_traj_l0, = ax.plot(position_l0[0,:1], position_l0[1,:1], position_l0[2,:1],linewidth=1,color='red')
            
        
        c_x0, c_y0, c_z0 = position_0[0:3,0]
        r1_x0, r1_y0, r1_z0 = position_0[3:6,0]
        r2_x0, r2_y0, r2_z0 = position_0[6:9,0]
        r3_x0, r3_y0, r3_z0 = position_0[9:12,0]
        r4_x0, r4_y0, r4_z0 = position_0[12:15,0]
        line_arm1_0, = ax.plot([c_x0, r1_x0], [c_y0, r1_y0], [c_z0, r1_z0], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm2_0, = ax.plot([c_x0, r2_x0], [c_y0, r2_y0], [c_z0, r2_z0], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm3_0, = ax.plot([c_x0, r3_x0], [c_y0, r3_y0], [c_z0, r3_z0], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm4_0, = ax.plot([c_x0, r4_x0], [c_y0, r4_y0], [c_z0, r4_z0], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)

        c_x1, c_y1, c_z1 = position_1[0:3,0]
        r1_x1, r1_y1, r1_z1 = position_1[3:6,0]
        r2_x1, r2_y1, r2_z1 = position_1[6:9,0]
        r3_x1, r3_y1, r3_z1 = position_1[9:12,0]
        r4_x1, r4_y1, r4_z1 = position_1[12:15,0]
        line_arm1_1, = ax.plot([c_x1, r1_x1], [c_y1, r1_y1], [c_z1, r1_z1], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm2_1, = ax.plot([c_x1, r2_x1], [c_y1, r2_y1], [c_z1, r2_z1], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm3_1, = ax.plot([c_x1, r3_x1], [c_y1, r3_y1], [c_z1, r3_z1], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm4_1, = ax.plot([c_x1, r4_x1], [c_y1, r4_y1], [c_z1, r4_z1], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)

        c_x2, c_y2, c_z2 = position_2[0:3,0]
        r1_x2, r1_y2, r1_z2 = position_2[3:6,0]
        r2_x2, r2_y2, r2_z2 = position_2[6:9,0]
        r3_x2, r3_y2, r3_z2 = position_2[9:12,0]
        r4_x2, r4_y2, r4_z2 = position_2[12:15,0]
        line_arm1_2, = ax.plot([c_x2, r1_x2], [c_y2, r1_y2], [c_z2, r1_z2], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm2_2, = ax.plot([c_x2, r2_x2], [c_y2, r2_y2], [c_z2, r2_z2], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm3_2, = ax.plot([c_x2, r3_x2], [c_y2, r3_y2], [c_z2, r3_z2], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm4_2, = ax.plot([c_x2, r4_x2], [c_y2, r4_y2], [c_z2, r4_z2], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)

        c_x3, c_y3, c_z3 = position_3[0:3,0]
        r1_x3, r1_y3, r1_z3 = position_3[3:6,0]
        r2_x3, r2_y3, r2_z3 = position_3[6:9,0]
        r3_x3, r3_y3, r3_z3 = position_3[9:12,0]
        r4_x3, r4_y3, r4_z3 = position_3[12:15,0]
        line_arm1_3, = ax.plot([c_x3, r1_x3], [c_y3, r1_y3], [c_z3, r1_z3], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm2_3, = ax.plot([c_x3, r2_x3], [c_y3, r2_y3], [c_z3, r2_z3], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm3_3, = ax.plot([c_x3, r3_x3], [c_y3, r3_y3], [c_z3, r3_z3], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm4_3, = ax.plot([c_x3, r4_x3], [c_y3, r4_y3], [c_z3, r4_z3], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)

        c_x4, c_y4, c_z4 = position_4[0:3,0]
        r1_x4, r1_y4, r1_z4 = position_4[3:6,0]
        r2_x4, r2_y4, r2_z4 = position_4[6:9,0]
        r3_x4, r3_y4, r3_z4 = position_4[9:12,0]
        r4_x4, r4_y4, r4_z4 = position_4[12:15,0]
        line_arm1_4, = ax.plot([c_x4, r1_x4], [c_y4, r1_y4], [c_z4, r1_z4], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm2_4, = ax.plot([c_x4, r2_x4], [c_y4, r2_y4], [c_z4, r2_z4], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm3_4, = ax.plot([c_x4, r3_x4], [c_y4, r3_y4], [c_z4, r3_z4], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm4_4, = ax.plot([c_x4, r4_x4], [c_y4, r4_y4], [c_z4, r4_z4], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)

        c_x5, c_y5, c_z5 = position_5[0:3,0]
        r1_x5, r1_y5, r1_z5 = position_5[3:6,0]
        r2_x5, r2_y5, r2_z5 = position_5[6:9,0]
        r3_x5, r3_y5, r3_z5 = position_5[9:12,0]
        r4_x5, r4_y5, r4_z5 = position_5[12:15,0]
        line_arm1_5, = ax.plot([c_x5, r1_x5], [c_y5, r1_y5], [c_z5, r1_z5], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm2_5, = ax.plot([c_x5, r2_x5], [c_y5, r2_y5], [c_z5, r2_z5], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm3_5, = ax.plot([c_x5, r3_x5], [c_y5, r3_y5], [c_z5, r3_z5], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)
        line_arm4_5, = ax.plot([c_x5, r4_x5], [c_y5, r4_y5], [c_z5, r4_z5], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=10)

        c_xl, c_yl, c_zl = position_l[0:3,0]
        r1_xl, r1_yl, r1_zl = position_l[3:6,0]
        r2_xl, r2_yl, r2_zl = position_l[6:9,0]
        r3_xl, r3_yl, r3_zl = position_l[9:12,0]
        r4_xl, r4_yl, r4_zl = position_l[12:15,0]
        r5_xl, r5_yl, r5_zl = position_l[15:18,0]
        r6_xl, r6_yl, r6_zl = position_l[18:21,0]

        line_loadaim1, =ax.plot([c_xl, r1_xl], [c_yl, r1_yl], [c_zl, r1_zl], linewidth=0.5, color='black')
        line_loadaim2, =ax.plot([c_xl, r2_xl], [c_yl, r2_yl], [c_zl, r2_zl], linewidth=0.5, color='black')
        line_loadaim3, =ax.plot([c_xl, r3_xl], [c_yl, r3_yl], [c_zl, r3_zl], linewidth=0.5, color='black')
        line_loadaim4, =ax.plot([c_xl, r4_xl], [c_yl, r4_yl], [c_zl, r4_zl], linewidth=0.5, color='black')
        line_loadaim5, =ax.plot([c_xl, r5_xl], [c_yl, r5_yl], [c_zl, r5_zl], linewidth=0.5, color='black')
        line_loadaim6, =ax.plot([c_xl, r6_xl], [c_yl, r6_yl], [c_zl, r6_zl], linewidth=0.5, color='black')
        line_loadaim12, =ax.plot([r1_xl, r2_xl], [r1_yl, r2_yl], [r1_zl, r2_zl], linewidth=1, color='orange')
        line_loadaim23, =ax.plot([r2_xl, r3_xl], [r2_yl, r3_yl], [r2_zl, r3_zl], linewidth=1, color='orange')
        line_loadaim34, =ax.plot([r3_xl, r4_xl], [r3_yl, r4_yl], [r3_zl, r4_zl], linewidth=1, color='orange')
        line_loadaim45, =ax.plot([r4_xl, r5_xl], [r4_yl, r5_yl], [r4_zl, r5_zl], linewidth=1, color='orange')
        line_loadaim56, =ax.plot([r5_xl, r6_xl], [r5_yl, r6_yl], [r5_zl, r6_zl], linewidth=1, color='orange')
        line_loadaim61, =ax.plot([r6_xl, r1_xl], [r6_yl, r1_yl], [r6_zl, r1_zl], linewidth=1, color='orange')

        tether1, =ax.plot([c_x0, r1_xl], [c_y0, r1_yl], [c_z0, r1_zl], linewidth=0.3, color='black',zorder=10)
        tether2, =ax.plot([c_x1, r2_xl], [c_y1, r2_yl], [c_z1, r2_zl], linewidth=0.3, color='black',zorder=10)
        tether3, =ax.plot([c_x2, r3_xl], [c_y2, r3_yl], [c_z2, r3_zl], linewidth=0.3, color='black',zorder=10)
        tether4, =ax.plot([c_x3, r4_xl], [c_y3, r4_yl], [c_z3, r4_zl], linewidth=0.3, color='black',zorder=10)
        tether5, =ax.plot([c_x4, r5_xl], [c_y4, r5_yl], [c_z4, r5_zl], linewidth=0.3, color='black',zorder=10)
        tether6, =ax.plot([c_x5, r6_xl], [c_y5, r6_yl], [c_z5, r6_zl], linewidth=0.3, color='black',zorder=10)
        self.set_axes_equal(ax)
        # customize
        if position_ref is not None:
            if k==0:
                leg=plt.legend(['Desired', 'Episode:'+str(k)],prop=font1,loc=(-0.15,0.85),labelspacing=0.1,handlelength=1)
                leg.get_frame().set_linewidth(0.5)
            elif k==1:
                leg=plt.legend(['Desired', 'Episode:'+str(k),'Episode:0'],prop=font1,loc=(-0.15,0.8),labelspacing=0.1,handlelength=1)
                leg.get_frame().set_linewidth(0.5)
            elif k==4:
                leg=plt.legend(['Desired', 'Episode:'+str(k),'Episode:1','Episode:0'],prop=font1,loc=(-0.15,0.75),labelspacing=0.1,handlelength=1)
                leg.get_frame().set_linewidth(0.5)
            else:
                leg=plt.legend(['Desired', 'Episode:'+str(k),'Episode:4','Episode:1','Episode:0'],prop=font1,loc=(-0.15,0.7),labelspacing=0.1,handlelength=1)
                leg.get_frame().set_linewidth(0.5)

        def update_traj(num):
            # customize
            time_text.set_text(time_template % (num * dt))
            line_traj_l.set_data_3d([position_l[0,:num], position_l[1,:num],position_l[2,:num]])

            
            # uav
            c_x0, c_y0, c_z0 = position_0[0:3,num]
            r1_x0, r1_y0, r1_z0 = position_0[3:6,num]
            r2_x0, r2_y0, r2_z0 = position_0[6:9,num]
            r3_x0, r3_y0, r3_z0 = position_0[9:12,num]
            r4_x0, r4_y0, r4_z0 = position_0[12:15,num]
            line_arm1_0.set_data_3d([c_x0, r1_x0], [c_y0, r1_y0],[c_z0, r1_z0])
            line_arm2_0.set_data_3d([c_x0, r2_x0], [c_y0, r2_y0],[c_z0, r2_z0])
            line_arm3_0.set_data_3d([c_x0, r3_x0], [c_y0, r3_y0],[c_z0, r3_z0])
            line_arm4_0.set_data_3d([c_x0, r4_x0], [c_y0, r4_y0],[c_z0, r4_z0])

            c_x1, c_y1, c_z1 = position_1[0:3,num]
            r1_x1, r1_y1, r1_z1 = position_1[3:6,num]
            r2_x1, r2_y1, r2_z1 = position_1[6:9,num]
            r3_x1, r3_y1, r3_z1 = position_1[9:12,num]
            r4_x1, r4_y1, r4_z1 = position_1[12:15,num]
            line_arm1_1.set_data_3d([c_x1, r1_x1], [c_y1, r1_y1],[c_z1, r1_z1])
            line_arm2_1.set_data_3d([c_x1, r2_x1], [c_y1, r2_y1],[c_z1, r2_z1])
            line_arm3_1.set_data_3d([c_x1, r3_x1], [c_y1, r3_y1],[c_z1, r3_z1])
            line_arm4_1.set_data_3d([c_x1, r4_x1], [c_y1, r4_y1],[c_z1, r4_z1])

            c_x2, c_y2, c_z2 = position_2[0:3,num]
            r1_x2, r1_y2, r1_z2 = position_2[3:6,num]
            r2_x2, r2_y2, r2_z2 = position_2[6:9,num]
            r3_x2, r3_y2, r3_z2 = position_2[9:12,num]
            r4_x2, r4_y2, r4_z2 = position_2[12:15,num]
            line_arm1_2.set_data_3d([c_x2, r1_x2], [c_y2, r1_y2],[c_z2, r1_z2])
            line_arm2_2.set_data_3d([c_x2, r2_x2], [c_y2, r2_y2],[c_z2, r2_z2])
            line_arm3_2.set_data_3d([c_x2, r3_x2], [c_y2, r3_y2],[c_z2, r3_z2])
            line_arm4_2.set_data_3d([c_x2, r4_x2], [c_y2, r4_y2],[c_z2, r4_z2])

            c_x3, c_y3, c_z3 = position_3[0:3,num]
            r1_x3, r1_y3, r1_z3 = position_3[3:6,num]
            r2_x3, r2_y3, r2_z3 = position_3[6:9,num]
            r3_x3, r3_y3, r3_z3 = position_3[9:12,num]
            r4_x3, r4_y3, r4_z3 = position_3[12:15,num]
            line_arm1_3.set_data_3d([c_x3, r1_x3], [c_y3, r1_y3],[c_z3, r1_z3])
            line_arm2_3.set_data_3d([c_x3, r2_x3], [c_y3, r2_y3],[c_z3, r2_z3])
            line_arm3_3.set_data_3d([c_x3, r3_x3], [c_y3, r3_y3],[c_z3, r3_z3])
            line_arm4_3.set_data_3d([c_x3, r4_x3], [c_y3, r4_y3],[c_z3, r4_z3])

            c_x4, c_y4, c_z4 = position_4[0:3,num]
            r1_x4, r1_y4, r1_z4 = position_4[3:6,num]
            r2_x4, r2_y4, r2_z4 = position_4[6:9,num]
            r3_x4, r3_y4, r3_z4 = position_4[9:12,num]
            r4_x4, r4_y4, r4_z4 = position_4[12:15,num]
            line_arm1_4.set_data_3d([c_x4, r1_x4], [c_y4, r1_y4],[c_z4, r1_z4])
            line_arm2_4.set_data_3d([c_x4, r2_x4], [c_y4, r2_y4],[c_z4, r2_z4])
            line_arm3_4.set_data_3d([c_x4, r3_x4], [c_y4, r3_y4],[c_z4, r3_z4])
            line_arm4_4.set_data_3d([c_x4, r4_x4], [c_y4, r4_y4],[c_z4, r4_z4])

            c_x5, c_y5, c_z5 = position_5[0:3,num]
            r1_x5, r1_y5, r1_z5 = position_5[3:6,num]
            r2_x5, r2_y5, r2_z5 = position_5[6:9,num]
            r3_x5, r3_y5, r3_z5 = position_5[9:12,num]
            r4_x5, r4_y5, r4_z5 = position_5[12:15,num]
            line_arm1_5.set_data_3d([c_x5, r1_x5], [c_y5, r1_y5],[c_z5, r1_z5])
            line_arm2_5.set_data_3d([c_x5, r2_x5], [c_y5, r2_y5],[c_z5, r2_z5])
            line_arm3_5.set_data_3d([c_x5, r3_x5], [c_y5, r3_y5],[c_z5, r3_z5])
            line_arm4_5.set_data_3d([c_x5, r4_x5], [c_y5, r4_y5],[c_z5, r4_z5])

            c_xl, c_yl, c_zl = position_l[0:3,num]
            r1_xl, r1_yl, r1_zl = position_l[3:6,num]
            r2_xl, r2_yl, r2_zl = position_l[6:9,num]
            r3_xl, r3_yl, r3_zl = position_l[9:12,num]
            r4_xl, r4_yl, r4_zl = position_l[12:15,num]
            r5_xl, r5_yl, r5_zl = position_l[15:18,num]
            r6_xl, r6_yl, r6_zl = position_l[18:21,num]
            
            line_loadaim1.set_data_3d([c_xl, r1_xl], [c_yl, r1_yl], [c_zl, r1_zl])
            line_loadaim2.set_data_3d([c_xl, r2_xl], [c_yl, r2_yl], [c_zl, r2_zl])
            line_loadaim3.set_data_3d([c_xl, r3_xl], [c_yl, r3_yl], [c_zl, r3_zl])
            line_loadaim4.set_data_3d([c_xl, r4_xl], [c_yl, r4_yl], [c_zl, r4_zl])
            line_loadaim5.set_data_3d([c_xl, r5_xl], [c_yl, r5_yl], [c_zl, r5_zl])
            line_loadaim6.set_data_3d([c_xl, r6_xl], [c_yl, r6_yl], [c_zl, r6_zl])

            line_loadaim12.set_data_3d([r1_xl, r2_xl], [r1_yl, r2_yl], [r1_zl, r2_zl])
            line_loadaim23.set_data_3d([r2_xl, r3_xl], [r2_yl, r3_yl], [r2_zl, r3_zl])
            line_loadaim34.set_data_3d([r3_xl, r4_xl], [r3_yl, r4_yl], [r3_zl, r4_zl])
            line_loadaim45.set_data_3d([r4_xl, r5_xl], [r4_yl, r5_yl], [r4_zl, r5_zl])
            line_loadaim56.set_data_3d([r5_xl, r6_xl], [r5_yl, r6_yl], [r5_zl, r6_zl])
            line_loadaim61.set_data_3d([r6_xl, r1_xl], [r6_yl, r1_yl], [r6_zl, r1_zl])

            tether1.set_data_3d([c_x0, r1_xl], [c_y0, r1_yl], [c_z0, r1_zl])
            tether2.set_data_3d([c_x1, r2_xl], [c_y1, r2_yl], [c_z1, r2_zl])
            tether3.set_data_3d([c_x2, r3_xl], [c_y2, r3_yl], [c_z2, r3_zl])
            tether4.set_data_3d([c_x3, r4_xl], [c_y3, r4_yl], [c_z3, r4_zl])
            tether5.set_data_3d([c_x4, r5_xl], [c_y4, r5_yl], [c_z4, r5_zl])
            tether6.set_data_3d([c_x5, r6_xl], [c_y5, r6_yl], [c_z5, r6_zl])

            # trajectory ref
            num=sim_horizon-1
            line_traj_ref.set_data_3d(position_ref[0,:num], position_ref[1,:num],position_ref[2,:num])
            if k==1:
                line_traj_l0.set_data_3d([position_l0[0,:num], position_l0[1,:num],position_l0[2,:num]])
            if k==4:
                line_traj_l1.set_data_3d([position_l1[0,:num], position_l1[1,:num],position_l1[2,:num]])
                line_traj_l0.set_data_3d([position_l0[0,:num], position_l0[1,:num],position_l0[2,:num]])
            if k==18 or k==25:
                line_traj_l9.set_data_3d([position_l9[0,:num], position_l9[1,:num],position_l9[2,:num]])
                line_traj_l1.set_data_3d([position_l1[0,:num], position_l1[1,:num],position_l1[2,:num]])
                line_traj_l0.set_data_3d([position_l0[0,:num], position_l0[1,:num],position_l0[2,:num]])
            self.set_axes_equal(ax)
            return line_traj_l,line_arm1_0, line_arm2_0, line_arm3_0, line_arm4_0, \
                   line_arm1_1, line_arm2_1, line_arm3_1, line_arm4_1, \
                   line_arm1_2, line_arm2_2, line_arm3_2, line_arm4_2, \
                   line_arm1_3, line_arm2_3, line_arm3_3, line_arm4_3, \
                   line_arm1_4, line_arm2_4, line_arm3_4, line_arm4_4, \
                   line_arm1_5, line_arm2_5, line_arm3_5, line_arm4_5, \
                   line_loadaim1, line_loadaim2, line_loadaim3, line_loadaim4, line_loadaim5, line_loadaim6, \
                   line_loadaim12, line_loadaim23, line_loadaim34, line_loadaim45, line_loadaim56, line_loadaim61, \
                   tether1, tether2, tether3, tether4, tether5, tether6, \
                   line_traj_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=dt*3000, blit=True)

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('training' + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()
    

    def play_animation_ref(self, k, wing_len, load_len, STATE_0, STATE_1, STATE_2, STATE_3, STATE_4, STATE_5, STATE_l, dt ,save_option=0):
        cm_2_inch = 2.54
        font1 = {'family':'Times New Roman',
         'weight':'normal',
         'style':'normal', 'size':5}
        fig=plt.figure(1, figsize=(10/cm_2_inch,10/cm_2_inch),dpi=600)
        ax = plt.axes(projection="3d")
        ax.xaxis._axinfo["grid"].update({"linewidth":0.5})
        ax.yaxis._axinfo["grid"].update({"linewidth":0.5})
        ax.zaxis._axinfo["grid"].update({"linewidth":0.5})
        ax.xaxis._axinfo["grid"].update({"linestyle":'--'})
        ax.yaxis._axinfo["grid"].update({"linestyle":'--'})
        ax.zaxis._axinfo["grid"].update({"linestyle":'--'})
        ax.tick_params(axis='x',which='major',pad=-6, length=1)
        ax.set_xlabel('x [m]', labelpad=-13, **font1)
        ax.tick_params(axis='y',which='major',pad=-6, length=1)
        ax.set_ylabel('y [m]', labelpad=-13, **font1)
        ax.tick_params(axis='z',which='major',pad=-5, length=1)
        ax.set_zlabel('z [m]', labelpad=-15, **font1)
        # ax.set_xlim(-4,4)
        # ax.set_ylim(-4,4)
        # ax.set_zlim(-5,5)
        self.set_axes_equal(ax)
        # ax.set_zticks([0,2,4,6])
        for t in ax.xaxis.get_major_ticks(): 
            t.label.set_font('Times New Roman') 
            t.label.set_fontsize(6)
        for t in ax.yaxis.get_major_ticks(): 
            t.label.set_font('Times New Roman') 
            t.label.set_fontsize(6)
        for t in ax.zaxis.get_major_ticks(): 
            t.label.set_font('Times New Roman') 
            t.label.set_fontsize(6)
        for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
            axis.line.set_linewidth(0.5)
        # ax.view_init(20,-70)
        ax.view_init(18,15)
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_zlim(-3,7)

        # time label
        time_template = 'Time: %.2f [s]'
        time_text = ax.text2D(0.5, 0.9, "Time", transform=ax.transAxes,**font1)
       
        # data
        position_0 = self.get_quadrotor_position(wing_len,STATE_0)
        position_1 = self.get_quadrotor_position(wing_len,STATE_1)
        position_2 = self.get_quadrotor_position(wing_len,STATE_2)
        position_3 = self.get_quadrotor_position(wing_len,STATE_3)
        position_4 = self.get_quadrotor_position(wing_len,STATE_4)
        position_5 = self.get_quadrotor_position(wing_len,STATE_5)
        position_l = self.get_payload_position6(load_len,STATE_l)
        # position_l0 = self.get_payload_position6(load_len,STATE_l0)
        # position_l1 = self.get_payload_position6(load_len,STATE_l1)
        sim_horizon = np.size(position_l, 1)

        # animation
        # line_traj_ref, = ax.plot(position_ref[0, :1], position_ref[1, :1], position_ref[2, :1], linestyle='--', linewidth=0.5, color='orange')
        # if k==0:
        #     line_traj_l, = ax.plot(position_l[0,:1], position_l[1,:1], position_l[2,:1],linewidth=1,color='blue',zorder=1+0.3*k)
        # elif k==1:
        #     line_traj_l, = ax.plot(position_l[0,:1], position_l[1,:1], position_l[2,:1],linewidth=1,color='blue',zorder=1+0.3*k)
        #     # line_traj_l0, = ax.plot(position_l0[0,:1], position_l0[1,:1], position_l0[2,:1],linewidth=1,color='red')
        # else:
        if k ==1:
            line_traj_l, = ax.plot(position_l[0,:1], position_l[1,:1], position_l[2,:1],linewidth=1,color='green',zorder=1+1.5*k)
        if k==0:
            line_traj_l, = ax.plot(position_l[0,:1], position_l[1,:1], position_l[2,:1],linewidth=1,color='blue',zorder=1+0.3*k)
            # line_traj_l1, = ax.plot(position_l1[0,:1], position_l1[1,:1], position_l1[2,:1],linewidth=1,color='green')
            # line_traj_l0, = ax.plot(position_l0[0,:1], position_l0[1,:1], position_l0[2,:1],linewidth=1,color='red')
            
        #gate 1
        g1_point1 = np.array([3.3,-1,1.95])
        g1_point2 = np.array([6.3,-1,1.95])
        g1_point3 = np.array([3.3,-1,3.57])
        g1_point4 = np.array([6.3,-1,3.57])
        g1_p1x, g1_p1y, g1_p1z = g1_point1[0:3]
        g1_p2x, g1_p2y, g1_p2z = g1_point2[0:3]
        g1_p3x, g1_p3y, g1_p3z = g1_point3[0:3]
        g1_p4x, g1_p4y, g1_p4z = g1_point4[0:3]
        line_slotl1, = ax.plot([g1_p1x,g1_p2x],[g1_p1y,g1_p2y],[g1_p1z,g1_p2z],linewidth=1.5,color='red',zorder=2)
        line_sloth1, = ax.plot([g1_p3x,g1_p4x],[g1_p3y,g1_p4y],[g1_p3z,g1_p4z],linewidth=1.5,color='red',zorder=10)

        #gate 2
        g2_point1 = np.array([-3.3,1,1.95])
        g2_point2 = np.array([-6.3,1,1.95])
        g2_point3 = np.array([-3.3,1,3.5])
        g2_point4 = np.array([-6.3,1,3.5])
        g2_p1x, g2_p1y, g2_p1z = g2_point1[0:3]
        g2_p2x, g2_p2y, g2_p2z = g2_point2[0:3]
        g2_p3x, g2_p3y, g2_p3z = g2_point3[0:3]
        g2_p4x, g2_p4y, g2_p4z = g2_point4[0:3]
        line_slotl2, = ax.plot([g2_p1x,g2_p2x],[g2_p1y,g2_p2y],[g2_p1z,g2_p2z],linewidth=1.5,color='red',zorder=2)
        line_sloth2, = ax.plot([g2_p3x,g2_p4x],[g2_p3y,g2_p4y],[g2_p3z,g2_p4z],linewidth=1.5,color='red',zorder=10)

        c_x0, c_y0, c_z0 = position_0[0:3,0]
        r1_x0, r1_y0, r1_z0 = position_0[3:6,0]
        r2_x0, r2_y0, r2_z0 = position_0[6:9,0]
        r3_x0, r3_y0, r3_z0 = position_0[9:12,0]
        r4_x0, r4_y0, r4_z0 = position_0[12:15,0]
        line_arm1_0, = ax.plot([c_x0, r1_x0], [c_y0, r1_y0], [c_z0, r1_z0], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm2_0, = ax.plot([c_x0, r2_x0], [c_y0, r2_y0], [c_z0, r2_z0], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm3_0, = ax.plot([c_x0, r3_x0], [c_y0, r3_y0], [c_z0, r3_z0], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm4_0, = ax.plot([c_x0, r4_x0], [c_y0, r4_y0], [c_z0, r4_z0], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)

        c_x1, c_y1, c_z1 = position_1[0:3,0]
        r1_x1, r1_y1, r1_z1 = position_1[3:6,0]
        r2_x1, r2_y1, r2_z1 = position_1[6:9,0]
        r3_x1, r3_y1, r3_z1 = position_1[9:12,0]
        r4_x1, r4_y1, r4_z1 = position_1[12:15,0]
        line_arm1_1, = ax.plot([c_x1, r1_x1], [c_y1, r1_y1], [c_z1, r1_z1], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm2_1, = ax.plot([c_x1, r2_x1], [c_y1, r2_y1], [c_z1, r2_z1], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm3_1, = ax.plot([c_x1, r3_x1], [c_y1, r3_y1], [c_z1, r3_z1], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm4_1, = ax.plot([c_x1, r4_x1], [c_y1, r4_y1], [c_z1, r4_z1], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)

        c_x2, c_y2, c_z2 = position_2[0:3,0]
        r1_x2, r1_y2, r1_z2 = position_2[3:6,0]
        r2_x2, r2_y2, r2_z2 = position_2[6:9,0]
        r3_x2, r3_y2, r3_z2 = position_2[9:12,0]
        r4_x2, r4_y2, r4_z2 = position_2[12:15,0]
        line_arm1_2, = ax.plot([c_x2, r1_x2], [c_y2, r1_y2], [c_z2, r1_z2], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm2_2, = ax.plot([c_x2, r2_x2], [c_y2, r2_y2], [c_z2, r2_z2], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm3_2, = ax.plot([c_x2, r3_x2], [c_y2, r3_y2], [c_z2, r3_z2], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm4_2, = ax.plot([c_x2, r4_x2], [c_y2, r4_y2], [c_z2, r4_z2], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)

        c_x3, c_y3, c_z3 = position_3[0:3,0]
        r1_x3, r1_y3, r1_z3 = position_3[3:6,0]
        r2_x3, r2_y3, r2_z3 = position_3[6:9,0]
        r3_x3, r3_y3, r3_z3 = position_3[9:12,0]
        r4_x3, r4_y3, r4_z3 = position_3[12:15,0]
        line_arm1_3, = ax.plot([c_x3, r1_x3], [c_y3, r1_y3], [c_z3, r1_z3], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm2_3, = ax.plot([c_x3, r2_x3], [c_y3, r2_y3], [c_z3, r2_z3], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm3_3, = ax.plot([c_x3, r3_x3], [c_y3, r3_y3], [c_z3, r3_z3], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm4_3, = ax.plot([c_x3, r4_x3], [c_y3, r4_y3], [c_z3, r4_z3], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)

        c_x4, c_y4, c_z4 = position_4[0:3,0]
        r1_x4, r1_y4, r1_z4 = position_4[3:6,0]
        r2_x4, r2_y4, r2_z4 = position_4[6:9,0]
        r3_x4, r3_y4, r3_z4 = position_4[9:12,0]
        r4_x4, r4_y4, r4_z4 = position_4[12:15,0]
        line_arm1_4, = ax.plot([c_x4, r1_x4], [c_y4, r1_y4], [c_z4, r1_z4], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm2_4, = ax.plot([c_x4, r2_x4], [c_y4, r2_y4], [c_z4, r2_z4], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm3_4, = ax.plot([c_x4, r3_x4], [c_y4, r3_y4], [c_z4, r3_z4], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm4_4, = ax.plot([c_x4, r4_x4], [c_y4, r4_y4], [c_z4, r4_z4], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)

        c_x5, c_y5, c_z5 = position_5[0:3,0]
        r1_x5, r1_y5, r1_z5 = position_5[3:6,0]
        r2_x5, r2_y5, r2_z5 = position_5[6:9,0]
        r3_x5, r3_y5, r3_z5 = position_5[9:12,0]
        r4_x5, r4_y5, r4_z5 = position_5[12:15,0]
        line_arm1_5, = ax.plot([c_x5, r1_x5], [c_y5, r1_y5], [c_z5, r1_z5], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm2_5, = ax.plot([c_x5, r2_x5], [c_y5, r2_y5], [c_z5, r2_z5], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm3_5, = ax.plot([c_x5, r3_x5], [c_y5, r3_y5], [c_z5, r3_z5], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)
        line_arm4_5, = ax.plot([c_x5, r4_x5], [c_y5, r4_y5], [c_z5, r4_z5], linewidth=0.5, color='black', marker='o', markersize=0.5,zorder=5)

        c_xl, c_yl, c_zl = position_l[0:3,0]
        r1_xl, r1_yl, r1_zl = position_l[3:6,0]
        r2_xl, r2_yl, r2_zl = position_l[6:9,0]
        r3_xl, r3_yl, r3_zl = position_l[9:12,0]
        r4_xl, r4_yl, r4_zl = position_l[12:15,0]
        r5_xl, r5_yl, r5_zl = position_l[15:18,0]
        r6_xl, r6_yl, r6_zl = position_l[18:21,0]

        line_loadaim1, =ax.plot([c_xl, r1_xl], [c_yl, r1_yl], [c_zl, r1_zl], linewidth=0.5, color='black',zorder=1+1.5*k)
        line_loadaim2, =ax.plot([c_xl, r2_xl], [c_yl, r2_yl], [c_zl, r2_zl], linewidth=0.5, color='black',zorder=1+1.5*k)
        line_loadaim3, =ax.plot([c_xl, r3_xl], [c_yl, r3_yl], [c_zl, r3_zl], linewidth=0.5, color='black',zorder=1+1.5*k)
        line_loadaim4, =ax.plot([c_xl, r4_xl], [c_yl, r4_yl], [c_zl, r4_zl], linewidth=0.5, color='black',zorder=1+1.5*k)
        line_loadaim5, =ax.plot([c_xl, r5_xl], [c_yl, r5_yl], [c_zl, r5_zl], linewidth=0.5, color='black',zorder=1+1.5*k)
        line_loadaim6, =ax.plot([c_xl, r6_xl], [c_yl, r6_yl], [c_zl, r6_zl], linewidth=0.5, color='black',zorder=1+1.5*k)
        line_loadaim12, =ax.plot([r1_xl, r2_xl], [r1_yl, r2_yl], [r1_zl, r2_zl], linewidth=1, color='orange',zorder=1+1.5*k)
        line_loadaim23, =ax.plot([r2_xl, r3_xl], [r2_yl, r3_yl], [r2_zl, r3_zl], linewidth=1, color='orange',zorder=1+1.5*k)
        line_loadaim34, =ax.plot([r3_xl, r4_xl], [r3_yl, r4_yl], [r3_zl, r4_zl], linewidth=1, color='orange',zorder=1+1.5*k)
        line_loadaim45, =ax.plot([r4_xl, r5_xl], [r4_yl, r5_yl], [r4_zl, r5_zl], linewidth=1, color='orange',zorder=1+1.5*k)
        line_loadaim56, =ax.plot([r5_xl, r6_xl], [r5_yl, r6_yl], [r5_zl, r6_zl], linewidth=1, color='orange',zorder=1+1.5*k)
        line_loadaim61, =ax.plot([r6_xl, r1_xl], [r6_yl, r1_yl], [r6_zl, r1_zl], linewidth=1, color='orange',zorder=1+1.5*k)

        tether1, =ax.plot([c_x0, r1_xl], [c_y0, r1_yl], [c_z0, r1_zl], linewidth=0.3, color='black',zorder=5)
        tether2, =ax.plot([c_x1, r2_xl], [c_y1, r2_yl], [c_z1, r2_zl], linewidth=0.3, color='black',zorder=1+1.5*k)
        tether3, =ax.plot([c_x2, r3_xl], [c_y2, r3_yl], [c_z2, r3_zl], linewidth=0.3, color='black',zorder=1+1.5*k)
        tether4, =ax.plot([c_x3, r4_xl], [c_y3, r4_yl], [c_z3, r4_zl], linewidth=0.3, color='black',zorder=5)
        tether5, =ax.plot([c_x4, r5_xl], [c_y4, r5_yl], [c_z4, r5_zl], linewidth=0.3, color='black',zorder=5)
        tether6, =ax.plot([c_x5, r6_xl], [c_y5, r6_yl], [c_z5, r6_zl], linewidth=0.3, color='black',zorder=5)
        self.set_axes_equal(ax)
        # customize
        
        if k==1:
            leg=plt.legend(['w/ $\Delta T$','Slot'],prop=font1,loc=(0,0.8),labelspacing=0.1,handlelength=1+k)
        if k==0:
            leg=plt.legend(['w/o $\Delta T$','Slot'],prop=font1,loc=(0,0.8),labelspacing=0.1,handlelength=1)
        leg.get_frame().set_linewidth(0.5)
        # elif k==1:
        #     leg=plt.legend(['Desired', 'Episode:'+str(k),'Episode:0'],prop=font1,loc=(-0.15,0.8),labelspacing=0.1,handlelength=1)
        #     leg.get_frame().set_linewidth(0.5)
        # else:
        #     leg=plt.legend(['Desired', 'Episode:'+str(k),'Episode:1','Episode:0'],prop=font1,loc=(-0.15,0.75),labelspacing=0.1,handlelength=1)
        #     leg.get_frame().set_linewidth(0.5)

        def update_traj(num):
            # customize
            time_text.set_text(time_template % (num * dt))
            line_traj_l.set_data_3d([position_l[0,:num], position_l[1,:num],position_l[2,:num]])
            # slot

            line_slotl1.set_data_3d([g1_p1x,g1_p2x],[g1_p1y,g1_p2y],[g1_p1z,g1_p2z])
            line_sloth1.set_data_3d([g1_p3x,g1_p4x],[g1_p3y,g1_p4y],[g1_p3z,g1_p4z])
            line_slotl2.set_data_3d([g2_p1x,g2_p2x],[g2_p1y,g2_p2y],[g2_p1z,g2_p2z])
            line_sloth2.set_data_3d([g2_p3x,g2_p4x],[g2_p3y,g2_p4y],[g2_p3z,g2_p4z])
            
            # uav
            c_x0, c_y0, c_z0 = position_0[0:3,num]
            r1_x0, r1_y0, r1_z0 = position_0[3:6,num]
            r2_x0, r2_y0, r2_z0 = position_0[6:9,num]
            r3_x0, r3_y0, r3_z0 = position_0[9:12,num]
            r4_x0, r4_y0, r4_z0 = position_0[12:15,num]
            line_arm1_0.set_data_3d([c_x0, r1_x0], [c_y0, r1_y0],[c_z0, r1_z0])
            line_arm2_0.set_data_3d([c_x0, r2_x0], [c_y0, r2_y0],[c_z0, r2_z0])
            line_arm3_0.set_data_3d([c_x0, r3_x0], [c_y0, r3_y0],[c_z0, r3_z0])
            line_arm4_0.set_data_3d([c_x0, r4_x0], [c_y0, r4_y0],[c_z0, r4_z0])

            c_x1, c_y1, c_z1 = position_1[0:3,num]
            r1_x1, r1_y1, r1_z1 = position_1[3:6,num]
            r2_x1, r2_y1, r2_z1 = position_1[6:9,num]
            r3_x1, r3_y1, r3_z1 = position_1[9:12,num]
            r4_x1, r4_y1, r4_z1 = position_1[12:15,num]
            line_arm1_1.set_data_3d([c_x1, r1_x1], [c_y1, r1_y1],[c_z1, r1_z1])
            line_arm2_1.set_data_3d([c_x1, r2_x1], [c_y1, r2_y1],[c_z1, r2_z1])
            line_arm3_1.set_data_3d([c_x1, r3_x1], [c_y1, r3_y1],[c_z1, r3_z1])
            line_arm4_1.set_data_3d([c_x1, r4_x1], [c_y1, r4_y1],[c_z1, r4_z1])

            c_x2, c_y2, c_z2 = position_2[0:3,num]
            r1_x2, r1_y2, r1_z2 = position_2[3:6,num]
            r2_x2, r2_y2, r2_z2 = position_2[6:9,num]
            r3_x2, r3_y2, r3_z2 = position_2[9:12,num]
            r4_x2, r4_y2, r4_z2 = position_2[12:15,num]
            line_arm1_2.set_data_3d([c_x2, r1_x2], [c_y2, r1_y2],[c_z2, r1_z2])
            line_arm2_2.set_data_3d([c_x2, r2_x2], [c_y2, r2_y2],[c_z2, r2_z2])
            line_arm3_2.set_data_3d([c_x2, r3_x2], [c_y2, r3_y2],[c_z2, r3_z2])
            line_arm4_2.set_data_3d([c_x2, r4_x2], [c_y2, r4_y2],[c_z2, r4_z2])

            c_x3, c_y3, c_z3 = position_3[0:3,num]
            r1_x3, r1_y3, r1_z3 = position_3[3:6,num]
            r2_x3, r2_y3, r2_z3 = position_3[6:9,num]
            r3_x3, r3_y3, r3_z3 = position_3[9:12,num]
            r4_x3, r4_y3, r4_z3 = position_3[12:15,num]
            line_arm1_3.set_data_3d([c_x3, r1_x3], [c_y3, r1_y3],[c_z3, r1_z3])
            line_arm2_3.set_data_3d([c_x3, r2_x3], [c_y3, r2_y3],[c_z3, r2_z3])
            line_arm3_3.set_data_3d([c_x3, r3_x3], [c_y3, r3_y3],[c_z3, r3_z3])
            line_arm4_3.set_data_3d([c_x3, r4_x3], [c_y3, r4_y3],[c_z3, r4_z3])

            c_x4, c_y4, c_z4 = position_4[0:3,num]
            r1_x4, r1_y4, r1_z4 = position_4[3:6,num]
            r2_x4, r2_y4, r2_z4 = position_4[6:9,num]
            r3_x4, r3_y4, r3_z4 = position_4[9:12,num]
            r4_x4, r4_y4, r4_z4 = position_4[12:15,num]
            line_arm1_4.set_data_3d([c_x4, r1_x4], [c_y4, r1_y4],[c_z4, r1_z4])
            line_arm2_4.set_data_3d([c_x4, r2_x4], [c_y4, r2_y4],[c_z4, r2_z4])
            line_arm3_4.set_data_3d([c_x4, r3_x4], [c_y4, r3_y4],[c_z4, r3_z4])
            line_arm4_4.set_data_3d([c_x4, r4_x4], [c_y4, r4_y4],[c_z4, r4_z4])

            c_x5, c_y5, c_z5 = position_5[0:3,num]
            r1_x5, r1_y5, r1_z5 = position_5[3:6,num]
            r2_x5, r2_y5, r2_z5 = position_5[6:9,num]
            r3_x5, r3_y5, r3_z5 = position_5[9:12,num]
            r4_x5, r4_y5, r4_z5 = position_5[12:15,num]
            line_arm1_5.set_data_3d([c_x5, r1_x5], [c_y5, r1_y5],[c_z5, r1_z5])
            line_arm2_5.set_data_3d([c_x5, r2_x5], [c_y5, r2_y5],[c_z5, r2_z5])
            line_arm3_5.set_data_3d([c_x5, r3_x5], [c_y5, r3_y5],[c_z5, r3_z5])
            line_arm4_5.set_data_3d([c_x5, r4_x5], [c_y5, r4_y5],[c_z5, r4_z5])

            c_xl, c_yl, c_zl = position_l[0:3,num]
            r1_xl, r1_yl, r1_zl = position_l[3:6,num]
            r2_xl, r2_yl, r2_zl = position_l[6:9,num]
            r3_xl, r3_yl, r3_zl = position_l[9:12,num]
            r4_xl, r4_yl, r4_zl = position_l[12:15,num]
            r5_xl, r5_yl, r5_zl = position_l[15:18,num]
            r6_xl, r6_yl, r6_zl = position_l[18:21,num]
            
            line_loadaim1.set_data_3d([c_xl, r1_xl], [c_yl, r1_yl], [c_zl, r1_zl])
            line_loadaim2.set_data_3d([c_xl, r2_xl], [c_yl, r2_yl], [c_zl, r2_zl])
            line_loadaim3.set_data_3d([c_xl, r3_xl], [c_yl, r3_yl], [c_zl, r3_zl])
            line_loadaim4.set_data_3d([c_xl, r4_xl], [c_yl, r4_yl], [c_zl, r4_zl])
            line_loadaim5.set_data_3d([c_xl, r5_xl], [c_yl, r5_yl], [c_zl, r5_zl])
            line_loadaim6.set_data_3d([c_xl, r6_xl], [c_yl, r6_yl], [c_zl, r6_zl])

            line_loadaim12.set_data_3d([r1_xl, r2_xl], [r1_yl, r2_yl], [r1_zl, r2_zl])
            line_loadaim23.set_data_3d([r2_xl, r3_xl], [r2_yl, r3_yl], [r2_zl, r3_zl])
            line_loadaim34.set_data_3d([r3_xl, r4_xl], [r3_yl, r4_yl], [r3_zl, r4_zl])
            line_loadaim45.set_data_3d([r4_xl, r5_xl], [r4_yl, r5_yl], [r4_zl, r5_zl])
            line_loadaim56.set_data_3d([r5_xl, r6_xl], [r5_yl, r6_yl], [r5_zl, r6_zl])
            line_loadaim61.set_data_3d([r6_xl, r1_xl], [r6_yl, r1_yl], [r6_zl, r1_zl])

            tether1.set_data_3d([c_x0, r1_xl], [c_y0, r1_yl], [c_z0, r1_zl])
            tether2.set_data_3d([c_x1, r2_xl], [c_y1, r2_yl], [c_z1, r2_zl])
            tether3.set_data_3d([c_x2, r3_xl], [c_y2, r3_yl], [c_z2, r3_zl])
            tether4.set_data_3d([c_x3, r4_xl], [c_y3, r4_yl], [c_z3, r4_zl])
            tether5.set_data_3d([c_x4, r5_xl], [c_y4, r5_yl], [c_z4, r5_zl])
            tether6.set_data_3d([c_x5, r6_xl], [c_y5, r6_yl], [c_z5, r6_zl])

            # trajectory ref
            num=sim_horizon-1
            # line_traj_ref.set_data_3d(position_ref[0,:num], position_ref[1,:num],position_ref[2,:num])
            # if k==1:
            #     line_traj_l0.set_data_3d([position_l0[0,:num], position_l0[1,:num],position_l0[2,:num]])
            # if k>1:
            #     line_traj_l1.set_data_3d([position_l1[0,:num], position_l1[1,:num],position_l1[2,:num]])
            #     line_traj_l0.set_data_3d([position_l0[0,:num], position_l0[1,:num],position_l0[2,:num]])
            self.set_axes_equal(ax)
            return line_traj_l,line_arm1_0, line_arm2_0, line_arm3_0, line_arm4_0, \
                   line_arm1_1, line_arm2_1, line_arm3_1, line_arm4_1, \
                   line_arm1_2, line_arm2_2, line_arm3_2, line_arm4_2, \
                   line_arm1_3, line_arm2_3, line_arm3_3, line_arm4_3, \
                   line_arm1_4, line_arm2_4, line_arm3_4, line_arm4_4, \
                   line_arm1_5, line_arm2_5, line_arm3_5, line_arm4_5, \
                   line_loadaim1, line_loadaim2, line_loadaim3, line_loadaim4, line_loadaim5, line_loadaim6, \
                   line_loadaim12, line_loadaim23, line_loadaim34, line_loadaim45, line_loadaim56, line_loadaim61, \
                   tether1, tether2, tether3, tether4, tether5, tether6, time_text, \
                   line_slotl1, line_slotl2, line_sloth1, line_sloth2

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=dt*4000, blit=True)

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('training' + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()




        






    

    



