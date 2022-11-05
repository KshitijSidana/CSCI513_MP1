"""This file is the main controller file

Here, you will design the controller for your for the adaptive cruise control system.
"""

from mp1_simulator.simulator import Observation

import numpy as np
from casadi import *
import do_mpc
import matplotlib.pyplot as plt

# NOTE: Very important that the class name remains the same
class Controller:
    def __init__(self, target_speed: float, distance_threshold: float):
        self.target_speed = target_speed
        self.distance_threshold = distance_threshold

    def run_step(self, obs: Observation) -> float:
        """This is the main run step of the controller.

        Here, you will have to read in the observatios `obs`, process it, and output an
        acceleration value. The acceleration value must be some value between -10.0 and 10.0.

        Note that the acceleration value is really some control input that is used
        internally to compute the throttle to the car.

        Below is some example code where the car just outputs the control value 10.0
        """

        ego_velocity = obs.velocity
        target_velocity = obs.target_velocity
        dist_to_lead = obs.distance_to_lead

        # Do your magic...
        ret = 0.00
        ## MODEL
        model_type = 'continuous' # either 'discrete' or 'continuous'
        model = do_mpc.model.Model(model_type)

        ### States and control inputs
        #   States struct (optimization variables):
        V_ego = model.set_variable(var_type='_x', var_name='V_ego', shape=(1,1))
        S_led = model.set_variable(var_type='_x', var_name='S_led', shape=(1,1))
        # T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1,1))
        # T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1,1))

        #   Input struct (optimization variables):
        A = model.set_variable(var_type='_u', var_name='A')
        # F = model.set_variable(var_type='_u', var_name='F')
        # Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

        # #   Certain parameters
        # K0_ab = 1.287e12 # K0 [h^-1]
        # K0_bc = 1.287e12 # K0 [h^-1]
        # K0_ad = 9.043e9 # K0 [l/mol.h]
        # R_gas = 8.3144621e-3 # Universal gas constant
        # E_A_ab = 9758.3*1.00 #* R_gas# [kj/mol]
        # E_A_bc = 9758.3*1.00 #* R_gas# [kj/mol]
        # E_A_ad = 8560.0*1.0 #* R_gas# [kj/mol]
        # H_R_ab = 4.2 # [kj/mol A]
        # H_R_bc = -11.0 # [kj/mol B] Exothermic
        # H_R_ad = -41.85 # [kj/mol A] Exothermic
        # Rou = 0.9342 # Density [kg/l]
        # Cp = 3.01 # Specific Heat capacity [kj/Kg.K]
        # Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
        # A_R = 0.215 # Area of reactor wall [m^2]
        # V_R = 10.01 #0.01 # Volume of reactor [l]
        # m_k = 5.0 # Coolant mass[kg]
        # T_in = 130.0 # Temp of inflow [Celsius]
        # K_w = 4032.0 # [kj/h.m^2.K]
        # C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]

        #   Uncertain parameters:
        alpha = model.set_variable(var_type='_p', var_name='alpha')
        beta = model.set_variable(var_type='_p', var_name='beta')

        # #   Auxiliary terms
        # K_1 = beta * K0_ab * exp((-E_A_ab)/((T_R+273.15)))
        # K_2 =  K0_bc * exp((-E_A_bc)/((T_R+273.15)))
        # K_3 = K0_ad * exp((-alpha*E_A_ad)/((T_R+273.15)))

        # #   Additionally, we define an artificial variable of interest, that is not a state of the system, but will be later used for plotting:
        # T_dif = model.set_expression(expr_name='T_dif', expr=T_R-T_K)

        #   With the help of the  𝑘𝑖 -s and  𝑇dif  we can define the ODEs:
        model.set_rhs('V_ego', A)
        model.set_rhs('S_led', -1*V_ego)
       
        #   Build the model
        model.setup()



        ## CONTROLLER
        mpc = do_mpc.controller.MPC(model)

        setup_mpc = {
            'n_horizon': 20,
            'n_robust': 1,
            'open_loop': 0,
            't_step': 0.005,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 2,
            'collocation_ni': 2,
            'store_full_solution': True,
            # Use MA27 linear solver in ipopt for faster calculations:
            #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
        }

        mpc.set_param(**setup_mpc) # the 88 operator unwraps the dict and assigns the fn param as key=val pairs.

        #   Because the magnitude of the states and inputs is very different, we introduce scaling factors:
        mpc.scaling['_x', 'V_ego'] = 1
        mpc.scaling['_x', 'S_led'] = 1
        mpc.scaling['_u', 'A'] = 1

        ### Objective
        #   The goal of the CSTR is to obtain a mixture with a concentration of  𝐶B,ref=0.6  mol/l. 
        #   Additionally, we add a penalty on input changes for both control inputs, 
        #   to obtain a smooth control performance.

        _x = model.x
        # target_velocity = obs.target_velocity
        # dist_to_lead = obs.distance_to_lead
        # self.target_speed = target_speed
        # self.distance_threshold = distance_threshold

        mterm = (_x['V_ego'] - self.target_speed)**2 # terminal cost
        lterm = (_x['S_led'] - self.distance_threshold)**2 # stage cost

        mpc.set_objective(mterm=mterm, lterm=lterm)

        mpc.set_rterm(A=0.1) # input penalty

        ### Constraints

        #   In the next step, the constraints of the control problem are set.
        #   In this case, there are only upper and lower bounds for each state and the input:

        # lower bounds of the states
        # mpc.bounds['lower', '_x', 'V_ego'] = 0
        mpc.bounds['lower', '_x', 'S_led'] = self.distance_threshold 

        # upper bounds of the states
        # mpc.bounds['upper', '_x', 'V_ego'] = 500 # default is infinite 
        # mpc.bounds['upper', '_x', 'S_led'] = 101 # sim doesnot give a val higher than 100

        # lower bounds of the inputs
        mpc.bounds['lower', '_u', 'A'] = -10

        # upper bounds of the inputs
        mpc.bounds['upper', '_u', 'A'] = 10

        # SOFT CONSTRSINT NOT NEEDED
        # If a constraint is not critical, it is possible to implement it as a soft constraint. 
        # mpc.set_nl_cons('T_R', _x['T_R'], ub=140, soft_constraint=True, penalty_term_cons=1e2)

        ### Uncertain values
        #   The explicit values of the two uncertain parameters $\alpha$ and $\beta$, which are considered in the scenario tree, are given by:
        alpha_var = np.array([1., 1.1, 0.9])
        beta_var  = np.array([1., 1.05, 0.95])

        mpc.set_uncertainty_values(alpha = alpha_var, beta = beta_var)
        # This means with n_robust=1, that 9 different scenarios are considered. 
        
        # The setup of the MPC controller is concluded by:
        mpc.setup()


        ## ESTIMATOR
        estimator = do_mpc.estimator.StateFeedback(model)

        ## SIMULATOR
        simulator = do_mpc.simulator.Simulator(model)
        # For the simulation, we use the same time step t_step as for the optimizer:
        params_simulator = {
            'integration_tool': 'cvodes',
            'abstol': 1e-10,
            'reltol': 1e-10,
            't_step': 0.05
        }

        simulator.set_param(**params_simulator)
        ### Realizations of uncertain parameters

        # For the simulatiom, it is necessary to define the numerical realizations of 
        # the uncertain parameters in `p_num` and the time-varying parameters in `tvp_num`.
        # First, we get the structure of the uncertain and time-varying parameters:

        p_num = simulator.get_p_template()
        tvp_num = simulator.get_tvp_template()

        # We define two functions which are called in each simulation step, 
        # which return the current realizations of the parameters, 
        # with respect to defined inputs (in this case t_now):

        # function for time-varying parameters
        def tvp_fun(t_now):
            return tvp_num

        # uncertain parameters
        p_num['alpha'] = 1
        p_num['beta'] = 1
        def p_fun(t_now):
            return p_num

        # These two custum functions are used in the simulation via:
        simulator.set_tvp_fun(tvp_fun)
        simulator.set_p_fun(p_fun)

        # By defining p_fun as above, the function will always return the value 1.0 
        # for both  𝛼  and  𝛽 . To finish the configuration of the simulator, call:
        simulator.setup()

        # Closed-loop simulation
        # For the simulation of the MPC configured for the CSTR, we inspect the file main.py. 
        # We define the initial state of the system and 
        # set it for all parts of the closed-loop configuration:

        # Set the initial state of mpc, simulator and estimator:
        V_ego_0 = ego_velocity
        S_led_0 = dist_to_lead
        
        # C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
        # C_b_0 = 0.5 # This is the controlled variable [mol/l]
        # T_R_0 = 134.14 #[C]
        # T_K_0 = 130.0 #[C]
        # x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)
        x0 = np.array([V_ego_0, S_led_0]).reshape(-1,1)

        mpc.x0 = x0
        simulator.x0 = x0
        estimator.x0 = x0

        mpc.set_initial_guess()
        u0 = None
        # Now, we simulate the closed-loop for 10 steps (and suppress the output of the cell with the magic command %%capture):
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        
        for k in range(10):
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)
            print(f"k, {k}, u0, {u0}, y_next[0], {y_next[0]}, y_next[1], {y_next[1]}, x0[0], {x0[0]}, x0[1], {x0[1]}")

        ret = u0[0][0]
        print(f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(f"&&&&&&&&&&&&&&&& {u0[0][0]} &&&&&&&&&&&&&&")
        print(f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        # # ## Animating the results

        # # # To animate the results, we first configure the **do-mpc** graphics object, which is initiated with the respective data object:
        # mpc_graphics = do_mpc.graphics.Graphics(mpc.data)

        # from matplotlib import rcParams
        # rcParams['axes.grid'] = True
        # rcParams['font.size'] = 18
        
        # fig, ax = plt.subplots(5, sharex=True, figsize=(16,12))
        # # Configure plot:
        # mpc_graphics.add_line(var_type='_x', var_name='V_ego', axis=ax[0])
        # mpc_graphics.add_line(var_type='_x', var_name='S_led', axis=ax[1])
        # mpc_graphics.add_line(var_type='_u', var_name='A', axis=ax[2])
        # ax[0].set_ylabel('time [h]')
        # ax[1].set_ylabel('time [h]')
        # ax[2].set_xlabel('time [h]')



        # # Update properties for all prediction lines:
        # for line_i in mpc_graphics.pred_lines.full:
        #     line_i.set_linewidth(2)
        # # Highlight nominal case:
        # for line_i in np.sum(mpc_graphics.pred_lines['_x', :, :,0]):
        #     line_i.set_linewidth(5)
        # for line_i in np.sum(mpc_graphics.pred_lines['_u', :, :,0]):
        #     line_i.set_linewidth(5)
        # # for line_i in np.sum(mpc_graphics.pred_lines['_aux', :, :,0]):
        # #     line_i.set_linewidth(5)

        # # Add labels
        # label_lines0 = mpc_graphics.result_lines['_x', 'V_ego']
        # label_lines1 = mpc_graphics.result_lines['_x', 'S_led']
        # label_lines2 = mpc_graphics.result_lines['_x', 'A']
        # ax[0].legend(label_lines0, ['V_ego'])
        # ax[1].legend(label_lines1, ['S_led'])
        # ax[2].legend(label_lines2, ['A'])

        # fig.align_ylabels()


        # from matplotlib.animation import FuncAnimation, ImageMagickWriter
        
        # def update(t_ind):
        #     print('Writing frame: {}.'.format(t_ind), end='\r')
        #     mpc_graphics.plot_results(t_ind=t_ind)
        #     mpc_graphics.plot_predictions(t_ind=t_ind)
        #     mpc_graphics.reset_axes()
        #     lines = mpc_graphics.result_lines.full
        #     return lines

        # n_steps = mpc.data['_time'].shape[0]


        # anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

        # gif_writer = ImageMagickWriter(fps=5)
        # anim.save('anim_CSTR.gif', writer=gif_writer)


        return ret
