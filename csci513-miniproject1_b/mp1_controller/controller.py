"""This file is the main controller file

Here, you will design the controller for your for the adaptive cruise control system.
"""
'''
    NOTE
        Map - Town06
        Requires:
            - numpy
            - time
'''
from mp1_simulator.simulator import Observation

import numpy as np
import time

# NOTE: Very important that the class name remains the same
class Controller:
    def __init__(self, target_speed: float, distance_threshold: float):
        self.target_speed = target_speed
        self.distance_threshold = distance_threshold
        
        # Set-up for Magic...
        self.previousTime = time.time()
        self.v_error = [0.00]
        self.d_error = [0.00]
        self.d_del_i = [0.00]
        self.v_del_i = [0.00]
        # with open("/home/kshitij/Desktop/OP/op_pid.txt", "a") as f:
        #     f.write("t_pow, currentTime, elapsedTime, setpoint, input, error, self.cumError, rateError, out, kp, ki, kd\n")

    # Secret Spells for Magic...
    def pid (self, input: float, setpoint: float, kp: float, ki: float, kd: float, t_pow: float, episodes = 20)->float:
        currentTime = time.time();                        # get current time
        elapsedTime = currentTime - self.previousTime;    # compute time elapsed from previous computation
        error = setpoint - input                          # error

        cumError = float()
        lastError = float()

        if t_pow == 1:
            lastError = self.v_error[-1]

            self.v_error.append(error)                               # store current error
            self.v_del_i.append(error * (elapsedTime**t_pow)) ;      # store current integral
            
            if len(self.v_error)>episodes:
                self.v_error = self.v_error[-episodes:]
                try:
                    self.v_del_i = self.v_del_i[-episodes:]
                except:
                    # print("Failed | try:                     self.v_del_i = self.v_del_i[-episodes:]")
                    pass
            
            cumError = sum(self.v_del_i)
        
        elif t_pow ==2 :
            lastError = self.d_error[-1]

            self.d_error.append(error)                               # store current error
            self.v_del_i.append(error * (elapsedTime**t_pow)) ;      # store current integral
            
            if len(self.d_error)>episodes:
                self.d_error = self.d_error[-episodes:]
                try:
                    self.d_del_i = self.d_del_i[-episodes:]
                except:
                    # print("Failed | try:                     self.d_del_i = self.d_del_i[-episodes:]")
                    pass

            cumError = sum(self.v_del_i)

        rateError = (error - lastError)/(elapsedTime**t_pow);       # compute derivative
        out = kp*error + ki*cumError + kd*rateError;                # PID output               
        self.previousTime = currentTime;                            # store current time

        # with open("/home/kshitij/Desktop/OP/op_pid.txt", "a") as f:
        #     f.write(f"{t_pow},{currentTime},{elapsedTime},{setpoint},{input},{error},{cumError},{rateError},{out},{kp},{ki},{kd}\n")
        # print(f"{t_pow},{currentTime},{elapsedTime},{setpoint},{input},{error},{self.cumError},{rateError},{out},{kp},{ki},{kd}\n")

        return out

    def run_step(self, obs: Observation, estimate_dist) -> float:        
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
        
        #### Magic...
        
        step_count = 15     # number of time steps to calculate braking distance    # used to predict distance travelled at current speed
        time_step = 0.1     # time between two control signals                      # used to predict distance travelled at current speed
        dist_buffer = step_count * time_step * ego_velocity**1.25                   # buffer to account for distance travelled while stopping
        dist_to_cover = dist_to_lead - (self.distance_threshold + 0.75)             # distnace to cover untill we reach (distance threshold + 0.75)
        #                                                                           # Offset of 0.75 as collision is detected when dist_to_lead ~= 4.636

        flag = ""

        ### Base case - control based on distance
        Kc = 0.25*8
        Pc = 0.005*20
        setpoint = dist_to_cover
        input =  dist_buffer
        flag = "ddflt"
        ret = self.pid (input, setpoint, 0.5*Kc, 0.5*Pc, Pc/10, t_pow= 2)

        ### Best case - control based on speed
        if dist_to_cover > max(2*target_velocity, dist_buffer) and ret > 0: 
            # Super safe to reach target velocity ASAP!
            flag = "d>2tv"
            Kc = 0.25*68
            Pc = 0.005*17
            setpoint = target_velocity
            input = ego_velocity
            ret = self.pid (input, setpoint, 0.5*Kc, 0.5*Pc, Pc/1000, t_pow=1, episodes = 12)
            
        ### Edge cases 
        ## Unlikely to happen given the nature of the aforementioned conditions

        # Going too fast 
        if ego_velocity >= target_velocity:
            flag = "v>=tv"
            ret = 4*(target_velocity - ego_velocity)

        # Getting too close
        if dist_to_cover <= 0.05* self.distance_threshold:
            flag = "e_brk"
            ret = -10

        print(f"{flag} ,ret, {ret}, egoV, {ego_velocity},/,{target_velocity}, set, {setpoint}, ip, {input}, d2lead, {obs.distance_to_lead}, d2cover, {dist_to_cover}, dbuff, {dist_buffer}")
        
        ### Finally
        ## Before we return the value, we clip it between [-10, 10]
        ret = np.clip(ret, -10.0, 10.0)
        
        return ret