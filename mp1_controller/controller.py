"""This file is the main controller file

Here, you will design the controller for your for the adaptive cruise control system.
"""

from mp1_simulator.simulator import Observation

import numpy as np
from casadi import *
import do_mpc
import matplotlib.pyplot as plt

import time

# NOTE: Very important that the class name remains the same
class Controller:
    def __init__(self, target_speed: float, distance_threshold: float):
        self.target_speed = target_speed
        self.distance_threshold = distance_threshold
        
        self.previousTime = time.time()
        self.v_error = [0.00]
        self.d_error = [0.00]
        self.d_del_i = [0.00]
        self.v_del_i = [0.00]
        with open("/home/kshitij/Desktop/OP/op_pid.txt", "w") as f:
            f.write("t_pow, currentTime, elapsedTime, setpoint, input, error, self.cumError, rateError, out, kp, ki, kd\n")


    def pid (self, input: float, setpoint: float, kp: float, ki: float, kd: float, t_pow: float, episodes = 20)->float:
        currentTime = time.time();                          # get current time
        
        elapsedTime = currentTime - self.previousTime;    # compute time elapsed from previous computation
        
        error = setpoint - input                             # error

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
                    print("Failed | try:                     self.v_del_i = self.v_del_i[-episodes:]")
            
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
                    print("Failed | try:                     self.d_del_i = self.d_del_i[-episodes:]")
            
            cumError = sum(self.v_del_i)

        rateError = (error - lastError)/(elapsedTime**t_pow);       # compute derivative
 
        out = kp*error + ki*cumError + kd*rateError;                # PID output               
 
        self.previousTime = currentTime;                            # store current time


        with open("/home/kshitij/Desktop/OP/op_pid.txt", "a") as f:
            f.write(f"{t_pow},{currentTime},{elapsedTime},{setpoint},{input},{error},{cumError},{rateError},{out},{kp},{ki},{kd}\n")
        # print(f"{t_pow},{currentTime},{elapsedTime},{setpoint},{input},{error},{self.cumError},{rateError},{out},{kp},{ki},{kd}\n")

        return out

    # def pid (self, input: float, setpoint: float, kp: float, ki: float, kd: float, t_pow: float, episodes = 20)->float:
    #     currentTime = time.time();                          # get current time
    #     elapsedTime = (currentTime - self.previousTime);    # compute time elapsed from previous computation
        
    #     error = input - setpoint # error

    #     cumError = float()
    #     lastError = float()
        
        
    #     if t_pow == 1:
    #         cumError = sum(self.vError)
    #         lastError = self.vError[-1]
    #         self.vError.append(error)                               # store current error
    #         if len(self.vError)>episodes:
    #             self.vError = self.vError[-episodes:]
        
    #     elif t_pow ==2 :
    #         cumError = sum(self.dError)
    #         lastError = self.dError[-1] 
    #         self.dError.append(error)                               # store current error
    #         if len(self.dError)>episodes:
    #             self.dError = self.dError[-episodes:]

    #     cumError += error * (elapsedTime**t_pow) ;                  # compute integral
    #     rateError = (error - lastError)/(elapsedTime**t_pow);       # compute derivative
 
    #     out = kp*error + ki*cumError + kd*rateError;                # PID output               
 
    #     self.previousTime = currentTime;                            # store current time


    #     with open("/home/kshitij/Desktop/OP/op_pid.txt", "a") as f:
    #         f.write(f"{currentTime},{elapsedTime},{setpoint},{input},{error},{self.cumError},{rateError},{out},{kp},{ki},{kd}\n")
    #     # print(f"{currentTime},{elapsedTime},{setpoint},{input},{error},{self.cumError},{rateError},{out},{kp},{ki},{kd}\n")

    #     return out

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
        
        # Magic...
        
        step_count = 15     # number of time steps to calculate braking distance    | used to predict distance travelled at current speed
        time_step = 0.1     # time between two control signals                      | used to predict distance travelled at current speed
        dist_buffer = step_count * time_step * ego_velocity**1.25      # buffer to account for distance travelled while stopping
        dist_to_cover = dist_to_lead - self.distance_threshold      # distnace to cover untill we reach distance threshold 


        Kc = 0.25*6
        Pc = 0.00325
        setpoint = dist_to_cover
        input =  self.distance_threshold + dist_buffer
        ret = self.pid (input, setpoint, 0.5*Kc, 0.5*Pc, Pc/8, t_pow= 2)

        if dist_to_lead > 2*target_velocity and ret > -15: # hence safe to chase target velocity
            Kc = 0.25*10
            Pc = 0.00325*10
            setpoint = target_velocity
            input = ego_velocity
            ret = self.pid (input, setpoint, 0.5*Kc, 0.5*Pc, Pc/8, t_pow=1)
            print(f"vel ,ret, {ret}, egoV, {ego_velocity},/,{target_velocity}, set, {setpoint}, ip, {input}, d2lead, {dist_to_lead}, d2cover, {dist_to_cover}, dbuff, {dist_buffer}")
        else:
            print(f"dst ,ret, {ret}, egoV, {ego_velocity},/,{target_velocity}, set, {setpoint}, ip, {input}, d2lead, {dist_to_lead}, d2cover, {dist_to_cover}, dbuff, {dist_buffer}")


        # Going too fast 
        if ego_velocity >= target_velocity:
            ret = 2*(target_velocity - ego_velocity)


        return ret       

        
        
        ret = 0         # return value
        step_count = 15 # number of time steps to calculate braking distance    | used to predict distance travelled at current speed
        time_step = 0.1 # time between two control signals                      | used to predict distance travelled at current speed
        P_v = 1         # Proportional gain for velocity 
        P_d = 2         # Proportional gain for velocity 

        dist_to_cover = dist_to_lead - self.distance_threshold      # distnace to cover untill we reach distance threshold 
        velocity_to_gain = target_velocity - ego_velocity           # velocity to gain untill we reach target velocity 
        dist_buffer = step_count * time_step * ego_velocity**2         # buffer to account for distance travelled while stopping
        

        def verbose(condition):
            print(f"condition, {condition}, ret, {ret}, ego_velocity, {ego_velocity}, /, {target_velocity}, dist_to_lead, {dist_to_lead}, , dist_to_cover, {dist_to_cover}, dist_buffer, {dist_buffer}, velocity_to_gain, {velocity_to_gain}")
            with open("/home/kshitij/Desktop/OP/op.txt", "a") as f:
                f.write("condition, ")
                f.write(str(condition))
                f.write(", thr_brk, ")
                f.write(str(ret))
                f.write(", ego_vel, ")
                f.write(str(ego_velocity))
                f.write(", tgt_vel, ")
                f.write(str(target_velocity))
                f.write(", dtolead, ")
                f.write(str(dist_to_lead))
                f.write(", dtocvr, ")
                f.write(str(dist_to_cover))
                f.write(", d_bufr, ")
                f.write(str(dist_buffer))
                f.write(", v2gain, ")
                f.write(str(velocity_to_gain))
                f.write("\n")
        
        # Emergency stop - too close to target  | failed spec
        if dist_to_cover <=0 - 0.1 * ego_velocity:
            ret = -10
            verbose("F em_stp")
            return ret

        # Slow down - going too fast  | failed spec
        if velocity_to_gain < 0 :
            velocity_to_loose = (target_velocity - ego_velocity)
            ret = P_v * velocity_to_loose
            verbose("F slow")
            return ret

        # # achieve almost cruise speed asap 
        # if dist_to_lead>target_velocity:
        #     ret = 10
        #     verbose("asap")
        #     return ret



        flag = ""
        if dist_to_cover > min(target_velocity*2 ,dist_buffer + self.distance_threshold):
            #  we have a lot of distance to cover    
            # try to attain target velocity
            if velocity_to_gain > 0.1*target_velocity:
                ret = P_v * velocity_to_gain + P_d * dist_to_cover
                flag = "P_v  +  P_d"
            elif velocity_to_gain > 0:
                ret = P_v * velocity_to_gain
                flag = "P_v  no P_d"
            else:
                flag = "at_tgt_vel"
                ret = 0

        elif dist_to_cover <= dist_buffer:
            #  we have a short distance to cover    
            if velocity_to_gain > 0:
                velocity_to_loose = -1 * ego_velocity
                ret = P_v * velocity_to_loose
                flag = "P_v vel"
                # ret = P_d * dist_to_cover
        else:
            #  we have almost reached    
            if velocity_to_gain > 0:
                ret  = P_d * (dist_to_cover)
                flag = "P_d dist_to_cover"
        
        
        verbose(flag)
        return ret