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
                
        ret = 0         # return value
        step_count = 25 # time between two control signals | used to predice distance travelled at current speed in ove time step
        time_step = 0.1 # time between two control signals | used to predice distance travelled at current speed in ove time step
        P_v = 1         # Proportal control for velocity 
        P_d = 2         # Proportal control for velocity 

        dist_to_cover = dist_to_lead - self.distance_threshold      # distnace to cover untill we reach distance threshold 
        velocity_to_gain = target_velocity - ego_velocity           # velocity to gain untill we reach target velocity 
        dist_buffer = step_count * time_step * ego_velocity         # buffer to account for distance travelled while stopping
        

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
        if dist_to_cover > dist_buffer + self.distance_threshold:
            #  we have a lot of distance to cover    
            # try to attain target velocity
            if velocity_to_gain > 0:
                ret = P_v * velocity_to_gain + P_d * dist_to_cover
                flag = "P_v P_d"
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