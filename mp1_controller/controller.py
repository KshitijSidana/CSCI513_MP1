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
        P_v = 5         # Proportal control for velocity 
        P_d = 20         # Proportal control for velocity 

        dist_to_cover = dist_to_lead - self.distance_threshold      # distnace to cover untill we reach distance threshold 
        velocity_to_gain = target_velocity - ego_velocity           # velocity to gain untill we reach target velocity 
        dist_buffer = step_count * time_step * ego_velocity                        # buffer to account for distance travelled while stopping
        

        def verbose(condition):
            print(f"condition, {condition}, ret, {ret}, ego_velocity, {ego_velocity}, /, {target_velocity}, dist_to_lead, {dist_to_lead}, , dist_to_cover, {dist_to_cover}, dist_buffer, {dist_buffer}, velocity_to_gain, {velocity_to_gain}")
            with open("/home/kshitij/Desktop/OP/op.txt", "a") as f:
                f.write("condition, ")
                f.write(str(condition))
                f.write(", ret, ")
                f.write(str(ret))
                f.write(", ego_velocity, ")
                f.write(str(ego_velocity))
                f.write(", target_velocity, ")
                f.write(str(target_velocity))
                f.write(", dist_to_lead, ")
                f.write(str(dist_to_lead))
                f.write(", dist_to_cover, ")
                f.write(str(dist_to_cover))
                f.write(", dist_buffer, ")
                f.write(str(dist_buffer))
                f.write(", velocity_to_gain, ")
                f.write(str(velocity_to_gain))
                f.write("\n")
        
        # Emergency stop - too close to target 
        if dist_to_cover-dist_buffer <=0:
            ret = -10
            verbose("em_stp")
            return ret

        # Slow down - going too fast 
        if velocity_to_gain < 0 :
            velocity_to_loose = (target_velocity - ego_velocity)
            ret = P_v * velocity_to_loose

            # distance condition violated
            if dist_to_cover-dist_buffer <=0:
                ret =-10
                verbose("dis_vio")
                return ret
            else:
                verbose("slow")
                return ret
       
        flag = ""
        # try to attain target velocity
        if velocity_to_gain > 0:
            ret = P_v * velocity_to_gain
            flag = "go_to_tgt"
            # distance condition
            if dist_to_cover-dist_buffer <=0:
                flag = "dist constraint"
                ret = P_d * dist_to_cover
            else:
                ret = min(ret, P_d * dist_to_cover)
                if ret == P_d * dist_to_cover:
                    flag = "min is dist constraint"
                else:
                    flag = "min is go to tgt"
                    
            verbose(flag)
            return ret


        # if ego_velocity > 0 and dist_to_cover-dist_buffer <=0:
        #     ret = -1 * P_d * dist_to_cover
        #     verbose()
        #     return ret
        

        # #  ego is already greater than target velo
        # ret = 0 # stop acc/throttle
        # verbose()
        # return ret


        # elif (velocity_to_gain<0):
        #     ret = -1* P_v * velocity_to_gain
        #     verbose()
        #     return ret
        # else:
        #     ret = 0
        #     verbose()
        #     return ret
        verbose("default_case")
        return ret
