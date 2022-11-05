"""This file is the main controller file

Here, you will design the controller for your for the adaptive cruise control system.
"""

from mp1_simulator.simulator import Observation


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
        if ego_velocity==self.target_speed and dist_to_lead <= self.distance_threshold:
            return -0.01
        
        if ego_velocity < self.target_speed*0.8:
            if dist_to_lead>self.distance_threshold *5:
                return 10*((100-dist_to_lead-self.distance_threshold)/100)
        if ego_velocity< self.target_speed*0.8:
            if dist_to_lead>self.distance_threshold *2.5:
                return 1*((100-dist_to_lead-self.distance_threshold)/100)
        if ego_velocity< self.target_speed*0.8:
            if dist_to_lead>self.distance_threshold *1:
                return -1*((100-dist_to_lead-self.distance_threshold)/100)
        # else:
        #     if dist_to_lead<self.distance_threshold * 1.1:
        #         return -10
        #     else :
        #         return -10


        return -10.0
