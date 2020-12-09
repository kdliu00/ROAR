from ROAR.agent_module.agent import Agent
from pathlib import Path
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
import logging
import cv2
import numpy as np


class ImitationAgent(Agent):
    def __init__(self, model, steer_model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.steer_model = steer_model

    def run_step(self, vehicle: Vehicle,
                 sensors_data: SensorsData) -> VehicleControl:
        super(ImitationAgent, self).run_step(vehicle=vehicle,
                                             sensors_data=sensors_data)
        image = sensors_data.front_rgb.data
        input_image = np.array([cv2.resize(image, (160, 120),
                                           interpolation=cv2.INTER_LANCZOS4)])

        output = self.model.predict(input_image, batch_size=None)
        steer_output = self.steer_model.predict(input_image, batch_size=None)

        control = VehicleControl()
        control.throttle = 0.7

        if round(steer_output[0][0].item()) == 0:
            print("Not steering")
            control.steering = 0.0
        else:
            print("Steering")
            control.steering = output[0][0].item()

        print(f"Control: {control}\n")

        return control
