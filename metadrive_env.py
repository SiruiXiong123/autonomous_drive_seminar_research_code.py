import copy
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from typing import Union
import numpy as np
import math
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.utils import clip, Config
# from egostate_obs import EgoStateobservation
from EgostateAndNavigation_obs import EgoStateNavigationobservation
from mpmath.matrices.eigen import hessenberg_qr
from openpyxl.styles.builtins import total
from sqlalchemy import lateral
from denormalization import denormalize_lidar_data
from risk_quantification import compute_tuoyuan_risk
from ttc_risk import compute_ttc_risk
from combined_risk import compute_combined_risk

METADRIVE_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    num_scenarios=1,

    # ===== PG Map Config =====
    map=4,  # int or string: an easy way to fill map_config
    block_dist_config=PGBlockDistConfig,
    random_lane_width=False,
    random_lane_num=False,
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: None,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 4,
        "exit_length": 50,
        "start_position": [0, 0],
    },
    store_map=True,

    # ===== Traffic =====
    traffic_density=0.1,
    need_inverse_traffic=False,
    traffic_mode=TrafficMode.Trigger,  # "Respawn", "Trigger"
    random_traffic=False,  # Traffic is randomized at default.
    # this will update the vehicle_config and set to traffic
    traffic_vehicle_config=dict(
        show_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=False,
        show_lidar=False,
        show_lane_line_detector=False,
        show_side_detector=False,
    ),

    # ===== Object =====
    accident_prob=0.,  # accident may happen on each block with this probability, except multi-exits block
    static_traffic_object=True,  # object won't react to any collisions

    # ===== Others =====
    use_AI_protector=False,
    save_level=0.5,

    # ===== Agent =====
    random_spawn_lane_index=True,
    vehicle_config=dict(navigation_module=NodeNetworkNavigation),
    agent_configs={
        DEFAULT_AGENT: dict(
            use_special_color=True,
            spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
        )
    },

    # ===== Reward Scheme =====
    # See: https://github.com/metadriverse/metadrive/issues/283
    success_reward=20.0,
    out_of_road_penalty=5.0,
    run_out_of_time_penalty=5.0,
    crash_vehicle_penalty=5.0,
    crash_object_penalty=5.0,
    crash_sidewalk_penalty=5.0,
    driving_reward=1.0,
    speed_reward=1.0,
    use_lateral_reward=False,
    heading_reward=0.10,
    overtake_reward=1.0,
    reward_w_on_lane = 0,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,

    speed_to_cal_time_limit = 4.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
    out_of_road_done=True,
    on_continuous_line_done=True,
    on_broken_line_done=False,
    crash_vehicle_done=True,
    crash_object_done=True,
    crash_human_done=True,


    debug = False,
)


class MetaDriveEnv(BaseEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = super(MetaDriveEnv, cls).default_config()
        config.update(METADRIVE_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self, config: Union[dict, None] = None):
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        super(MetaDriveEnv, self).__init__(config)

        # scenario setting
        self.start_seed = self.start_index = self.config["start_seed"]
        self.env_num = self.num_scenarios
        self.last_takeover_num = 0


        self.last_on_broken_line = 0
    def _post_process_config(self, config):
        config = super(MetaDriveEnv, self)._post_process_config(config)
        if not config["norm_pixel"]:
            self.logger.warning(
                "You have set norm_pixel = False, which means the observation will be uint8 values in [0, 255]. "
                "Please make sure you have parsed them later before feeding them to network!"
            )

        config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=self.default_config_copy
        )
        config["vehicle_config"]["norm_pixel"] = config["norm_pixel"]
        config["vehicle_config"]["random_agent_model"] = config["random_agent_model"]
        target_v_config = copy.deepcopy(config["vehicle_config"])
        if not config["is_multi_agent"]:
            target_v_config.update(config["agent_configs"][DEFAULT_AGENT])
            config["agent_configs"][DEFAULT_AGENT] = target_v_config
        return config

    def done_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        done = False
        max_step = self.config["horizon"] is not None and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        is_success = self._is_arrive_destination(vehicle)
        done_info = {
            TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
            TerminationState.CRASH_OBJECT: vehicle.crash_object,
            TerminationState.CRASH_BUILDING: vehicle.crash_building,
            TerminationState.CRASH_HUMAN: vehicle.crash_human,
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),
            TerminationState.SUCCESS: self._is_arrive_destination(vehicle),
            TerminationState.MAX_STEP: max_step,
            TerminationState.ENV_SEED: self.current_seed,
            # TerminationState.CURRENT_BLOCK: self.agent.navigation.current_road.block_ID(),
            # crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False,
            "is_success": is_success if self._is_arrive_destination(vehicle) else False,

        }

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info[TerminationState.CRASH] = (
                done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
                or done_info[TerminationState.CRASH_BUILDING] or done_info[TerminationState.CRASH_SIDEWALK]
                or done_info[TerminationState.CRASH_HUMAN]
        )

        # determine env return
        if done_info[TerminationState.SUCCESS]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: arrive_dest.".format(self.current_seed),
                extra={"log_once": True},
            )
        if done_info[TerminationState.OUT_OF_ROAD] and self.config["out_of_road_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: out_of_road.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash vehicle ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash object ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_BUILDING]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash building ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash human".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.MAX_STEP]:
            # single agent horizon has the same meaning as max_step_per_agent
            if self.config["truncate_as_terminate"]:
                done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: max step ".format(self.current_seed),
                extra={"log_once": True}
            )
        return done, done_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        return step_info['cost'], step_info

    @staticmethod
    def _is_arrive_destination(vehicle):
        """
        Args:
            vehicle: The BaseVehicle instance.

        Returns:
            flag: Whether this vehicle arrives its destination.
        """
        long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
        flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
                vehicle.navigation.get_current_lane_width() / 2 >= lat >=
                (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
        )
        return flag

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        # return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        ret = not vehicle.on_lane
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        elif self.config["on_continuous_line_done"]:
            ret = ret or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or vehicle.crash_sidewalk
        if self.config["on_broken_line_done"]:
            ret = ret or vehicle.on_broken_line
        return ret

    def get_navigation_len(self, vehicle):
        checkpoints = vehicle.navigation.checkpoints
        road_network = vehicle.navigation.map.road_network
        total_dist = 0
        assert len(checkpoints) >=2
        for check_num in range(0, len(checkpoints)-1):
            front_node = checkpoints[check_num]
            end_node = checkpoints[check_num+1]
            cur_lanes = road_network.graph[front_node][end_node]
            target_lane_num = int(len(cur_lanes) / 2)
            target_lane = cur_lanes[target_lane_num]
            target_lane_length = target_lane.length
            total_dist += target_lane_length
        return total_dist

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        reward = 0.0

        if self._compute_navi_dist:
            self.navi_distance = self.get_navigation_len(vehicle)
            self._compute_navi_dist = False

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        current_reference_lane = vehicle.lane
        heading_diff = vehicle.heading_diff(current_reference_lane)
        heading_factor = (1 - math.exp(-10 * (1 - heading_diff)))
        lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)

        no_shaking_reward = 0.0
        steer_diff = 0.0
        steering = abs(vehicle.steering / vehicle.MAX_STEERING)
        steering_last = clip((vehicle.last_current_action[1][0] + 1) / 2, 0.0, 1.0)
        steering_now = clip((vehicle.steering / vehicle.MAX_STEERING + 1) / 2, 0.0, 1.0)
        steer_diff = abs(steering_now - steering_last)
        speed = (vehicle.speed_km_h / vehicle.max_speed_km_h)
        progress = long_now - long_last

        # now_count = 0
        # other_v_info = None
        # other_v_info = self.get_single_observation().lidar_observe(vehicle)[:24]
        # vehicles_info = denormalize_lidar_data(other_v_info)
        v_s = vehicle.speed

        lane_width = vehicle.navigation.get_current_lane_width()
        # total_risk = compute_tuoyuan_risk(vehicles_info, v_s=v_s,
        #              lane_width=lane_width,v_l = vehicle.LENGTH,v_w=vehicle.WIDTH)
        # total_risk = compute_tuoyuan_risk(vehicles_info, v_s=v_s,lane_width=lane_width
        #                                   ,v_l = vehicle.LENGTH,v_w=vehicle.WIDTH)
        # total_risk = compute_combined_risk(vehicles_info, v_s=v_s, lane_width=lane_width
        #                                   , v_l=vehicle.LENGTH, v_w=vehicle.WIDTH)
        # reward -= 0.2 * total_risk
        # print("总风险",total_risk)


        # ==========out of road 风险
        if vehicle.dist_to_left_side < 0.5*lane_width:
            out_of_risk_l = 0.1/(((vehicle.dist_to_left_side-0.5*vehicle.WIDTH)/(0.5*lane_width))**2+1)
        else:
            out_of_risk_l = 0.0
        if vehicle.dist_to_right_side < 0.5*lane_width:
            out_of_risk_r = 0.1/(((vehicle.dist_to_right_side-0.5*vehicle.WIDTH)/(0.5*lane_width))**2+1)
        else:
            out_of_risk_r = 0.0
        reward -= 2 * (out_of_risk_l + out_of_risk_r)
        # print("左侧出界风险,右侧出界风险",out_of_risk_l,out_of_risk_r)


        if vehicle.dist_to_left_side >=0.5*lane_width and vehicle.dist_to_right_side >= 0.5*lane_width:
            reward += 0.02

         # ====================平稳驾驶
        reward += (1 - steer_diff) * 0.05
        reward -= (steer_diff ** 2) * 0.05

        # overtake_flag = False
        # #===================超车奖励
        # current_takeover_num = vehicle.get_overtake_num()
        # delta = current_takeover_num - self.last_takeover_num
        # if delta > 0 and not vehicle.crash_vehicle:
        #     reward += self.config["overtake_reward"] * delta
        #     print(f"🚀超车了，增长 {delta} 次")
        #     overtake_flag = True

        # self.last_takeover_num = current_takeover_num
        heading = 1/(abs(0.5-heading_diff)+1)

        reward += self.config["driving_reward"] * progress * positive_road
        reward += self.config["speed_reward"] * speed
        reward += self.config["heading_reward"] * heading
        reward -= 0.01 * heading



        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        elif vehicle.crash_sidewalk:
            reward = -self.config["crash_sidewalk_penalty"]
        elif self.env_step_num > self.navi_distance / self.config['speed_to_cal_time_limit'] * 10:
            reward = - self.config["run_out_of_time_penalty"]
        step_info["route_completion"] = vehicle.navigation.route_completion

        return reward, step_info


    def setup_engine(self):
        super(MetaDriveEnv, self).setup_engine()
        from metadrive.manager.traffic_manager import PGTrafficManager
        from metadrive.manager.pg_map_manager import PGMapManager
        from metadrive.manager.object_manager import TrafficObjectManager
        self.engine.register_manager("map_manager", PGMapManager())
        self.engine.register_manager("traffic_manager", PGTrafficManager())
        if abs(self.config["accident_prob"] - 0) > 1e-2:
            self.engine.register_manager("object_manager", TrafficObjectManager())


if __name__ == '__main__':
    if __name__ == '__main__':
        cfg = dict(
            num_scenarios=1,
            start_seed=1000,
            random_lane_width=True,
            random_lane_num=True,
            use_render=False,
            traffic_density=0.1
        )
        # 初始化环境
        env = MetaDriveEnv(cfg)
        # 其他代码...


    def _act(env, action):
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward)
        assert isinstance(info, dict)


    env = MetaDriveEnv(cfg)
    print("Environment Configuration Keys:")
    print(env.config.keys())

    try:
        # Step 1: Reset the environment and print obs
        obs, _ = env.reset()
        print("Observation after reset (obs):", obs)
        print("Type of obs:", type(obs))
        if hasattr(obs, "shape"):
            print("Shape of obs:", obs.shape)
        else:
            print("Obs does not have a shape attribute.")

        # Step 2: Print observation space and compare with obs
        print("Observation Space:", env.observation_space)
        print("Type of Observation Space:", type(env.observation_space))

        # Debugging assertion error
        if not env.observation_space.contains(obs):
            print("Obs is not contained in Observation Space.")
            print("Expected:", env.observation_space)
            print("Actual:", obs)
        else:
            print("Obs is valid within Observation Space.")

        # Assert obs matches the observation space
        assert env.observation_space.contains(obs), "Observation does not match the Observation Space!"

        # Step 3: Take random actions and test the loop
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])

        # Step 4: Check vehicle and print attributes
        vehicle = env.agents.get(DEFAULT_AGENT, None)
        if vehicle:
            print(f"Vehicle Speed (km/h): {vehicle.speed_km_h}")
            print(dir(vehicle.navigation))
            print(print(vehicle.max_speed_km_h))
            print(vehicle.get_overtake_num())



    finally:
        # Ensure the environment closes properly
        env.close()

