from metadrive.obs.state_obs import StateObservation
from metadrive.obs.observation_base import BaseObservation
import gymnasium as gym
import numpy as np


class EgoStateNavigationobservation(BaseObservation):
    def __init__(self, config):
        super(EgoStateNavigationobservation, self).__init__(config)
        self.state_obs = StateObservation(config)



    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)


    def observe(self, vehicle):

        state = self.state_observe(vehicle)
        return state.astype(np.float32)

    def state_observe(self, vehicle):
        return self.state_obs.observe(vehicle)

