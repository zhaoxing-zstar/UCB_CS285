import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        batch_szie = observation.shape[0]
        qa_values = self.critic.qa_values(observation)

        assert qa_values.shape[0] == batch_szie
        action = np.argmax(qa_values, axis = 1)
        return action.squeeze()