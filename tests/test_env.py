import unittest
import gymnasium as gym
import numpy as np

class TestLiteSocNavGym(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('LiteSocNavGym-v0', debug=0, human_debug=False)

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertIsNotNone(obs)
        self.assertIsInstance(obs, np.ndarray)

    def test_step(self):
        obs, info = self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, done, truncated, info = self.env.step(action)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def tearDown(self):
        self.env.close()

if __name__ == '__main__':
    unittest.main()
