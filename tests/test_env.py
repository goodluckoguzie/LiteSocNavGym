import unittest
import numpy as np
from socnav_env import SocNavEnv

class TestSocNavEnv(unittest.TestCase):
    def setUp(self):
        """Set up a new instance of the environment for each test."""
        self.env = SocNavEnv(debug=0, human_debug=False)

    def test_reset(self):
        """Test if the environment resets correctly."""
        obs, info = self.env.reset()
        self.assertIsNotNone(obs, "Observation is None after reset.")
        self.assertIsInstance(obs, np.ndarray, "Observation is not a NumPy array.")
        self.assertIsInstance(info, dict, "Info is not a dictionary.")

    def test_step(self):
        """Test the step function of the environment."""
        obs, info = self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, done, truncated, info = self.env.step(action)
        self.assertIsInstance(obs, np.ndarray, "Observation is not a NumPy array.")
        self.assertIsInstance(reward, (float, int), "Reward is not a float or int.")
        self.assertIsInstance(done, bool, "Done is not a boolean.")
        self.assertIsInstance(truncated, bool, "Truncated is not a boolean.")
        self.assertIsInstance(info, dict, "Info is not a dictionary.")

    def test_invalid_action(self):
        """Test if the environment handles invalid actions."""
        self.env.reset()
        with self.assertRaises(Exception, msg="Invalid action did not raise an exception."):
            self.env.step(None)  # Passing an invalid action

    def test_render(self):
        """Test if the render function works without errors."""
        self.env.reset()
        try:
            frame = self.env.render()
            self.assertIsInstance(frame, np.ndarray, "Render did not return a valid NumPy array.")
        except Exception as e:
            self.fail(f"Render raised an exception: {e}")

    def tearDown(self):
        """Ensure the environment is properly closed."""
        self.env.close()

if __name__ == '__main__':
    unittest.main()
