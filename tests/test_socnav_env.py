import unittest
import numpy as np
from socnav_env import LiteSocNavGym

class TestLiteSocNavGym(unittest.TestCase):
    def setUp(self):
        """Set up the environment for testing."""
        self.env = LiteSocNavGym(debug=0, human_debug=False)

    def test_reset(self):
        """Test if the environment resets correctly."""
        obs, info = self.env.reset()
        self.assertIsNotNone(obs, "Observation is None after reset.")
        self.assertIsInstance(obs, np.ndarray, "Observation is not a NumPy array.")
        self.assertIsInstance(info, dict, "Info is not a dictionary.")
        self.assertGreater(len(obs), 0, "Observation array is empty.")

    def test_step(self):
        """Test the step function of the environment."""
        self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, done, truncated, info = self.env.step(action)
        self.assertIsInstance(obs, np.ndarray, "Observation is not a NumPy array.")
        self.assertIsInstance(reward, float, "Reward is not a float.")
        self.assertIsInstance(done, bool, "Done is not a boolean.")
        self.assertIsInstance(truncated, bool, "Truncated is not a boolean.")
        self.assertIsInstance(info, dict, "Info is not a dictionary.")

    def test_invalid_action(self):
        """Test if the environment handles invalid actions."""
        self.env.reset()
        with self.assertRaises(Exception):
            self.env.step(None)  # Passing an invalid action

    def test_render(self):
        """Test the rendering functionality."""
        self.env.reset()
        frame = self.env.render()
        self.assertIsInstance(frame, np.ndarray, "Rendered frame is not a NumPy array.")
        self.assertEqual(len(frame.shape), 3, "Rendered frame does not have 3 dimensions.")
        self.assertGreater(frame.shape[0], 0, "Rendered frame height is zero.")
        self.assertGreater(frame.shape[1], 0, "Rendered frame width is zero.")

    def tearDown(self):
        """Ensure the environment is properly closed."""
        self.env.close()

if __name__ == "__main__":
    unittest.main()
