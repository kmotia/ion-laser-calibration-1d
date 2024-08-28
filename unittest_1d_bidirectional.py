import unittest
from unittest.mock import patch
import numpy as np
from scipy.stats import norm
from optimization_1d_bidirectional import locate_best_mirror_pos

class TestLocatePeak(unittest.TestCase):
    precision_check = 0.001

####################################################### Success checks. Return optimal mirror position. #######################################################

    @patch('optimization_1d_bidirectional.measure_ion_response') 
    def test_center(self, mock_measure_ion_response):
        mock_measure_ion_response.side_effect = lambda pos: self.synthetic_response(pos, mean=0.5, amplitude=100, std_dev=0.01, noise_level=1)
        best_pos = locate_best_mirror_pos(start=0, stop=1, step_size=0.1, precision=self.precision_check)
        self.assertAlmostEqual(best_pos, 0.5, delta=self.precision_check, msg=f"Optimal mirror position {best_pos} out of expected range (0.5 ± {self.precision_check})")
       
    @patch('optimization_1d_bidirectional.measure_ion_response')
    def test_peak_zero(self, mock_measure_ion_response):
        mock_measure_ion_response.side_effect = lambda pos: self.synthetic_response(pos, mean=0.1, amplitude=100, std_dev=0.01, noise_level=1)
        best_pos = locate_best_mirror_pos(start=0, stop=1, step_size=0.1, precision=self.precision_check)
        self.assertAlmostEqual(best_pos, 0.1, delta=self.precision_check, msg=f"Optimal mirror position {best_pos} out of expected range (0.1 ± {self.precision_check})")

    @patch('optimization_1d_bidirectional.measure_ion_response')
    def test_peak_one(self, mock_measure_ion_response):
        mock_measure_ion_response.side_effect = lambda pos: self.synthetic_response(pos, mean=0.9, amplitude=100, std_dev=0.01, noise_level=1)
        best_pos = locate_best_mirror_pos(start=0, stop=1, step_size=0.1, precision=self.precision_check)
        self.assertAlmostEqual(best_pos, 0.9, delta=self.precision_check, msg=f"Optimal mirror position {best_pos} out of expected range (0.9 ± {self.precision_check})")

    @patch('optimization_1d_bidirectional.measure_ion_response')
    def test_peak_one_and_small_initial_search(self, mock_measure_ion_response):
        mock_measure_ion_response.side_effect = lambda pos: self.synthetic_response(pos, mean=0.9, amplitude=100, std_dev=0.01, noise_level=1)
        best_pos = locate_best_mirror_pos(start=0, stop=0.33, step_size=0.1, precision=self.precision_check)
        self.assertAlmostEqual(best_pos, 0.9, delta=self.precision_check, msg=f"Optimal mirror position {best_pos} out of expected range (0.9 ± {self.precision_check})")

    @patch('optimization_1d_bidirectional.measure_ion_response')
    def test_high_noise_level(self, mock_measure_ion_response):
        mock_measure_ion_response.side_effect = lambda pos: self.synthetic_response(pos, mean=0.5, amplitude=100, std_dev=0.01, noise_level=10)
        best_pos = locate_best_mirror_pos(start=0, stop=1, step_size=0.1, precision=self.precision_check)
        self.assertAlmostEqual(best_pos, 0.5, delta=self.precision_check, msg=f"Optimal mirror position {best_pos} out of expected range (0.9 ± {self.precision_check})")

####################################################### Failure checks. Return None. #######################################################

    @patch('optimization_1d_bidirectional.measure_ion_response')
    def test_stepsize_too_small(self, mock_measure_ion_response):
        mock_measure_ion_response.side_effect = lambda pos: self.synthetic_response(pos, mean=0.5, amplitude=100, std_dev=0.01, noise_level=1)
        best_pos = locate_best_mirror_pos(start=0, stop=1, step_size=0, precision=self.precision_check)
        self.assertIsNone(best_pos, f"Expected None, but got {best_pos}")

    @patch('optimization_1d_bidirectional.measure_ion_response')
    def test_stepsize_too_large(self, mock_measure_ion_response):
        mock_measure_ion_response.side_effect = lambda pos: self.synthetic_response(pos, mean=0.5, amplitude=100, std_dev=0.01, noise_level=1)
        best_pos = locate_best_mirror_pos(start=0, stop=1, step_size=0.4, precision=self.precision_check)
        self.assertIsNone(best_pos, f"Expected None, but got {best_pos}")

    @patch('optimization_1d_bidirectional.measure_ion_response')
    def test_start_too_small(self, mock_measure_ion_response):
        mock_measure_ion_response.side_effect = lambda pos: self.synthetic_response(pos, mean=0.5, amplitude=100, std_dev=0.01, noise_level=1)
        best_pos = locate_best_mirror_pos(start=-1, stop=1, step_size=0.01, precision=self.precision_check)
        self.assertIsNone(best_pos, f"Expected None, but got {best_pos}")

    @patch('optimization_1d_bidirectional.measure_ion_response')
    def test_stop_too_large(self, mock_measure_ion_response):
        mock_measure_ion_response.side_effect = lambda pos: self.synthetic_response(pos, mean=0.5, amplitude=100, std_dev=0.01, noise_level=1)
        best_pos = locate_best_mirror_pos(start=0, stop=2, step_size=0.01, precision=self.precision_check)
        self.assertIsNone(best_pos, f"Expected None, but got {best_pos}")

    @patch('optimization_1d_bidirectional.measure_ion_response')
    def test_no_peak(self, mock_measure_ion_response):
        mock_measure_ion_response.side_effect = lambda pos: self.synthetic_response(pos, mean=0.5, amplitude=0, std_dev=0.01, noise_level=1)
        best_pos = locate_best_mirror_pos(start=0, stop=1, step_size=0.1, precision=self.precision_check)
        self.assertIsNone(best_pos, f"Expected None, but got {best_pos}")

##############################################################################################################

    def synthetic_response(self, pos, mean=0.5, amplitude=100, std_dev=0.01, noise_level=1):
        """
        Generates synthetic ion response data based on a Gaussian function.
        This is a helper method to mock measure_ion_response.
        """
        gaussian_value = amplitude * np.sqrt(2 * np.pi) * std_dev * norm.pdf(pos, mean, std_dev)
        noisy_response = gaussian_value + np.random.normal(0, noise_level)
        clipped_response = np.clip(noisy_response, 0, amplitude)
        return int(round(clipped_response))

if __name__ == '__main__':
    unittest.main()
