import numpy as np 
import matplotlib.pyplot as plt  
from scipy.optimize import curve_fit  
from scipy.stats import norm

def gaussian(x, amplitude, mean, std_dev):                          
    """
    Gaussian function based on amplitude, mean, and standard deviation.
    
    Parameters:
    x (numpy.ndarray): Mirror positions.
    amplitude (float): The height of the Gaussian peak.
    mean (float): The center of the peak.
    std_dev (float): The standard deviation.
    
    Returns:
    numpy.ndarray: Gaussian values at each x.
    """
    # Calculate gaussian. Normalize pdf by np.sqrt(2 * np.pi) and std_dev. This ensures peak = amplitude.
    return amplitude * np.sqrt(2 * np.pi) * std_dev * norm.pdf(x, mean, std_dev)  

def measure_ion_response(pos):
    """
    Gets the reading of an ion response for a given mirror position. (Just a stub for now).

    Parameters:
    pos (float): Mirror position.
    """
    pass

def move_mirror_to_position(pos):
    """
    Placeholder function to simulate moving a mirror to a specific position. (Just a stub for now).
    
    Parameters:
    pos (float): Mirror position.
    """
    pass

def locate_best_mirror_pos(start=0, stop=1, step_size=0.1, precision=0.001, amp_min=80, min_step_size=0.0001):
    """
    Locates the peak of the ion response by iteratively measuring and fitting a Gaussian.

    Parameters:
    start (float, optional): Starting mirror position for the scan. Default is 0.
    stop (float, optional): Ending mirror position for the scan. Default is 1.
    step_size (float, optional): Initial step size. Default is 0.1.
    precision (float, optional): Desired precision for locating the optimal mirror position. Default is 0.001.
    amp_min (float, optional): Minimum acceptable amplitude of the Gaussian fit. Default is 80.
    min_step_size (float, optional): Minimum step size allowed during search. Default is 0.0005.

    Returns:
    float: Estimated position of the optimal mirror position with required precision, or None if the step size is too small.
    """

    # Check for valid search range
    if not (0 <= start <= 1) or not (0 <= stop <= 1):
        print("Error: Start and stop values must be within the range [0, 1].")
        return None

    # Check if start is less than stop
    if start >= stop:
        print("Error: Start value must be less than stop value.")
        return None
    
    # Check for valid step size
    if step_size <= min_step_size or step_size >= (stop-start)/3: 
        print("Error: Step size must be a positive value larger than min_step_size, and less than a 1/3 of the search range.")
        return None
    
    

    def try_gaussian_fit(x_data, y_data):
        """
        Attempts to fit Gaussian to the ion response data.

        Parameters:
        x_data (numpy.ndarray): Array of x mirror positions.
        y_data (numpy.ndarray): Array of ion response measurements.

        Returns:
        tuple or None: Optimal parameters of the Gaussian fit (amplitude, mean, std_dev) if successful, None otherwise.
        """
        
        if len(x_data) < 3:
            return None
        try:
            popt, _ = curve_fit(gaussian, x_data, y_data,   # Uses non-linear least squares to fit a gaussian function to the data. It allows us to approximate data points that have not been physically measured. 
                                p0=[100, np.mean(x_data), np.std(x_data)], 
                                bounds=([0, 0, 0],[100, 1, np.inf]))  
            return popt
        except RuntimeError:
            return None

    # Dictionary to store all of the data from searches
    ion_responses = {}
    # STEP 1: Initial search to find a valid fit using initial step size
    while True:

        search_direction = 1  # (1 for left to right), (-1 for right to left)

        # Define search direction
        if search_direction == 1:
            x_vals = np.clip(np.arange(start, stop, step_size), 0, 1) # Right to left
        else:
            x_vals = np.clip(np.arange(stop, start, -step_size), 0, 1)  # Left to right

        # Gather ion response data over mirror positions
        for pos in x_vals:
            if pos not in ion_responses:  # Only measure new positions
                move_mirror_to_position(pos)
                ion_response = measure_ion_response(pos)
                ion_responses[pos] = ion_response

        x_data = np.array(list(ion_responses.keys()))
        y_data = np.array(list(ion_responses.values()))

        # Attempt to find a gaussian fit to the ion response data
        popt = try_gaussian_fit(x_data, y_data)

        # If we find a valid fit, break and move to STEP 2
        if popt is not None and amp_min < popt[0] <= 100:
            break

        # End of current sweep. Switch the search direction for the next iteration
        search_direction *= -1

        # Expand the search range if we haven't founded a valid fit with the current search parameters
        max_pos = max(ion_responses, key=ion_responses.get)
        range_width = stop - start
        start = max(0, max_pos - 1.25 * range_width)
        stop = min(1, max_pos + 1.25 * range_width)

        # If search range reaches entire space, halve the step size
        if start == 0 and stop == 1:
            step_size /= 2
        if step_size <= min_step_size:
            print(f"Error: Step size became smaller than min_step_size, {min_step_size}.")
            return None 
            
    # STEP 2: Narrow the search window around the peak of the gaussian fit
    amplitude, mean, stddev = popt
    narrow_search_range = 2 * stddev
    start = max(0, mean - narrow_search_range)
    stop = min(1, mean + narrow_search_range)
    # Set step size for narrow search
    step_size = precision

    # STEP 3: Refine search and validate fit and precision
    while True:

        direction = 1 # (1 for left to right), (-1 for right to left)

        if direction == 1:
            x_vals = np.clip(np.arange(start, stop, step_size), 0, 1) # Right to left
        else:
            x_vals = np.clip(np.arange(start, stop, -step_size), 0, 1) # Left to right

        for pos in x_vals:
            if pos not in ion_responses:  # Only measure new positions
                move_mirror_to_position(pos)
                ion_response = measure_ion_response(pos)
                ion_responses[pos] = ion_response

        x_data = np.array(list(ion_responses.keys()))
        y_data = np.array(list(ion_responses.values()))
        
        popt = try_gaussian_fit(x_data, y_data)

        # Check if we have a valid fit
        if popt is not None:
            amplitude, mean, stddev = popt
            if amp_min < amplitude <= 100 and step_size < precision:                                     
                plot_results(x_data, y_data, popt)
                return mean

        # If valid fit condition or precision conditon not met, halve the step size
        step_size /= 2
        if step_size <= min_step_size:
            print(f"Error: Step size became smaller than min_step_size, {min_step_size}.")
            return None
        
def plot_results(x_data, y_data, popt):
    """
    Plots the measured ion response data and the fitted Gaussian surface.

    Parameters:
    x_data (list): The x mirror positions.
    y_data (list): The y mirror positions.
    z_data (list): The ion responses.
    popt (numpy.ndarray): Optimal parameters of the Gaussian fit (amplitude, x_mean, x_stddev).

    Returns:
    None
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(x_data, y_data, label='Measured Data') 
    
    # Generate a line for the gaussian fit
    x_fit = np.linspace(0, 1, 1000) 
    y_fit = gaussian(x_fit, *popt) 
    plt.plot(x_fit, y_fit, color='red', label='Gaussian Fit')

    mean = popt[1] # Grab the mean from the optimal fit parameters
    plt.axvline(mean, color='green', linestyle='--', label=f"Optimal Mirr. Position = {mean:0.4f}")

    plt.title('Ion Response vs Mirror Position')  
    plt.xlabel('Mirror Position (unitless)')
    plt.ylabel('Ion Response (photons/measurement round)')  
    plt.xlim(0,1)
    plt.ylim(0,100)
    plt.legend()  
    plt.show()  
