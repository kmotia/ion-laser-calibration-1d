def calibrate_mirror_to_response(pos):  # add synthetic data here?
    pass

def locate_best_mirror_pos():
    # STEP 1: Initial search to find a valid fit using initial step size
    # STEP 2: Refine search and validate fit and precision
    # Return data

def plotting_function()

# Do a grid search? Possibly gradient descent. However, gradient descent would prob be fucky around the edges of the scan space...
# Let's start with a grid search. 
# Scan in one direction? Let's just scan in both directions. It's prob pretty easy to implement. Possibly make the scanning steps negative?
#   - Test it first
# 