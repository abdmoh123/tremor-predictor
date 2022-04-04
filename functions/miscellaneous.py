# validates boundary values of an array/list (prevents errors)
def check_bounds(l_bound, u_bound, rows):
    # ensures that the lower bound is smaller than the upper bound
    if l_bound >= u_bound:
        l_bound = u_bound - 1
    # checks if bounds are valid (prevents index out of bounds error)
    if (l_bound < 0) | (l_bound >= len(rows)):
        l_bound = 0
    if (u_bound > len(rows)) | (u_bound <= 0):
        u_bound = len(rows)
    return l_bound, u_bound
