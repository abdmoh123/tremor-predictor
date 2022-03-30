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


# iterates through list to find the highest value
def find_highest(features, magnitude=False):
    max_value = 0
    # can find the largest magnitude or the most positive value
    if magnitude:
        for value in features:
            if abs(value) > max_value:
                max_value = abs(value)
    else:
        for value in features:
            if value > max_value:
                max_value = value
    return max_value


def find_lowest(features, magnitude=False):
    min_value = features[0]
    # can find the smallest magnitude or the most negative value
    if magnitude:
        for value in features:
            if abs(value) < abs(min_value):
                min_value = abs(value)
    else:
        for value in features:
            if value < min_value:
                min_value = value
    return min_value
