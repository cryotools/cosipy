import numpy as np
def mmm(input):
    print("max: ", max(input))
    print("min: ", min(input))
    print("mean: ", np.mean(input))

def where_index(input, arg):
    count = np.where(input > arg)
    quantity_condition= np.count_nonzero(count)
    print(count)
    print(quantity_condition)
    return count, quantity_condition