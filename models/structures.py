"""
Class ELE is the element object.
Class NOD is the node object.
"""


class ELE:
    def __init__(self):
        self.ctag = 0  # id of occupying cell, 0 if no cell
        self.prolif = 0  # indicator whether a cell has already proliferated in the current iteration


class NOD:
    def __init__(self):
        self.fx = 0.5  # force in x-direction
        self.fy = 0.5  # force in y-direction
        self.ux = 0.0  # displacement in x-direction
        self.uy = 0.0  # displacement in y-direction
        self.restrictx = False
        self.restricty = False

