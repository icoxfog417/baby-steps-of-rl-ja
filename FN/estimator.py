class Estimator():

    def __init__(self, actions):
        self.actions = actions
        self.model = None
        self.initialized = False

    def reset(self):
        self.model = None
        self.initialized = False

    def initialize(self, experiences):
        raise Exception("You have to implements initialize method")

    def estimate(self, s):
        raise Exception("You have to implements estimate method")

    def update(self, s, a, diff):
        raise Exception("You have to implements update method")
