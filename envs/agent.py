class Agent:
    def __init__(self):
        self.pos = None
        self.linear_vel = None # [min_vel, max_vel]
        self.angular_vel = None # [min_vel, max_vel]

    def get_pos(self):
        return self.pos

    def get_linear_vel(self):
        return self.linear_vel
    
    def get_angular_vel(self):
        return self.angular_vel

    def set_pos(self, pos):
        self.pos = pos

    def set_linear_vel(self, linear_vel):
        self.linear_vel = linear_vel
    
    def set_angular_vel(self, angular_vel):
        self.angular_vel = angular_vel