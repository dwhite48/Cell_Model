#Simple vector class
import math as math

class vector(object):

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_distance(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return math.sqrt(x**2 + y**2 + z**2)

    def mag(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __repr__(self):
        return "(" + repr(self.x) + ", " + repr(self.y) + ", " + repr(self.z) +")"

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        if(isinstance(other, vector)):
           return (self.x == other.x and self.y == other.y and self.z == other.z)
        return False

    def __add__(self, other):
        if(isinstance(other, vector)):
            return vector(self.x + other.x , self.y + other.y, self.z + other.z)
        else:
            raise ValueError()

    def __sub__(self, other):
        if(isinstance(other, vector)):
            return vector(self.x - other.x , self.y - other.y, self.z - other.z)
        else:
            raise ValueError()        

    def __mult__(self, other):
        if(isinstance(other, vector)):
            return vector(other.x * self.x, other.y * self.y, other.z * self.z)
        elif(isinstance(other, int)):
            return vector(other * self.x, other * self.y, other * self.z)
        elif(isinstance(other, float)):
            return vector(other * self.x, other * self.y, other * self.z)
        else:
            raise ValueError()    

    def __div__(self, other):
        if(isinstance(other, vector)):
            return vector(self.x / other.x , self.y / other.y, self.z / other.z)
        elif(isinstance(other, int)):
            return vector(other* self.x, other * self.y, other * self.z)
        elif(isinstance(other, float)):
            return vector(other* self.x, other * self.y, other * self.z)
        else:
            raise ValueError()

    def __neg__(self, other):
        if(isinstance(other, vector)):
            return vector(-self.x, -self.y, -self.z)
        
