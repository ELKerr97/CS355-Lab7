# Import a library of functions called 'pygame'
import pygame
import math
import numpy as np

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Point3D:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        self.w = 1
        
class Line3D():
    
    def __init__(self, start, end):
        self.start = start
        self.end = end

def loadOBJ(filename):
    
    vertices = []
    indices = []
    lines = []
    
    f = open(filename, "r")
    for line in f:
        t = str.split(line)
        if not t:
            continue
        if t[0] == "v":
            vertices.append(Point3D(float(t[1]),float(t[2]),float(t[3])))
            
        if t[0] == "f":
            for i in range(1,len(t) - 1):
                index1 = int(str.split(t[i],"/")[0])
                index2 = int(str.split(t[i+1],"/")[0])
                indices.append((index1,index2))
            
    f.close()
    
    #Add faces as lines
    for index_pair in indices:
        index1 = index_pair[0]
        index2 = index_pair[1]
        lines.append(Line3D(vertices[index1 - 1],vertices[index2 - 1]))
        
    #Find duplicates
    duplicates = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            
            # Case 1 -> Starts match
            if line1.start.x == line2.start.x and line1.start.y == line2.start.y and line1.start.z == line2.start.z:
                if line1.end.x == line2.end.x and line1.end.y == line2.end.y and line1.end.z == line2.end.z:
                    duplicates.append(j)
            # Case 2 -> Start matches end
            if line1.start.x == line2.end.x and line1.start.y == line2.end.y and line1.start.z == line2.end.z:
                if line1.end.x == line2.start.x and line1.end.y == line2.start.y and line1.end.z == line2.start.z:
                    duplicates.append(j)
                    
    duplicates = list(set(duplicates))
    duplicates.sort()
    duplicates = duplicates[::-1]
    
    #Remove duplicates
    for j in range(len(duplicates)):
        del lines[duplicates[j]]
    
    return lines

def loadHouse():
    house = []
    #Floor
    house.append(Line3D(Point3D(-5, 0, -5), Point3D(5, 0, -5)))
    house.append(Line3D(Point3D(5, 0, -5), Point3D(5, 0, 5)))
    house.append(Line3D(Point3D(5, 0, 5), Point3D(-5, 0, 5)))
    house.append(Line3D(Point3D(-5, 0, 5), Point3D(-5, 0, -5)))
    #Ceiling
    house.append(Line3D(Point3D(-5, 5, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(5, 5, -5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(5, 5, 5), Point3D(-5, 5, 5)))
    house.append(Line3D(Point3D(-5, 5, 5), Point3D(-5, 5, -5)))
    #Walls
    house.append(Line3D(Point3D(-5, 0, -5), Point3D(-5, 5, -5)))
    house.append(Line3D(Point3D(5, 0, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(5, 0, 5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(-5, 0, 5), Point3D(-5, 5, 5)))
    #Door
    house.append(Line3D(Point3D(-1, 0, 5), Point3D(-1, 3, 5)))
    house.append(Line3D(Point3D(-1, 3, 5), Point3D(1, 3, 5)))
    house.append(Line3D(Point3D(1, 3, 5), Point3D(1, 0, 5)))
    #Roof
    house.append(Line3D(Point3D(-5, 5, -5), Point3D(0, 8, -5)))
    house.append(Line3D(Point3D(0, 8, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(-5, 5, 5), Point3D(0, 8, 5)))
    house.append(Line3D(Point3D(0, 8, 5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(0, 8, 5), Point3D(0, 8, -5)))
    
    return house

def loadCar():
    car = []
    #Front Side
    car.append(Line3D(Point3D(-3, 2, 2), Point3D(-2, 3, 2)))
    car.append(Line3D(Point3D(-2, 3, 2), Point3D(2, 3, 2)))
    car.append(Line3D(Point3D(2, 3, 2), Point3D(3, 2, 2)))
    car.append(Line3D(Point3D(3, 2, 2), Point3D(3, 1, 2)))
    car.append(Line3D(Point3D(3, 1, 2), Point3D(-3, 1, 2)))
    car.append(Line3D(Point3D(-3, 1, 2), Point3D(-3, 2, 2)))

    #Back Side
    car.append(Line3D(Point3D(-3, 2, -2), Point3D(-2, 3, -2)))
    car.append(Line3D(Point3D(-2, 3, -2), Point3D(2, 3, -2)))
    car.append(Line3D(Point3D(2, 3, -2), Point3D(3, 2, -2)))
    car.append(Line3D(Point3D(3, 2, -2), Point3D(3, 1, -2)))
    car.append(Line3D(Point3D(3, 1, -2), Point3D(-3, 1, -2)))
    car.append(Line3D(Point3D(-3, 1, -2), Point3D(-3, 2, -2)))
    
    #Connectors
    car.append(Line3D(Point3D(-3, 2, 2), Point3D(-3, 2, -2)))
    car.append(Line3D(Point3D(-2, 3, 2), Point3D(-2, 3, -2)))
    car.append(Line3D(Point3D(2, 3, 2), Point3D(2, 3, -2)))
    car.append(Line3D(Point3D(3, 2, 2), Point3D(3, 2, -2)))
    car.append(Line3D(Point3D(3, 1, 2), Point3D(3, 1, -2)))
    car.append(Line3D(Point3D(-3, 1, 2), Point3D(-3, 1, -2)))

    return car

def loadTire():
    tire = []
    #Front Side
    tire.append(Line3D(Point3D(-1, .5, .5), Point3D(-.5, 1, .5)))
    tire.append(Line3D(Point3D(-.5, 1, .5), Point3D(.5, 1, .5)))
    tire.append(Line3D(Point3D(.5, 1, .5), Point3D(1, .5, .5)))
    tire.append(Line3D(Point3D(1, .5, .5), Point3D(1, -.5, .5)))
    tire.append(Line3D(Point3D(1, -.5, .5), Point3D(.5, -1, .5)))
    tire.append(Line3D(Point3D(.5, -1, .5), Point3D(-.5, -1, .5)))
    tire.append(Line3D(Point3D(-.5, -1, .5), Point3D(-1, -.5, .5)))
    tire.append(Line3D(Point3D(-1, -.5, .5), Point3D(-1, .5, .5)))

    #Back Side
    tire.append(Line3D(Point3D(-1, .5, -.5), Point3D(-.5, 1, -.5)))
    tire.append(Line3D(Point3D(-.5, 1, -.5), Point3D(.5, 1, -.5)))
    tire.append(Line3D(Point3D(.5, 1, -.5), Point3D(1, .5, -.5)))
    tire.append(Line3D(Point3D(1, .5, -.5), Point3D(1, -.5, -.5)))
    tire.append(Line3D(Point3D(1, -.5, -.5), Point3D(.5, -1, -.5)))
    tire.append(Line3D(Point3D(.5, -1, -.5), Point3D(-.5, -1, -.5)))
    tire.append(Line3D(Point3D(-.5, -1, -.5), Point3D(-1, -.5, -.5)))
    tire.append(Line3D(Point3D(-1, -.5, -.5), Point3D(-1, .5, -.5)))

    #Connectors
    tire.append(Line3D(Point3D(-1, .5, .5), Point3D(-1, .5, -.5)))
    tire.append(Line3D(Point3D(-.5, 1, .5), Point3D(-.5, 1, -.5)))
    tire.append(Line3D(Point3D(.5, 1, .5), Point3D(.5, 1, -.5)))
    tire.append(Line3D(Point3D(1, .5, .5), Point3D(1, .5, -.5)))
    tire.append(Line3D(Point3D(1, -.5, .5), Point3D(1, -.5, -.5)))
    tire.append(Line3D(Point3D(.5, -1, .5), Point3D(.5, -1, -.5)))
    tire.append(Line3D(Point3D(-.5, -1, .5), Point3D(-.5, -1, -.5)))
    tire.append(Line3D(Point3D(-1, -.5, .5), Point3D(-1, -.5, -.5)))
    
    return tire

    
# Initialize the game engine
pygame.init()
 
# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

# Set the height and width of the screen
size = [512, 512]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Shape Drawing")
 
#Set needed variables
done = False
clock = pygame.time.Clock()
start = Point(0.0,0.0)
end = Point(0.0,0.0)
cam_x, cam_y, cam_z, cam_angle = -40.0, 15.0, -25.0, -48.0
look_x, look_y, look_z = 0, 0, 1
up = np.array([0,1,0])
car_x, car_y, car_z = 0.0, 0.0, 0.0
wheel_x_offset, wheel_rot_offset = 0.0, 0.0
move_speed = 1.0
rotate_speed = 1.0
house_lines = []

# Load all houses to house
def create_house_transformation_m(x,y,z,angle):
    translation = np.array([[1, 0, 0, x],
                            [0, 1, 0, y],
                            [0, 0, 1, z],
                            [0, 0, 0, 1]])
    
    rad = math.radians(angle)

    rotation = np.array([[math.cos(rad), 0, -math.sin(rad), 0],
                         [0,             1, 0,              0],
                         [math.sin(rad), 0, math.cos(rad),  0],
                         [0,             0, 0,              1]])
    
    return translation @ rotation

def create_house(x, y, z, angle, lines):
    house = loadHouse()
    trans_matrix = create_house_transformation_m(x, y, z, angle)

    for line in house:
        line.start = trans_matrix @ np.array([line.start.x, line.start.y, line.start.z, 1])
        line.end = trans_matrix @ np.array([line.end.x, line.end.y, line.end.z, 1])
        lines.append(Line3D(Point3D(line.start[0], line.start[1], line.start[2]), Point3D(line.end[0], line.end[1], line.end[2])))

# Add houses
create_house(0, 0, 0, 0, house_lines)
create_house(20, 0, 0, 0, house_lines)
create_house(40, 0, 0, 0, house_lines)
create_house(0, 0, 40, 180, house_lines)
create_house(20, 0, 40, 180, house_lines)
create_house(40, 0, 40, 180, house_lines)

# Add car lines
def create_car(x, y, z, lines):
    car = loadCar()
    trans_matrix = np.array([[1, 0, 0, x],
                            [0, 1, 0, y],
                            [0, 0, 1, z],
                            [0, 0, 0, 1]])
    
    for line in car:
        line.start = trans_matrix @ np.array([line.start.x, line.start.y, line.start.z, 1])
        line.end = trans_matrix @ np.array([line.end.x, line.end.y, line.end.z, 1])
        lines.append(Line3D(Point3D(line.start[0], line.start[1], line.start[2]), Point3D(line.end[0], line.end[1], line.end[2])))

create_car(0, 0, 10, house_lines)


def magnitude(vector): 
    return math.sqrt(sum(pow(element, 2) for element in vector))

def world_to_camera_matrix(cam_x, cam_y, cam_z, look_x, look_y, look_z):
    translation = np.array([[1, 0, 0, -cam_x],
                            [0, 1, 0, -cam_y],
                            [0, 0, 1, -cam_z],
                            [0, 0, 0, 1]])
    
    fr = np.array([cam_x, cam_y, cam_z])
    at = np.array([cam_x + look_x, cam_y, cam_z + look_z])
    
    z = np.subtract(at, fr)/(magnitude(np.subtract(at,fr)))
    x = np.cross(z,up)/magnitude(np.cross(z,up))
    y = np.cross(x,z)/magnitude(np.cross(x,z))
    
    rotation = np.array([[x[0], x[1], x[2], 0],
                         [y[0], y[1], y[2], 0],
                         [z[0], z[1], z[2], 0],
                         [0,    0,    0,    1]])
    
    return rotation @ translation

def clip_matrix(fov):
    rad = math.radians(fov)
    zoomx = 1.0 / math.tan(rad/2)
    ar = size[0]/size[1]
    zoomy = ar * zoomx
    n = 0.1
    f = 300.0

    cmatrix = np.array([[zoomx, 0, 0, 0],
                        [0, zoomy, 0, 0],
                        [0, 0, (f+n)/(f-n), (-2*n*f)/(f-n)],
                        [0, 0, 1, 0]])
    return cmatrix

def should_clip(point):
    w = point[3]
    for p in point[:3]:
        if p <= -w or p >= w:
            return True
    return False

def viewport_transf_m(w, h):
    return np.array([[w/2, 0, w/2],
                    [0, -h/2, h/2],
                    [0, 0, 1]])

# Generate matrices
fov = 45
clip_m = clip_matrix(fov) # clip matrix
w_to_c_m = world_to_camera_matrix(cam_x, cam_y, cam_z, look_x, look_y, look_z) # world to camera matrix
width, height = size[0], size[1]
viewport_transf = viewport_transf_m(width, height)

#Loop until the user clicks the close button.
while not done:
 
    # This limits the while loop to a max of 100 times per second.
    # Leave this out and we will use all CPU we can.
    clock.tick(100)

    # Clear the screen and set the screen background
    screen.fill(BLACK)

    #Controller Code#
    #####################################################################

    for event in pygame.event.get():
        if event.type == pygame.QUIT: # If user clicked close
            done=True
            
    pressed = pygame.key.get_pressed()

    if pressed[pygame.K_w]:
        # Move forward
        cam_x += move_speed * math.sin(math.radians(-cam_angle))
        cam_z += move_speed * math.cos(math.radians(-cam_angle))

    elif pressed[pygame.K_a]:
        # Move left
        cam_x += move_speed * math.cos(math.radians(-cam_angle))
        cam_z -= move_speed * math.sin(math.radians(-cam_angle))

    elif pressed[pygame.K_s]:
        # Move back
        cam_x -= move_speed * math.sin(math.radians(-cam_angle))
        cam_z -= move_speed * math.cos(math.radians(-cam_angle))

    elif pressed[pygame.K_d]:
        # Move right
        cam_x -= move_speed * math.cos(math.radians(-cam_angle))
        cam_z += move_speed * math.sin(math.radians(-cam_angle))

    elif pressed[pygame.K_r]:
        # Move up
        cam_y += move_speed
    
    elif pressed[pygame.K_f]:
        # Move down
        cam_y -= move_speed

    elif pressed[pygame.K_q]:
        # Look left
        cam_angle -= rotate_speed

    elif pressed[pygame.K_e]:
        # Look right
        cam_angle += rotate_speed

    elif pressed[pygame.K_h]:
        cam_x, cam_y, cam_z, cam_angle = -40.0, 10.0, -25.0, -48.0

    #Viewer Code#
    #####################################################################

    look_x = math.sin(math.radians(-cam_angle))
    look_z = math.cos(math.radians(-cam_angle)) 
    w_to_c_m = world_to_camera_matrix(cam_x, cam_y, cam_z, look_x, look_y, look_z) # world to camera matrix

    for s in house_lines:
        # Apply world-to-camera transformation
        w_to_c_start = w_to_c_m @ np.array([s.start.x, s.start.y, s.start.z, s.start.w])
        w_to_c_end = w_to_c_m @ np.array([s.end.x, s.end.y, s.end.z, s.end.w])

        # Apply clip matrix
        clip_start = clip_m @ w_to_c_start
        clip_end = clip_m @ w_to_c_end

        # Clipping test
        # TODO: Add near plane test
        if should_clip(clip_start) and should_clip(clip_end):
            continue
        
        # Normalize the points (divide by w)
        clip_start = clip_start / clip_start[3]
        clip_end = clip_end / clip_end[3]

        st = viewport_transf @ np.array([clip_start[0], clip_start[1], 1])
        en = viewport_transf @ np.array([clip_end[0], clip_end[1], 1])

        #BOGUS DRAWING PARAMETERS SO YOU CAN SEE THE HOUSE WHEN YOU START UP
        pygame.draw.line(screen, BLUE, (st[0], st[1]), (en[0], en[1]))

    # Add houses and car

    # Go ahead and update the screen with what we've drawn.
    # This MUST happen after all the other drawing commands.
    pygame.display.flip()

# Be IDLE friendly
pygame.quit()