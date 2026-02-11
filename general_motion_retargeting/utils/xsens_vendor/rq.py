from scipy.spatial.transform import Rotation as R

def get_str(_a):
    print("    " * (3)
    +"[\r\n"
    +"    " * (4)
    +str(_a[0])
    +",\r\n"
    +"    " * (4)
    +str(_a[1])
    +",\r\n"
    +"    " * (4)
    +str(_a[2])
    +",\r\n"
    +"    " * (4)
    +str(_a[3])
    +"\r\n"
    +"    " * (3)
    +"]")


roll_angle = 90
a = R.from_rotvec([roll_angle, 0, 0],degrees=True).as_quat(scalar_first=True)
get_str(a)
a = R.from_rotvec([roll_angle, -90, 0],degrees=True).as_quat(scalar_first=True)
get_str(a)
a = R.from_rotvec([-roll_angle, 0, 0],degrees=True).as_quat(scalar_first=True)
get_str(a)
a = R.from_rotvec([-roll_angle, 90, 0],degrees=True).as_quat(scalar_first=True)
get_str(a)
a = R.from_rotvec([-roll_angle, 90, 90],degrees=True).as_quat(scalar_first=True)
get_str(a)
b = R.from_quat([ -0.5, 0.5, 0.5, 0.5 ], scalar_first=True).as_rotvec(degrees=True)
print(b)
b = R.from_quat(
            [
                0.5,
                -0.5,
                -0.5,
                -0.5
            ], scalar_first=True).as_rotvec(degrees=True)
print(b)
# a = R.from_rotvec([0, 8, 0],degrees=True).as_quat(scalar_first=True)
# get_str(a)

# a = R.from_rotvec([0, 8, 0],degrees=True).as_quat(scalar_first=True)
# get_str(a)