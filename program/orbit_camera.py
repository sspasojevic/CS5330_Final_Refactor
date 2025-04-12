from pyrr import Vector3, Matrix44
from math import radians, sin, cos

class OrbitCamera:
    def __init__(self, center:Vector3=Vector3([0.0, 0.0, 0.0]), radius:float=5.0, yaw:float=0.0, pitch:float=0.0):
        """Initializes an orbit camera to view the scene.

        Args:
            center (Vector3, optional): Sets the center point (world coordinates). Defaults to Vector3([0.0, 0.0, 0.0]).
            radius (float, optional): Sets the radius (zoom). Defaults to 5.0.
            yaw (float, optional): Sets the yaw angle. Defaults to 0.0.
            pitch (float, optional): Sets the pitch angle. Defaults to 0.0.
        """
        self.center = center
        self.radius = radius
        self.yaw = yaw
        self.pitch = self.clamp(pitch)
        self.up = Vector3([0.0, 1.0, 0.0])
        self.update_vectors()

    def update_vectors(self):
        """Updates the camera view vectors 
        """
        yaw_rad, pitch_rad = radians(self.yaw), radians(self.pitch)

        self.front = Vector3([
            cos(pitch_rad) * cos(yaw_rad),
            sin(pitch_rad),
            cos(pitch_rad) * sin(yaw_rad)
        ]).normalized

        self.right = self.front.cross(self.up).normalized
        self.position = self.center - self.front * self.radius

    def get_view_matrix(self) -> Matrix44:
        """Gets the 4x4 view matrix from the camera based off its position, the center (target), and up vectors

        Returns:
            Matrix44: A 4x4 view matrix
        """
        self.update_vectors()
        return Matrix44.look_at(
            self.position,
            self.center,
            self.up,
            dtype='f4'
        )
        
    def rotate(self, dy:float, dp:float)->None:
        """Rotates the camera along spherical coordinates (yaw, pitch)

        Args:
            dy (float): delta yaw
            dp (float): delta pitch
        """
        self.yaw += dy
        self.pitch = self.clamp(self.pitch + dp)

    def zoom(self, dr: float) -> None:
        """Adjusts the camera's radius to simulate zoom

        Args:
            dr (float): delta radius
        """
        self.radius = max(1.0, self.radius + dr)

    def pan(self, dx:float, dy:float, dz:float, speed=1.0) -> None:
        """Moves the camera and center along the world ground relative to the look angle

        Args:
            dx (float): delta x
            dy (float): delta y
            dz (float): delta z
            speed (float, optional): The speed to move. Defaults to 1.0.
        """
        self.update_vectors()

        forward = Vector3([self.front.x, 0.0, self.front.z]).normalized
        
        offset = (self.right * dx + forward * dz) * speed

        self.center += offset
        self.position += offset        

    def clamp(self, val:float, min_val:float = -89.0, max_val:float = 89.0) -> float:
        """Clamps an input value to a min/max range

        Args:
            val (float): the value to clamp
            min_val (float, optional): min allowable value. Defaults to -89.0.
            max_val (float, optional): max allowable value. Defaults to 89.0.

        Returns:
            float: the clamped value in range [min_val, max_val]
        """
        return max(min_val, min(max_val, val))
