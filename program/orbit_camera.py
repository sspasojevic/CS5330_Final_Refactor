"""
Sara Spasojevic, Matej Zecic, Benjamin Northrop
3D Object Manipulation via Hand Gesture Recognition
CS5330 Final Project
April 20th 2025

OrbitCamera Module
-----------------
This module provides a camera system for 3D applications that orbits around a target position.
It implements spherical coordinate control for camera movement including rotation, zoom, and panning.

The OrbitCamera class maintains the camera's position based on spherical coordinates (radius, yaw, pitch)
relative to a center point, providing an intuitive control system for viewing 3D scenes.

This module is designed for use in OpenGL or other 3D graphics applications that use a view matrix
for rendering from the camera's perspective.
"""

# Imports
from pyrr import Vector3, Matrix44
from math import radians, sin, cos

class OrbitCamera:
    """
        A camera that orbits around a center point in 3D space.
        
        The camera position is determined by spherical coordinates (radius, yaw, pitch)
        relative to a center point. This provides an intuitive way to orbit around and view
        3D objects from different angles. The camera supports rotation, zooming, and panning operations.
        
        Attributes:
            center (Vector3): The point in world space that the camera orbits around
            radius (float): Distance from the center point (controls zoom level)
            yaw (float): Horizontal rotation angle in degrees
            pitch (float): Vertical rotation angle in degrees (clamped to avoid gimbal lock)
            up (Vector3): The up direction vector for the camera (typically world up)
            front (Vector3): The direction vector that the camera is looking towards
            right (Vector3): The right direction vector relative to the camera orientation
            position (Vector3): The calculated position of the camera in world space
        """
    def __init__(self, center:Vector3=Vector3([0.0, 0.0, 0.0]), radius:float=5.0, yaw:float=0.0, pitch:float=0.0):
        """
        Initializes an orbit camera to view the scene.

        Args:
            center (Vector3, optional): Sets the center point (world coordinates). Defaults to Vector3([0.0, 0.0, 0.0]).
            radius (float, optional): Sets the radius (zoom). Defaults to 5.0.
            yaw (float, optional): Sets the yaw angle. Defaults to 0.0.
            pitch (float, optional): Sets the pitch angle. Defaults to 0.0.
        """
        
        # Initialize camera parameters
        self.center = center  # Center point to orbit around
        self.radius = radius  # Distance from center (zoom level)
        self.yaw = yaw        # Horizontal rotation angle
        self.pitch = self.clamp(pitch)  # Vertical rotation angle (clamped)
        self.up = Vector3([0.0, 1.0, 0.0])  # World up vector
        
        # Calculate initial camera vectors
        self.update_vectors()

    def update_vectors(self):
        """
        Updates the camera view vectors based on current yaw and pitch.
        
        Recalculates the front direction vector based on spherical coordinates,
        then derives the right vector and camera position from it.
        """
        
        # Convert angles to radians for trigonometric calculations
        yaw_rad, pitch_rad = radians(self.yaw), radians(self.pitch)

        # Calculate front vector using spherical coordinate formulas
        self.front = Vector3([
            cos(pitch_rad) * cos(yaw_rad),
            sin(pitch_rad),
            cos(pitch_rad) * sin(yaw_rad)
        ]).normalized

        # Calculate right vector as cross product of front and up
        self.right = self.front.cross(self.up).normalized
        
        # Calculate camera position by moving from center along front vector
        self.position = self.center - self.front * self.radius

    def get_view_matrix(self) -> Matrix44:
        """
        Gets the 4x4 view matrix from the camera based on its position, target, and up vectors.
        
        Updates camera vectors before calculating the view matrix to ensure it reflects
        the current camera state.

        Returns:
            Matrix44: A 4x4 view matrix for rendering from this camera's perspective
        """
        
        # Ensure vectors are up to date
        self.update_vectors()
        
        # Create look-at matrix using pyrr library
        return Matrix44.look_at(
            self.position,  # Camera position
            self.center,    # Look target position
            self.up,        # Up direction
            dtype='f4'      # Float32 type for GPU compatibility
        )
        
    def rotate(self, dy:float, dp:float)->None:
        """
        Rotates the camera along spherical coordinates (yaw, pitch).
        
        Updates the camera's orientation by changing yaw (horizontal) and pitch (vertical)
        angles. The pitch is automatically clamped to prevent gimbal lock.

        Args:
            dy (float): delta yaw - change in horizontal angle in degrees
            dp (float): delta pitch - change in vertical angle in degrees
        """
        # Update yaw angle (horizontal rotation)
        self.yaw += dy
        
        # Update pitch angle (vertical rotation) with clamping
        self.pitch = self.clamp(self.pitch + dp)

    def zoom(self, dr: float) -> None:
        """
        Adjusts the camera's radius to simulate zoom.

        Args:
            dr (float): delta radius - change in distance from center
        """
        
        # Update radius with a minimum limit of 1.0
        self.radius = max(1.0, self.radius + dr)

    def pan(self, dx:float, dy:float, dz:float, speed=1.0) -> None:
        """
        Moves the camera and center along the world ground relative to the look angle.
        
        Panning maintains the relative orientation between camera and center while
        moving both in the specified direction.

        Args:
            dx (float): delta x - movement along right vector
            dy (float): delta y - not used in current implementation
            dz (float): delta z - movement along forward vector
            speed (float, optional): The speed modifier for movement. Defaults to 1.0.
        """
        
        # Ensure vectors are up to date
        self.update_vectors()

        # Calculate forward direction in xz-plane (ground plane)
        forward = Vector3([self.front.x, 0.0, self.front.z]).normalized
        
        # Calculate total offset using right and forward vectors
        offset = (self.right * dx + forward * dz) * speed

        # Apply offset to both center and position to maintain relative orientation
        self.center += offset
        self.position += offset        

    def clamp(self, val:float, min_val:float = -89.0, max_val:float = 89.0) -> float:
        """
        Clamps an input value to a min/max range.
        
        Primarily used to restrict pitch angle to prevent gimbal lock at poles.

        Args:
            val (float): the value to clamp
            min_val (float, optional): min allowable value. Defaults to -89.0.
            max_val (float, optional): max allowable value. Defaults to 89.0.

        Returns:
            float: the clamped value in range [min_val, max_val]
        """
        
        return max(min_val, min(max_val, val))
