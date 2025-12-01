import numpy as np
import pybullet as p
import math

# ----- Camera Class -----
class TopDownCamera:
    def __init__(self, img_width, img_height, camera_position, floor_plane_size, target_position=None):
        self._img_width = img_width
        self._img_height = img_height
        self._camera_position = camera_position
        self._floor_plane_size = floor_plane_size
        self._roll, self._pitch, self._yaw = 0, -90, 90

        if target_position is not None:
            self._view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_position,
                cameraTargetPosition=target_position,
                cameraUpVector=[0, 0, 1]
            )
        else:
            target = camera_position.copy()
            target[2] = 0
            self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target,
                distance=camera_position[2],
                yaw=self._yaw,
                pitch=self._pitch,
                roll=self._roll,
                upAxisIndex=2
            )

        aspect = img_width / img_height
        self.near, self.far = 0.1, 10
        fov = 2 * np.degrees(np.arctan((floor_plane_size / 2) / camera_position[2]))

        self._projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, self.near, self.far)

    def get_image(self):
        img_arr = p.getCameraImage(
            width=self._img_width,
            height=self._img_height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._projection_matrix
        )
        
        # Handle case where getCameraImage returns None
        if img_arr is None:
            # Return a black image as fallback
            return np.zeros((self._img_height, self._img_width, 3), dtype=np.uint8)
        
        rgba = np.reshape(np.array(img_arr[2], dtype=np.uint8), (self._img_height, self._img_width, 4))
        return rgba[:, :, :3], img_arr

    def get_pixel_world_coords(self, pixel_x, pixel_y):
        u = pixel_x / self._img_width
        v = 1.0 - (pixel_y / self._img_height)
        world_y = (u * self._floor_plane_size) - self._floor_plane_size / 2
        world_x = -(v * self._floor_plane_size - self._floor_plane_size / 2)
        #world_z = img_arr[3]
        return [world_x, world_y] 

class PerspectiveCamera:
    def __init__(self, img_width, img_height, camera_position, floor_plane_size, yaw=90, pitch=-60, roll=30, fov=60, target_position=None):
        self._img_width = img_width
        self._img_height = img_height
        self._camera_position = camera_position
        self._floor_plane_size = floor_plane_size
        self._roll, self._pitch, self._yaw = roll, pitch, yaw
        self._fov = fov

        if target_position is not None:
            # Use explicit target position if provided
            self._view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_position,
                cameraTargetPosition=target_position,
                cameraUpVector=[0, 0, 1]
            )
        else:
            # Calculate target position based on camera position and angles
            target_position = self._calculate_target_from_angles()
            self._view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_position,
                cameraTargetPosition=target_position,
                cameraUpVector=[0, 0, 1]
            )

        aspect = img_width / img_height
        self.near, self.far = 0.1, 50
        self._projection_matrix = p.computeProjectionMatrixFOV(self._fov, aspect, self.near, self.far)
    
    def _calculate_target_from_angles(self, look_distance=5.0):
        """
        calculate where camera should look based on yaw/pitch/roll
        """
        # Convert angles to radians
        yaw_rad = math.radians(self._yaw)
        pitch_rad = math.radians(self._pitch)
        
        # Calculate the direction vector from the camera
        # Standard camera coordinate system:
        # - Positive yaw rotates left (counter-clockwise when viewed from above)
        # - Positive pitch tilts up
        # - We look in the negative Z direction by default
        
        # Calculate target position
        target_x = self._camera_position[0] + look_distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        target_y = self._camera_position[1] + look_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        target_z = self._camera_position[2] + look_distance * math.sin(pitch_rad)
        
        return [target_x, target_y, target_z]

    def get_image(self):
        img_arr = p.getCameraImage(
            width=self._img_width,
            height=self._img_height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._projection_matrix
        )
        
        # Handle case where getCameraImage returns None
        if img_arr is None:
            # Return a black image as fallback
            return np.zeros((self._img_height, self._img_width, 3), dtype=np.uint8)
        
        rgba = np.reshape(np.array(img_arr[2], dtype=np.uint8), (self._img_height, self._img_width, 4))
        return rgba[:, :, :3], img_arr

