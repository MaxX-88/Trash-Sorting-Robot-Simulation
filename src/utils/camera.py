import numpy as np
import pybullet as p
import math

# top down view for overhead shots of the scene
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
        # fov so the floor plane exactly fills the frame
        fov = 2 * np.degrees(np.arctan((floor_plane_size / 2) / camera_position[2]))

        self._projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, self.near, self.far)

    def get_image(self):
        img_arr = p.getCameraImage(
            width=self._img_width,
            height=self._img_height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._projection_matrix
        )
        
        if img_arr is None:
            return np.zeros((self._img_height, self._img_width, 3), dtype=np.uint8)

        rgba = np.reshape(np.array(img_arr[2], dtype=np.uint8), (self._img_height, self._img_width, 4))
        return rgba[:, :, :3], img_arr

    # convert pixel coords back to world x,y on the floor plane
    def get_pixel_world_coords(self, pixel_x, pixel_y):
        u = pixel_x / self._img_width
        v = 1.0 - (pixel_y / self._img_height)
        world_y = (u * self._floor_plane_size) - self._floor_plane_size / 2
        world_x = -(v * self._floor_plane_size - self._floor_plane_size / 2)
        return [world_x, world_y]


# angled view for more natural looking renders
class PerspectiveCamera:
    def __init__(self, img_width, img_height, camera_position, floor_plane_size, yaw=90, pitch=-60, roll=30, fov=60, target_position=None):
        self._img_width = img_width
        self._img_height = img_height
        self._camera_position = camera_position
        self._floor_plane_size = floor_plane_size
        self._roll, self._pitch, self._yaw = roll, pitch, yaw
        self._fov = fov

        if target_position is not None:
            self._view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_position,
                cameraTargetPosition=target_position,
                cameraUpVector=[0, 0, 1]
            )
        else:
            target_position = self._calculate_target_from_angles()
            self._view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_position,
                cameraTargetPosition=target_position,
                cameraUpVector=[0, 0, 1]
            )

        aspect = img_width / img_height
        self.near, self.far = 0.1, 50
        self._projection_matrix = p.computeProjectionMatrixFOV(self._fov, aspect, self.near, self.far)
    
    # figure out where to point the camera from yaw and pitch angles
    def _calculate_target_from_angles(self, look_distance=5.0):
        yaw_rad = math.radians(self._yaw)
        pitch_rad = math.radians(self._pitch)

        target_x = self._camera_position[0] + look_distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        target_y = self._camera_position[1] + look_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        target_z = self._camera_position[2] + look_distance * math.sin(pitch_rad)
        
        return [target_x, target_y, target_z]

    def get_image(self):
        #gets camera image in np array
        img_arr = p.getCameraImage(
            width=self._img_width,
            height=self._img_height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._projection_matrix
        )
        
        if img_arr is None:
            return np.zeros((self._img_height, self._img_width, 3), dtype=np.uint8)
        
        rgba = np.reshape(np.array(img_arr[2], dtype=np.uint8), (self._img_height, self._img_width, 4))
        return rgba[:, :, :3], img_arr
