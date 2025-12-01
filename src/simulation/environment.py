import pybullet as p
import pybullet_data
import math
from ultralytics import YOLO
from src.utils.camera import TopDownCamera, PerspectiveCamera
from src.control.pybullet_helpers import get_initial_joint_positions
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SimulationEnvironment:
    """
    handles the setup and management of the simulation environment
    """
    
    def __init__(self, config, headless=False):
        self.config = config
        self.setup_physics(headless)
        self.setup_environment()
        self.setup_robot()
        self.setup_cameras()
        self.setup_model()
    
    def setup_physics(self, headless=False):
        """initialize pybullet physics simulation"""
        if headless:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)

        # debug line from perspective camera to where it's looking (calculated from angles)
        # this will be calculated after the camera is created, so we'll add it after camera initialization
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity)
        
        # configure visualization settings based on config
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1 if self.config.enable_shadows else 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1 if self.config.enable_gui else 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        
        # set background color to white for cleaner look
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setRealTimeSimulation(0)
    
    def setup_environment(self):
        """setup the physical environment (floor, conveyor, counters, bins)"""
        # load custom floor plane
        pplane_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.config.floor_size, self.config.floor_size, self.config.floor_height],
            rgbaColor=[1, 1, 1, 1],  # white in normalized format (0-1 range)
            specularColor=[0, 0, 0]  # no shininess
        )
        plane_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.config.floor_size, self.config.floor_size, self.config.floor_height]
        )
        self.plane_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=plane_collision,
            baseVisualShapeIndex=pplane_visual,
            basePosition=self.config.floor_position
        )

        # load conveyor belt
        belt_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.config.belt_length / 2, self.config.belt_width / 2, self.config.belt_height / 2])
        belt_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.config.belt_length / 2, self.config.belt_width / 2, self.config.belt_height / 2], rgbaColor=[0, 0, 0, 1])
        self.belt_id = p.createMultiBody(0, belt_col, belt_vis, self.config.belt_position)

        # load counter thing
        counter_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.config.counter_length / 2, self.config.counter_width / 2, self.config.counter_height / 2])
        counter_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.config.counter_length / 2, self.config.counter_width / 2, self.config.counter_height / 2], rgbaColor=[0.5, 0.5, 0.5, 1])
        self.counter_id = p.createMultiBody(0, counter_col, counter_vis, self.config.counter_position)

        # load robot arm counter/platform
        arm_counter_length = 0.8  # smaller platform just for the arm base
        arm_counter_width = 0.4
        arm_counter_height = 1  # lower height
        arm_counter_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arm_counter_length / 2, arm_counter_width / 2, arm_counter_height / 2])
        arm_counter_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[arm_counter_length / 2, arm_counter_width / 2, arm_counter_height / 2], rgbaColor=[0.4, 0.4, 0.4, 1])
        arm_counter_position = [self.config.robot_arm_position[0], self.config.robot_arm_position[1]+0.1, arm_counter_height / 2]
        self.arm_counter_id = p.createMultiBody(0, arm_counter_col, arm_counter_vis, arm_counter_position)

        # load trash bins
        self.setup_trash_bins()
    
    def setup_trash_bins(self):
        """
        loads and colors the trash bins.
        """
        self.bin_recycling = p.loadURDF(self.config.trash_bin_urdf_path, basePosition=self.config.recycling_bin_position, globalScaling=self.config.bin_scale, useFixedBase=True)
        self.bin_trash = p.loadURDF(self.config.trash_bin_urdf_path, basePosition=self.config.trash_bin_position, globalScaling=self.config.bin_scale, useFixedBase=True)

        def transparent_box(body_id, body_color, edge_color):
            visual_shapes = p.getVisualShapeData(body_id)
            for shape in visual_shapes:
                link_index = shape[1]
                shape_name = shape[4].decode("utf-8") if isinstance(shape[4], bytes) else shape[4]
                if "edge" in shape_name.lower() or "rim" in shape_name.lower() or link_index == 1:
                    p.changeVisualShape(body_id, link_index, rgbaColor=edge_color)
                else:
                    p.changeVisualShape(body_id, link_index, rgbaColor=body_color)

        blue_body = [0, 0, 1, 0.5]
        blue_edge = [0, 0, 1, 1.0]

        gray_body = [0.7, 0.7, 0.7, 0.5]
        gray_edge = [0.5, 0.5, 0.5, 1.0]

        transparent_box(self.bin_recycling, blue_body, blue_edge)
        transparent_box(self.bin_trash, gray_body, gray_edge)
    
    def setup_robot(self):
        """setup the robot arm"""
        # load robot arm kuka
        self.kuka_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=self.config.robot_arm_position, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.kuka_id)
        for link_index in range(-1, self.num_joints):  
            color = [1, 1, 1, 1] if link_index % 2 == 0 else [0, 0, 0, 1]  # black and white colors
            p.changeVisualShape(self.kuka_id, link_index, rgbaColor=color)

        # store initial joint positions for reset function
        self.initial_joint_positions = get_initial_joint_positions(self.kuka_id, self.num_joints)
    
    def setup_cameras(self):
        """setup all cameras"""
        # load camera and model
        self.camera = TopDownCamera(self.config.img_width, self.config.img_height, self.config.camera_position, self.config.floor_plane_size)
        
        # calculate proper angles for perspective camera to look at conveyor belt
        cam_pos = self.config.perspective_camera_position
        belt_center = self.config.belt_position
        
        # calculate yaw (horizontal angle) - angle from camera to belt in xy plane
        dx = belt_center[0] - cam_pos[0]
        dy = belt_center[1] - cam_pos[1]
        yaw = math.degrees(math.atan2(dy, dx))
        
        # calculate pitch (vertical angle) - angle to look down at belt
        dz = belt_center[2] - cam_pos[2]
        horizontal_distance = math.sqrt(dx*dx + dy*dy)
        pitch = math.degrees(math.atan2(dz, horizontal_distance))
        
        self.perspective_camera = PerspectiveCamera(
            self.config.img_width, self.config.img_height, 
            self.config.perspective_camera_position, self.config.floor_plane_size,
            yaw=yaw, pitch=pitch, roll=30, fov=75
        )
        self.top_camera = PerspectiveCamera(self.config.img_width, self.config.img_height, self.config.top_camera_position, self.config.floor_plane_size, yaw=90, pitch=-90, roll=0, fov=75)
        
        # add debug line to show where perspective camera is looking
        perspective_target = self.perspective_camera._calculate_target_from_angles()
        p.addUserDebugLine(self.config.perspective_camera_position, perspective_target, [1,0,0], lineWidth=5, lifeTime=0)
    
    def setup_model(self):
        """setup the yolo model"""
        self.model = YOLO(self.config.model_path)
