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
    sets up and manages the sim environment
    """

    def __init__(self, config, headless=False):
        self.config = config
        self.setup_physics(headless)
        self.setup_environment()
        self.setup_robot()
        self.setup_cameras()
        self.setup_model()
    def setup_physics(self, headless=False):
        """init pybullet connection"""
        if headless:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)

        # debug viz stuff (shadows etc)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity)
        
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1 if self.config.enable_shadows else 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1 if self.config.enable_gui else 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setRealTimeSimulation(0)
    
    def setup_environment(self):
        """setup floor, belt, counters, bins"""
        # floor plane
        pplane_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.config.floor_size, self.config.floor_size, self.config.floor_height],
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0, 0, 0]
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

        # conveyor belt
        belt_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.config.belt_length / 2, self.config.belt_width / 2, self.config.belt_height / 2])
        belt_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.config.belt_length / 2, self.config.belt_width / 2, self.config.belt_height / 2], rgbaColor=[0, 0, 0, 1])
        self.belt_id = p.createMultiBody(0, belt_col, belt_vis, self.config.belt_position)

        # counter surface
        counter_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.config.counter_length / 2, self.config.counter_width / 2, self.config.counter_height / 2])
        counter_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.config.counter_length / 2, self.config.counter_width / 2, self.config.counter_height / 2], rgbaColor=[0.5, 0.5, 0.5, 1])
        self.counter_id = p.createMultiBody(0, counter_col, counter_vis, self.config.counter_position)

        # arm pedestal
        arm_counter_length = 0.8
        arm_counter_width = 0.4
        arm_counter_height = 1
        arm_counter_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arm_counter_length / 2, arm_counter_width / 2, arm_counter_height / 2])
        arm_counter_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[arm_counter_length / 2, arm_counter_width / 2, arm_counter_height / 2], rgbaColor=[0.4, 0.4, 0.4, 1])
        arm_counter_position = [self.config.robot_arm_position[0], self.config.robot_arm_position[1]+0.1, arm_counter_height / 2]
        self.arm_counter_id = p.createMultiBody(0, arm_counter_col, arm_counter_vis, arm_counter_position)

        self.setup_trash_bins()
    
    def setup_trash_bins(self):
        """loads bins and colors them (blue/gray)"""
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
        """setup kuka arm + colors"""
        self.kuka_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=self.config.robot_arm_position, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.kuka_id)
        for link_index in range(-1, self.num_joints):  
            color = [1, 1, 1, 1] if link_index % 2 == 0 else [0, 0, 0, 1]
            p.changeVisualShape(self.kuka_id, link_index, rgbaColor=color)

        self.initial_joint_positions = get_initial_joint_positions(self.kuka_id, self.num_joints)
    
    def setup_cameras(self):
        """setup cams - top, perspective, etc"""
        self.camera = TopDownCamera(self.config.img_width, self.config.img_height, self.config.camera_position, self.config.floor_plane_size)
        
        # calc angles so cam looks at belt
        cam_pos = self.config.perspective_camera_position
        belt_center = self.config.belt_position
        
        dx = belt_center[0] - cam_pos[0]
        dy = belt_center[1] - cam_pos[1]
        yaw = math.degrees(math.atan2(dy, dx))
        
        dz = belt_center[2] - cam_pos[2]
        horizontal_distance = math.sqrt(dx*dx + dy*dy)
        pitch = math.degrees(math.atan2(dz, horizontal_distance))
        
        self.perspective_camera = PerspectiveCamera(
            self.config.perspective_img_width, self.config.perspective_img_height, 
            self.config.perspective_camera_position, self.config.floor_plane_size,
            yaw=yaw, pitch=pitch, roll=30, fov=75
        )
        self.top_camera = PerspectiveCamera(self.config.top_img_width, self.config.top_img_height, self.config.top_camera_position, self.config.floor_plane_size, yaw=90, pitch=-90, roll=0, fov=75)
        
        # debug line - where perspective cam looks
        perspective_target = self.perspective_camera._calculate_target_from_angles()
        p.addUserDebugLine(self.config.perspective_camera_position, perspective_target, [1,0,0], lineWidth=5, lifeTime=0)
    
    def setup_model(self):
        """load yolo model"""
        self.model = YOLO(self.config.model_path)
