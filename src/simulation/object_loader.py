import pybullet as p
import os
import random
from math import radians
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ObjectLoader:
    """
    handles loading objs into sim
    """

    def __init__(self, config):
        self.config = config
        self.object_id = None
    
    def load_random_object(self):
        """
        loads random ycb obj above belt, applies physics
        """
        ycb_dir = self.config.ycb_urdf_path
        variant_dir = os.path.join(os.path.dirname(ycb_dir), 'ycb_variants')
        urdf_files = []
        
        # get urdfs from dir
        urdf_files += [os.path.join(ycb_dir, f) for f in os.listdir(ycb_dir) if f.endswith('.urdf')]
        if os.path.exists(variant_dir) and self.config.enable_variants:
            urdf_files += [os.path.join(variant_dir, f) for f in os.listdir(variant_dir) if f.endswith('.urdf')]
        # exclude specific objects from spawning (base + variants)
        exclude = getattr(self.config, 'spawn_exclude_urdf_files', [])
        if exclude:
            exclude_stems = [e.replace('.urdf', '') for e in exclude]
            urdf_files = [
                f for f in urdf_files
                if os.path.basename(f) not in exclude
                and not any(stem in os.path.basename(f) for stem in exclude_stems)
            ]

        # filter to recyclables only if flag set
        if self.config.spawn_only_recyclables:
            recyclable_paths = []
            for urdf_file in urdf_files:
                filename = os.path.basename(urdf_file)
                if filename in self.config.recyclable_urdf_files:
                    recyclable_paths.append(urdf_file)
            urdf_files = recyclable_paths
        
        # filter to trash only if flag set
        if self.config.spawn_only_trash:
            trash_paths = []
            for urdf_file in urdf_files:
                filename = os.path.basename(urdf_file)
                if filename in self.config.trash_urdf_files:
                    trash_paths.append(urdf_file)
            urdf_files = trash_paths
        
        if not urdf_files:
            raise RuntimeError("No YCB URDF files found!")
        random_urdf_file = random.choice(urdf_files)
        urdf_basename = os.path.basename(random_urdf_file)
        
        is_trash = urdf_basename in self.config.trash_urdf_files
        
        # get name from urdf filename
        object_name = urdf_basename.replace('.urdf', '')
        if '_' in object_name:
            object_name = '_'.join(object_name.split('_')[1:])

        # random pos on belt
        object_start_pos = [self.config.spawn_x_position, random.uniform(self.config.spawn_random_y_low, self.config.spawn_random_y_high), self.config.spawn_z_height]

        # random rotation (pitch for cans/bottles)
        random_yaw = random.uniform(0, 360)
        if urdf_basename in self.config.pitch_adjust_list:
            rotation = [0, -90, random_yaw]
        else:
            rotation = [0, 0, random_yaw]
        quaternion = p.getQuaternionFromEuler([radians(x) for x in rotation])
        
        self.object_id = p.loadURDF(random_urdf_file, basePosition=object_start_pos, baseOrientation=quaternion, globalScaling=self.config.object_scale)
        p.resetBaseVelocity(self.object_id, linearVelocity=[self.config.belt_velocity, 0, 0])
        
        # damping to prevent bouncing
        p.changeDynamics(self.object_id, -1, lateralFriction=self.config.object_lateral_friction, restitution=self.config.object_restitution, linearDamping=self.config.object_linear_damping, angularDamping=self.config.object_angular_damping)
        
        self.last_object_info = {
            'object_id': self.object_id,
            'is_trash': is_trash,
            'object_name': object_name
        }
        
        self.wait_for_belt_contact()

        return self.object_id
    
    def get_last_object_info(self):
        """returns (object_id, is_trash, object_name)"""
        if hasattr(self, 'last_object_info'):
            info = self.last_object_info
            return info['object_id'], info['is_trash'], info['object_name']
        return None, False, "unknown"

    def wait_for_belt_contact(self):
        logger.info("[SPAWN] Object spawned")
    
    def set_belt_id(self, belt_id):
        self.belt_id = belt_id
