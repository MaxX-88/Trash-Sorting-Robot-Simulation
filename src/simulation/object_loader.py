import pybullet as p
import os
import random
from math import radians
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ObjectLoader:
    """
    handles loading and spawning of objects in the simulation
    """
    
    def __init__(self, config):
        self.config = config
        self.object_id = None
    
    def load_random_object(self):
        """
        loads a random ycb object (including variants if enabled) onto the conveyor belt with randomized position and orientation.
        spawns above the belt and waits for contact before applying conveyor velocity.
        """
        ycb_dir = self.config.ycb_urdf_path
        variant_dir = os.path.join(os.path.dirname(ycb_dir), 'ycb_variants')
        urdf_files = []
        
        # collect urdfs from main ycb dir
        urdf_files += [os.path.join(ycb_dir, f) for f in os.listdir(ycb_dir) if f.endswith('.urdf')]
        # collect urdfs from variants dir if it exists
        if os.path.exists(variant_dir) and self.config.enable_variants:
            urdf_files += [os.path.join(variant_dir, f) for f in os.listdir(variant_dir) if f.endswith('.urdf')]
        
        # temp: filter for only recyclable objects if testing flag is enabled
        if self.config.spawn_only_recyclables:
            recyclable_paths = []
            for urdf_file in urdf_files:
                filename = os.path.basename(urdf_file)
                if filename in self.config.recyclable_urdf_files:
                    recyclable_paths.append(urdf_file)
            urdf_files = recyclable_paths
        
        # temp: filter for only trash objects if testing flag is enabled
        if self.config.spawn_only_trash:
            trash_paths = []
            for urdf_file in urdf_files:
                filename = os.path.basename(urdf_file)
                if filename in self.config.trash_urdf_files:
                    trash_paths.append(urdf_file)
            urdf_files = trash_paths
        
        if not urdf_files:
            raise RuntimeError("No YCB URDF files found in either main or variant directory!")
        random_urdf_file = random.choice(urdf_files)

        # randomized position
        object_start_pos = [self.config.spawn_x_position, random.uniform(self.config.spawn_random_y_low, self.config.spawn_random_y_high), self.config.spawn_z_height]

        # randomize yaw (rotation around z)
        random_yaw = random.uniform(0, 360)
        urdf_basename = os.path.basename(random_urdf_file)
        if urdf_basename in self.config.pitch_adjust_list: # list for objects that should be in this pitch
            rotation = [0, -90, random_yaw]
        else:
            rotation = [0, 0, random_yaw]
        quaternion = p.getQuaternionFromEuler([radians(x) for x in rotation])
        
        # load object with zero initial velocity (let it fall naturally)
        self.object_id = p.loadURDF(random_urdf_file, basePosition=object_start_pos, baseOrientation=quaternion, globalScaling=self.config.object_scale)
        p.resetBaseVelocity(self.object_id, linearVelocity=[0, 0, 0])  # start with no velocity
        
        # add damping to prevent bouncing
        p.changeDynamics(self.object_id, -1, lateralFriction=self.config.object_lateral_friction, restitution=self.config.object_restitution, linearDamping=self.config.object_linear_damping, angularDamping=self.config.object_angular_damping)
        
        # wait for object to settle on the belt before applying conveyor velocity
        self.wait_for_belt_contact()
        
        return self.object_id

    def wait_for_belt_contact(self):
        """
        waits for the object to make contact with the conveyor belt and settle before applying velocity.
        """
        logger.info("[SPAWN] Waiting for object to settle on belt")
        contact_made = False
        settled = False
        last_pos = None
        stable_count = 0
        
        # get belt_id from the simulation environment - we'll need to pass this in
        # for now, we'll assume it's available as a class variable or passed parameter
        belt_id = getattr(self, 'belt_id', None)
        if belt_id is None:
            logger.warning("[SPAWN] Belt ID not available, skipping contact check")
            p.resetBaseVelocity(self.object_id, linearVelocity=[self.config.belt_velocity, 0, 0])
            return
        
        for step in range(self.config.max_settle_steps):
            p.stepSimulation()
            
            # check if object is in contact with the belt
            contacts = p.getContactPoints(bodyA=self.object_id, bodyB=belt_id)
            if contacts:
                contact_made = True
                
                # check if object has settled (stopped moving significantly)
                pos, _ = p.getBasePositionAndOrientation(self.object_id)
                if last_pos is not None:
                    movement = abs(pos[2] - last_pos[2])  # check vertical movement
                    if movement < self.config.movement_threshold:
                        stable_count += 1
                        if stable_count >= self.config.stable_frames_required:
                            settled = True
                            logger.info(f"[SPAWN] Object settled on belt after {step} steps")
                            break
                    else:
                        stable_count = 0  # reset if object is still moving
                last_pos = pos
            
            # also check if object has fallen too far (error case just found)
            pos, _ = p.getBasePositionAndOrientation(self.object_id)
            if pos[2] < self.config.fall_through_threshold:  # object fell through the belt
                logger.warning("[SPAWN] Object fell through belt, respawning...")
                p.removeBody(self.object_id)
                self.load_random_object()  # recursive call to respawn
                return
        
        if not contact_made:
            logger.warning("[SPAWN] Object did not make contact with belt within timeout")
        elif not settled:
            logger.warning("[SPAWN] Object did not settle within timeout, applying velocity anyway")
        
        # apply conveyor velocity once settled
        p.resetBaseVelocity(self.object_id, linearVelocity=[self.config.belt_velocity, 0, 0])
        logger.info(f"[SPAWN] Applied conveyor velocity: {self.config.belt_velocity} m/s")
    
    def set_belt_id(self, belt_id):
        """
        set the belt id for contact checking
        """
        self.belt_id = belt_id
