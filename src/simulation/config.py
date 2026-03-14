import os
from dataclasses import dataclass, field

# path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class SimConfig:
    """
    config for sim environment and robot stuff
    """
    # physics
    gravity: float = -30   # gravity (increased for accurate drop)
    simulation_fps: float = 40  # hz

    # conveyor belt
    belt_velocity: float = 3
    belt_length: float = 4.0
    belt_width: float = 0.8
    belt_height: float = 0.05
    belt_position: list = field(default_factory=lambda: [-0.3, 0.1, 1.01])

    # counter stuff
    counter_length: float = 4.0
    counter_width: float = 0.8
    counter_height: float = 1
    counter_position: list = field(default_factory=lambda: [-0.3, 0.1, 0.49])
    
    # spawning
    spawn_x_position: float = -1.0
    spawn_z_height: float = 1.15  # above belt so it doesnt clip
    object_scale: float = 0.25
    enable_variants: bool = False
    
    # obj physics
    object_lateral_friction: float = 0.2
    object_restitution: float = 0.1
    object_linear_damping: float = 0.5
    object_angular_damping: float = 0.5
    
    # settling params
    max_settle_steps: int = 240
    movement_threshold: float = 0.001
    stable_frames_required: int = 20
    fall_through_threshold: float = -0.5  # if z goes below this something went wrong

    # removal boundaries
    conveyor_end_x: float = 1.5
    conveyor_start_x: float = -3.0
    min_z_position: float = -1.0
    yolo_zone_exit_x: float = 1.5
    
    # yolo stuff
    yolo_trigger_margin: float = 0.1
    camera_center_x: float = 0.0
    
    # robot arm
    pickup_x_coord: float = 0.535
    detection_line_x: float = -1.0
    confidence_threshold: float = 0.5
    arm_lead_time: float = 1.0
    arm_above_offset: float = 0.3  # now uses depth camera
    arm_lift_height: float = 1.5
    arm_threshold: float = 0.14
    arm_reset_threshold: float = 0.6
    arm_pause_before_drop: float = 0.5  # wait before drop to prevent launching
    arm_base_position: list = field(default_factory=lambda: [0, 0.6, 2])
    
    # bins
    recycling_bin_position: list = field(default_factory=lambda: [2.15, 0.1, 0.5])
    trash_bin_position: list = field(default_factory=lambda: [0.925, 1.1, 0.5])
    bin_scale: float = 2.0
    
    # floor and env
    floor_size: float = 5.0
    floor_height: float = 0.01
    floor_position: list = field(default_factory=lambda: [0, 0, -0.5])
    robot_arm_position: list = field(default_factory=lambda: [0, 0.6, 1])
    
    # paths
    model_path: str = os.path.join(PROJECT_ROOT, 'models/trash_detector/weights/new_best_model.pt')
    ycb_urdf_path: str = os.path.join(PROJECT_ROOT, 'assets', 'urdf', 'ycb')
    trash_bin_urdf_path: str = os.path.join(PROJECT_ROOT, "assets/urdf/trash_bin.urdf")
    drop_position: list = field(default_factory=lambda: [0.9, 0.8, 1.5])
    
    # camera settings
    # detection cam - lower res for speed
    detection_img_width: int = 512
    detection_img_height: int = 512
    # perspective cam - higher res for nicer looks
    perspective_img_width: int = 1024
    perspective_img_height: int = 1024
    # top cam - lower res for speed
    top_img_width: int = 512
    top_img_height: int = 512
    # legacy stuff for main.py backwards compat
    img_width: int = 512
    img_height: int = 512
    camera_position: list = field(default_factory=lambda: [0, 0.03, 3])
    perspective_camera_position: list = field(default_factory=lambda: [3, -2, 4])
    top_camera_position: list = field(default_factory=lambda: [0, 0, 6])
    floor_plane_size: float = 1.0
    
    # spawn randomization
    spawn_random_y_low: float = -0.05
    spawn_random_y_high: float = 0.4
    
    # objects that need pitch adjustment like cans bottles cups etc
    pitch_adjust_list: list = field(default_factory=lambda: [
        "002_master_chef_can.urdf", "003_cracker_box.urdf", "004_sugar_box.urdf", "005_tomato_soup_can.urdf", "006_mustard_bottle.urdf", "007_tuna_fish_can.urdf", "010_potted_meat_can.urdf", "021_bleach_cleanser.urdf", "022_windex_bottle.urdf", "065-a_cups.urdf", "065-b_cups.urdf", "065-c_cups.urdf", "065-d_cups.urdf", "065-e_cups.urdf", "065-f_cups.urdf", "065-g_cups.urdf", "065-h_cups.urdf", "065-i_cups.urdf", "065-j_cups.urdf"
    ])
    # recycling
    recycling_classes: list = field(default_factory=lambda: [
        "Master Chef Can",
        "Cracker Box",
        "Sugar Box",
        "Tomato Soup Can",
        #"Mustard Bottle",
        "Tuna Fish Can",
        "Pudding Box",
        "Gelatin Box",
        "Potted Meat Can",
        "Bleach Cleanser",
        "Windex Bottle",
        "Bowl",
        "Cups"
    ])
    
    # trash
    trash_classes: list = field(default_factory=lambda: [
        "Banana",
        "Strawberry", 
        "Apple",
        "Lemon",
        "Peach",
        "Pear",
        "Orange",
        "Plum",
        "Sponge",
        "Large Marker"
    ])
    
    # urdf -> recycling class mapping (for spawning)
    recyclable_urdf_files: list = field(default_factory=lambda: [
        "002_master_chef_can.urdf",    
        "003_cracker_box.urdf",        
        "004_sugar_box.urdf",          
        "005_tomato_soup_can.urdf",    
        #"006_mustard_bottle.urdf",  
        "007_tuna_fish_can.urdf",    
        "008_pudding_box.urdf",      
        "009_gelatin_box.urdf",        
        "010_potted_meat_can.urdf",  
        "021_bleach_cleanser.urdf",    
        "022_windex_bottle.urdf",    
        "024_bowl.urdf",        
        "065-a_cups.urdf"   
    ])
    
    # urdf -> trash mapping
    trash_urdf_files: list = field(default_factory=lambda: [
        "011_banana.urdf",            
        "012_strawberry.urdf",          
        "013_apple.urdf",            
        "014_lemon.urdf",             
        "015_peach.urdf",              
        "016_pear.urdf",           
        "017_orange.urdf",              
        "018_plum.urdf",              
        "026_sponge.urdf",             
        "040_large_marker.urdf"      
    ])
    
    # viz
    enable_shadows: bool = False  # shadows off for performance
    enable_gui: bool = False
    enable_top_camera: bool = True
    enable_perspective_frames: bool = True
    enable_detection_frames: bool = True
    
    # testing flags
    spawn_only_recyclables: bool = False
    spawn_only_trash: bool = False
    spawn_exclude_urdf_files: list = field(default_factory=lambda: ["006_mustard_bottle.urdf", "021_bleach_cleanser.urdf", "022_windex_bottle.urdf"])
    
    # logging
    enable_frame_logging: bool = True
    frame_log_interval: int = 5
