import os
from dataclasses import dataclass, field

# path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

@dataclass
class SimConfig:
    """
    configuration for the simulation environment and robot
    """
    # physics settings
    gravity: float = -98  #gravity
    simulation_fps: float = 90.0  # simulation frequency, hz
    
    # conveyor belt settings
    belt_velocity: float = 3
    belt_length: float = 4.0
    belt_width: float = 0.8
    belt_height: float = 0.05
    belt_position: list = field(default_factory=lambda: [-0.3, 0.1, 1.01])
    
    # counter settings
    counter_length: float = 4.0
    counter_width: float = 0.8
    counter_height: float = 1
    counter_position: list = field(default_factory=lambda: [-0.3, 0.1, 0.49])
    
    # object spawning settings
    spawn_x_position: float = -1.0
    spawn_z_height: float = 1.15  # set height above belt to prevent clipping
    object_scale: float = 0.25  # global scaling for objects
    enable_variants: bool = False  # enable/disable spawning variant objects
    
    # object physics settings
    object_lateral_friction: float = 0.2
    object_restitution: float = 0.1
    object_linear_damping: float = 0.5
    object_angular_damping: float = 0.5
    
    # object settling settings
    max_settle_steps: int = 240  # maximum steps to wait for settling
    movement_threshold: float = 0.001  # movement threshold for settling
    stable_frames_required: int = 20  # consecutive stable frames needed
    fall_through_threshold: float = -0.5  # z position threshold for fall-through detection
    
    # object removal boundaries
    conveyor_end_x: float = 1.5  # x position where objects are removed
    conveyor_start_x: float = -3.0  # x position where objects are removed
    min_z_position: float = -1.0  # minimum z position before removal
    yolo_zone_exit_x: float = 1.5  # x position where processed objects are removed
    
    # yolo detection settings
    yolo_trigger_margin: float = 0.1  # yolo activation zone
    camera_center_x: float = 0.0  # center x position
    
    # robot arm settings
    pickup_x_coord: float = 0.535
    detection_line_x: float = -1.0
    confidence_threshold: float = 0.5
    arm_lead_time: float = 0.65
    arm_above_offset: float = 0.3  # height above pickup position, should use depth camera later
    arm_lift_height: float = 1.5  # height to lift object
    arm_threshold: float = 0.14  # position threshold for arm movement
    arm_reset_threshold: float = 0.6  # threshold for arm reset/base movement
    arm_base_position: list = field(default_factory=lambda: [0, 0.6, 2])  # base position for arm reset
    
    # trash bin settings
    recycling_bin_position: list = field(default_factory=lambda: [2.15, 0.1, 0.5])
    trash_bin_position: list = field(default_factory=lambda: [0.925, 1.1, 0.5])
    bin_scale: float = 2.0
    
    # floor and environment settings
    floor_size: float = 5.0
    floor_height: float = 0.01
    floor_position: list = field(default_factory=lambda: [0, 0, -0.5])
    robot_arm_position: list = field(default_factory=lambda: [0, 0.6, 1])
    
    # paths and model settings
    model_path: str = os.path.join(PROJECT_ROOT, 'models/trash_detector/weights/new_best_model.pt')
    ycb_urdf_path: str = os.path.join(PROJECT_ROOT, 'assets', 'urdf', 'ycb')
    trash_bin_urdf_path: str = os.path.join(PROJECT_ROOT, "assets/urdf/roid_bin.urdf")
    drop_position: list = field(default_factory=lambda: [0.9, 0.8, 1.5])
    
    # camera settings
    img_width: int = 1024
    img_height: int = 1024
    camera_position: list = field(default_factory=lambda: [0, 0, 3])
    perspective_camera_position: list = field(default_factory=lambda: [3,-2,4])
    top_camera_position: list = field(default_factory=lambda: [0,0,6])
    floor_plane_size: float = 1.0
    
    # spawning settings
    spawn_random_y_low: float = -0.1
    spawn_random_y_high: float = -0.09999999
    
    # object lists
    pitch_adjust_list: list = field(default_factory=lambda: [
        "002_master_chef_can.urdf", "003_cracker_box.urdf", "004_sugar_box.urdf", "005_tomato_soup_can.urdf", "006_mustard_bottle.urdf", "007_tuna_fish_can.urdf", "010_potted_meat_can.urdf", "021_bleach_cleanser.urdf", "022_windex_bottle.urdf", "065-a_cups.urdf", "065-b_cups.urdf", "065-c_cups.urdf", "065-d_cups.urdf", "065-e_cups.urdf", "065-f_cups.urdf", "065-g_cups.urdf", "065-h_cups.urdf", "065-i_cups.urdf", "065-j_cups.urdf"
    ])
    recycling_classes: list = field(default_factory=lambda: [
        "Master Chef Can",
        "Cracker Box",
        "Sugar Box",
        "Tomato Soup Can",
        "Mustard Bottle",
        "Tuna Fish Can",
        "Pudding Box",
        "Gelatin Box",
        "Potted Meat Can",
        "Bleach Cleanser",
        "Windex Bottle",
        "Bowl",
        "Cups"
    ])
    
    # trash classes - objects that should be picked up and put in trash bin
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
    
    # mapping of urdf filenames to recycling class names (for filtering recyclables)
    recyclable_urdf_files: list = field(default_factory=lambda: [
        "002_master_chef_can.urdf",     # master chef can
        "003_cracker_box.urdf",         # cracker box
        "004_sugar_box.urdf",           # sugar box
        "005_tomato_soup_can.urdf",     # tomato soup can  
        "006_mustard_bottle.urdf",      # mustard bottle
        "007_tuna_fish_can.urdf",       # tuna fish can
        "008_pudding_box.urdf",         # pudding box
        "009_gelatin_box.urdf",         # gelatin box
        "010_potted_meat_can.urdf",     # potted meat can
        "021_bleach_cleanser.urdf",     # bleach cleanser
        "022_windex_bottle.urdf",       # windex bottle
        "024_bowl.urdf",                # bowl
        "065-a_cups.urdf"               # cups
    ])
    
    # mapping of urdf filenames to trash class names (for filtering trash items)
    trash_urdf_files: list = field(default_factory=lambda: [
        "011_banana.urdf",              # banana
        "012_strawberry.urdf",          # strawberry
        "013_apple.urdf",               # apple
        "014_lemon.urdf",               # lemon
        "015_peach.urdf",               # peach
        "016_pear.urdf",                # pear
        "017_orange.urdf",              # orange
        "018_plum.urdf",                # plum
        "026_sponge.urdf",              # sponge
        "040_large_marker.urdf"         # large marker
    ])
    
    # visualization settings
    enable_shadows: bool = False  # disable shadows for better performance
    enable_gui: bool = True  # keep gui enabled for controls
    enable_top_camera: bool = True  # enable/disable top camera for performance
    enable_perspective_frames: bool = True  # enable/disable perspective frame saving
    enable_detection_frames: bool = True  # enable/disable detection frame saving
    
    # testing settings
    spawn_only_recyclables: bool = False  # temp: only spawn recycling objects for testing (easy to revert)
    spawn_only_trash: bool = False  # temp: only spawn trash objects for testing (easy to revert)
