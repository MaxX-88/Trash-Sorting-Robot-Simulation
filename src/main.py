import pybullet as p
import numpy as np
import cv2
import time
import sys
import os
from collections import Counter

# path setup stuff
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.simulation import SimConfig, ArmState, SimulationEnvironment, ObjectLoader
from src.control.pybullet_helpers import move_arm_to, wait_for_arm_to_reach, grab_object, release_object, move_arm_to_joint_positions, wait_for_joints_to_reach
from src.utils.debug_gui import DebugInterface
from src.utils.logger import get_logger
from src.video_generator import create_videos_from_simulation, cleanup_frames

logger = get_logger(__name__)



class RobotController:
    """
    manages the robot arm simulation, detection, and interaction.
    """
    def __init__(self, config: SimConfig, headless: bool = True, capture_frames: bool = False):
        self.config = config
        self.target_info = None
        self.picked = False
        self.tracking = False
        self.constraint_id = None
        self.release_time = None
        self.object_processed = False  # track if object has been processed by yolo
        self.arm_processing_trash = False  # track if arm is actively processing a trash object
        self.arm_substate = None  # track substeps for non-blocking arm movement
        self.picking_start_step = None  # track simulation step when picking started for timeout
        self.picking_timeout_steps = 70  # timeout after 270 simulation steps (3 seconds at 90 fps)
        self.capture_frames = capture_frames
        self.frame_count = 0
        
        # initialize simulation environment and object loader
        self.sim_env = SimulationEnvironment(config, headless)
        self.object_loader = ObjectLoader(config)
        self.object_loader.set_belt_id(self.sim_env.belt_id)
        
        # setup frame capture directories if needed
        if self.capture_frames:
            self.frames_dir = os.path.join(PROJECT_ROOT, "frames")
            
            # create directories and clean frames only for enabled cameras
            if self.config.enable_perspective_frames:
                self.perspective_frames_dir = os.path.join(self.frames_dir, "perspective")
                os.makedirs(self.perspective_frames_dir, exist_ok=True)
                # clean existing frames
                if os.path.exists(self.perspective_frames_dir):
                    for frame_file in os.listdir(self.perspective_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(self.perspective_frames_dir, frame_file))
            else:
                # Clean perspective frames directory even if disabled to prevent accumulation
                perspective_frames_dir = os.path.join(self.frames_dir, "perspective")
                if os.path.exists(perspective_frames_dir):
                    for frame_file in os.listdir(perspective_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(perspective_frames_dir, frame_file))
            
            if self.config.enable_top_camera:
                self.top_frames_dir = os.path.join(self.frames_dir, "top")
                os.makedirs(self.top_frames_dir, exist_ok=True)
                # clean existing frames
                if os.path.exists(self.top_frames_dir):
                    for frame_file in os.listdir(self.top_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(self.top_frames_dir, frame_file))
            else:
                # Clean top frames directory even if disabled to prevent accumulation
                top_frames_dir = os.path.join(self.frames_dir, "top")
                if os.path.exists(top_frames_dir):
                    for frame_file in os.listdir(top_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(top_frames_dir, frame_file))
            
            if self.config.enable_detection_frames:
                self.detection_frames_dir = os.path.join(self.frames_dir, "detection")
                os.makedirs(self.detection_frames_dir, exist_ok=True)
                # clean existing frames
                if os.path.exists(self.detection_frames_dir):
                    for frame_file in os.listdir(self.detection_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(self.detection_frames_dir, frame_file))
            else:
                # Clean detection frames directory even if disabled to prevent accumulation
                detection_frames_dir = os.path.join(self.frames_dir, "detection")
                if os.path.exists(detection_frames_dir):
                    for frame_file in os.listdir(detection_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(detection_frames_dir, frame_file))
            
            enabled_cameras = []
            if self.config.enable_perspective_frames:
                enabled_cameras.append("perspective")
            if self.config.enable_top_camera:
                enabled_cameras.append("top")
            if self.config.enable_detection_frames:
                enabled_cameras.append("detection")
            
            logger.info(f"[CAPTURE] Frame capture enabled for: {', '.join(enabled_cameras)}")
        
        self._setup_frame_capture()
        self.gui = DebugInterface(self.sim_env.kuka_id, self.sim_env.num_joints, self.config)
        self.state = ArmState.IDLE
        self.previous_state = self.state
        self.last_img = None
        self.last_results = None
        self.debug_text_id = None
        self.last_debug_info = ""
        self.simulation_step = 0  # track total simulation steps

    def _setup_frame_capture(self):
        """setup frame capture directories if needed"""
        # setup frame capture directories if needed
        if self.capture_frames:
            self.frames_dir = os.path.join(PROJECT_ROOT, "frames")
            
            # create directories and clean frames only for enabled cameras
            if self.config.enable_perspective_frames:
                self.perspective_frames_dir = os.path.join(self.frames_dir, "perspective")
                os.makedirs(self.perspective_frames_dir, exist_ok=True)
                # clean existing frames
                if os.path.exists(self.perspective_frames_dir):
                    for frame_file in os.listdir(self.perspective_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(self.perspective_frames_dir, frame_file))
            else:
                # Clean perspective frames directory even if disabled to prevent accumulation
                perspective_frames_dir = os.path.join(self.frames_dir, "perspective")
                if os.path.exists(perspective_frames_dir):
                    for frame_file in os.listdir(perspective_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(perspective_frames_dir, frame_file))
            
            if self.config.enable_top_camera:
                self.top_frames_dir = os.path.join(self.frames_dir, "top")
                os.makedirs(self.top_frames_dir, exist_ok=True)
                # clean existing frames
                if os.path.exists(self.top_frames_dir):
                    for frame_file in os.listdir(self.top_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(self.top_frames_dir, frame_file))
            else:
                # Clean top frames directory even if disabled to prevent accumulation
                top_frames_dir = os.path.join(self.frames_dir, "top")
                if os.path.exists(top_frames_dir):
                    for frame_file in os.listdir(top_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(top_frames_dir, frame_file))
            
            if self.config.enable_detection_frames:
                self.detection_frames_dir = os.path.join(self.frames_dir, "detection")
                os.makedirs(self.detection_frames_dir, exist_ok=True)
                # clean existing frames
                if os.path.exists(self.detection_frames_dir):
                    for frame_file in os.listdir(self.detection_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(self.detection_frames_dir, frame_file))
            else:
                # Clean detection frames directory even if disabled to prevent accumulation
                detection_frames_dir = os.path.join(self.frames_dir, "detection")
                if os.path.exists(detection_frames_dir):
                    for frame_file in os.listdir(detection_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(detection_frames_dir, frame_file))
            
            enabled_cameras = []
            if self.config.enable_perspective_frames:
                enabled_cameras.append("perspective")
            if self.config.enable_top_camera:
                enabled_cameras.append("top")
            if self.config.enable_detection_frames:
                enabled_cameras.append("detection")
            
            logger.info(f"[CAPTURE] Frame capture enabled for: {', '.join(enabled_cameras)}")
    
    
    def _save_frame(self, perspective_img, top_img=None, detection_img=None):
        if not self.capture_frames:
            return
        
        # save perspective camera only if enabled
        if perspective_img is not None and self.config.enable_perspective_frames:
            perspective_bgr = cv2.cvtColor(perspective_img, cv2.COLOR_RGB2BGR)
            perspective_filename = os.path.join(self.perspective_frames_dir, f"frame_{self.frame_count:06d}.jpg")
            cv2.imwrite(perspective_filename, perspective_bgr)
        
        # save top camera only if provided and enabled
        if top_img is not None and self.config.enable_top_camera:
            top_bgr = cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR)
            top_filename = os.path.join(self.top_frames_dir, f"frame_{self.frame_count:06d}.jpg")
            cv2.imwrite(top_filename, top_bgr)
        
        # save detection window (yolo output) only if provided and enabled
        if detection_img is not None and self.config.enable_detection_frames:
            detection_bgr = cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR)
            detection_filename = os.path.join(self.detection_frames_dir, f"frame_{self.frame_count:06d}.jpg")
            cv2.imwrite(detection_filename, detection_bgr)
        
        self.frame_count += 1
    

    
    def _check_exit_key(self):
        """
        if q is pressed program exits
        """
        key = cv2.waitKey(1) & 0xFF
        return key == ord('q')

    
    def wait_drop(self, seconds):
        """
        literally waits
        """
        for _ in range(int(seconds * self.config.simulation_fps)):
            p.stepSimulation()
            #time.sleep(1./self.config.simulation_fps)

    def run(self):
        """
        main simulation loop
        """
        self.object_id = self.object_loader.load_random_object()
        self.state = ArmState.WAIT_FOR_OBJECT
        frame_count = 0
        sim_time = 0.0
        
        # first get, needs this
        #self.last_img = self.camera.get_image()[0]
        


        
        self.last_results = []
        while True:
            #self.last_img, self.last_img_arr = self.sim_env.camera.get_image()
            self.perspective_img, self.perspective_img_arr = self.sim_env.perspective_camera.get_image()
            
            # conditionally capture top camera based on config
            if self.config.enable_top_camera:
                self.top_img, self.top_img_arr = self.sim_env.top_camera.get_image()
            else:
                self.top_img = None
            
            # get object position
            obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
    
            # only spawn a new object if there is none AND arm is not processing a trash object
            past_yolo_zone = obj_pos[0] > self.config.yolo_zone_exit_x and self.object_processed
            if obj_pos[0] > self.config.conveyor_end_x or obj_pos[0] < self.config.conveyor_start_x or obj_pos[2] < self.config.min_z_position or past_yolo_zone and not self.arm_processing_trash:
            #if self.object_id is None and not self.arm_processing_trash:
                self.object_id = self.object_loader.load_random_object()
                self.object_processed = False  # reset for new object
                obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
            elif self.object_id is None and self.arm_processing_trash:
                logger.info("[SPAWN] Blocked - arm is processing trash object")
                obj_pos = [0, 0, 0]  # set default position when no object exists

            # check if object is in yolo detection zone, can't keep model always running as it will mess up loop times
            in_yolo_zone = (self.config.camera_center_x - self.config.yolo_trigger_margin <= obj_pos[0] < self.config.camera_center_x)
            
            # track when object passes through yolo zone
            if in_yolo_zone and not self.object_processed:
                self.object_processed = True  # mark as processed when entering yolo zone, lol
                logger.info(f"[YOLO] Object entered detection zone at X={obj_pos[0]:.2f}")
            
            inf_ms = 0
            if in_yolo_zone:
                import time as _time
                t0 = _time.time()
                logger.info(f"[YOLO] Running inference at frame {frame_count}")
                
                # add error handling for camera image capture
                try:
                    self.last_img, self.last_img_arr = self.sim_env.camera.get_image()
                    self.height_pixels = self.last_img_arr[3]
                    counter = Counter(self.height_pixels.flatten().tolist())
                    self.height = min(counter.keys())
                    logger.info(f"[DEPTH CAM] Height: {self.height}")
                    
                    near = self.sim_env.camera.near
                    far = self.sim_env.camera.far
                    self.depth = 3 - (far * near / (far - (far - near) * self.height))
                    logger.info(f"[DEPTH CAM] Depth: {self.depth}")
                    
                    # convert rgb to bgr for yolo, yolo was trained on bgr
                    img_bgr = cv2.cvtColor(self.last_img, cv2.COLOR_RGB2BGR)
                    self.last_results = self.sim_env.model(img_bgr, verbose=False)
                except Exception as e:
                    logger.warning(f"[YOLO] Error during inference: {e}")
                    self.last_img = np.zeros((self.config.img_height, self.config.img_width, 3), dtype=np.uint8)
                    self.last_results = []
                
                t1 = _time.time()
                inf_ms = (t1 - t0) * 1000
                logger.info(f"[YOLO] Inference completed in {inf_ms:.1f}ms")
            else:
                overlay_text = "YOLO: not run"
            # prepare info for debug overlay
            if self.last_results is not None and len(self.last_results) > 0 and hasattr(self.last_results[0], 'boxes') and self.last_results[0].boxes is not None:
                boxes = self.last_results[0].boxes.xyxy.cpu().numpy()
            else:
                boxes = []
            self.gui.update(self, fsm_state=self.state.name, sim_time=sim_time, target_info=self.target_info, boxes=boxes)
            
            contacts = p.getContactPoints(bodyA=self.object_id, bodyB=self.sim_env.belt_id)
            if contacts and not self.picked:
                p.resetBaseVelocity(self.object_id, linearVelocity=[self.config.belt_velocity, 0, 0])
            self._step_fsm(sim_time)
            if self.state != self.previous_state:
                logger.info(f"[FSM] State change: {self.previous_state.name} -> {self.state.name}")
                self.previous_state = self.state
            # always show yolo detection window, even if no detection
            try:
                output_img = self.last_img.copy() if self.last_img is not None else np.zeros((self.config.img_height, self.config.img_width, 3), dtype=np.uint8)
            except Exception as e:
                logger.warning(f"[DISPLAY] Error copying image: {e}")
                output_img = np.zeros((self.config.img_height, self.config.img_width, 3), dtype=np.uint8)
            
            # yolo debugging information
            yolo_debug_info = []
            yolo_debug_info.append(f"Frame: {frame_count}")
            yolo_debug_info.append(f"FSM: {self.state.name}")
            # yolo_debug_info.append(f"object x: {obj_pos[0]:.2f}")
            yolo_debug_info.append(f"YOLO Zone: {'YES' if in_yolo_zone else 'NO'}")
            yolo_debug_info.append(f"Arm Processing: {'YES' if self.arm_processing_trash else 'NO'}")
            
            if in_yolo_zone and inf_ms > 0:
                yolo_debug_info.append(f"Inference: {inf_ms:.1f}ms")
            
            # add detection information
            confs = self.last_results[0].boxes.conf.cpu().numpy() if len(self.last_results) > 0 and hasattr(self.last_results[0], 'boxes') else None
            if len(boxes) > 0 and confs is not None and len(confs) > 0:
                idx = np.argmax(confs)
                x1, y1, x2, y2 = boxes[idx]
                center_x = int(self.last_results[0].boxes.xywh[idx][0].item())
                center_y = int(self.last_results[0].boxes.xywh[idx][1].item())
                
                # get class information
                class_idx = int(self.last_results[0].boxes.cls[idx].cpu().numpy())
                class_name = self.sim_env.model.names[class_idx] if hasattr(self.sim_env.model, 'names') and class_idx < len(self.sim_env.model.names) else str(class_idx)
                confidence = confs[idx]
                
                # convert yolo coordinates to world coordinates
                detected_world_pos = self.sim_env.camera.get_pixel_world_coords(center_x, center_y)
                
                # add detection info
                yolo_debug_info.append(f"Detected: {class_name}")
                yolo_debug_info.append(f"Confidence: {confidence:.2f}")
                yolo_debug_info.append(f"Trash: {'YES' if class_name in self.config.trash_classes else 'NO'}")
                yolo_debug_info.append(f"YOLO Pos: ({detected_world_pos[0]:.2f}, {detected_world_pos[1]:.2f})")
                
                # draw detection box and center point
                cv2.circle(output_img, (center_x, center_y), 5, (0,255,0), -1)
                cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # add class name on the box
                cv2.putText(output_img, f"{class_name} ({confidence:.2f})", 
                        (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                yolo_debug_info.append("Detection: NONE")
            
            # draw all debug information on the image
            for i, info in enumerate(yolo_debug_info):
                y_pos = 30 + i * 25
                cv2.putText(output_img, info, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # save frames after all processing is complete
            if self.config.enable_top_camera:
                self._save_frame(self.perspective_img, self.top_img, output_img)
            else:
                self._save_frame(self.perspective_img, None, output_img)
            
            #cv2.imshow("YOLO Detection", output_img)
            cv2.imshow("YOLO Detection", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            cv2.imshow("Perspective Detection", cv2.cvtColor(self.perspective_img, cv2.COLOR_RGB2BGR))
            
            # show top camera only if enabled
            if self.config.enable_top_camera and self.top_img is not None:
                cv2.imshow("Top Detection", cv2.cvtColor(self.top_img, cv2.COLOR_RGB2BGR))
            
            # check for 'q' key press in opencv windows
            if self._check_exit_key():
                logger.info("[SIM] 'q' key pressed - terminating simulation")
                break
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"[SIM] Frame: {frame_count}")
        
            
            sim_time += 1.0 / self.config.simulation_fps
            for _ in range(3):
                p.stepSimulation()
                self.simulation_step += 1  # increment step counter for each physics step
            #time.sleep(1. / self.config.simulation_fps)
        
        # generate videos from captured frames
        if self.capture_frames:
            logger.info("[VIDEO] Generating videos from captured frames...")
            videos = create_videos_from_simulation(PROJECT_ROOT, fps=30, config=self.config)
            if videos:
                logger.info("[VIDEO] Videos created:")
                for camera, video_path in videos.items():
                    logger.info(f"[VIDEO] {camera}: {video_path}")
                # optionally clean up frames after video creation
                # cleanup_frames(PROJECT_ROOT)
            else:
                logger.warning("[VIDEO] No videos were created")

    def _step_fsm(self, sim_time):
        if self.state == ArmState.IDLE:
            self._handle_idle()
        elif self.state == ArmState.WAIT_FOR_OBJECT:
            self._handle_wait_for_object(sim_time)
        elif self.state == ArmState.PREPARE_PICK:
            self._handle_prepare_pick(sim_time)
        elif self.state == ArmState.PICKING:
            self._handle_picking()
        elif self.state == ArmState.LIFTING:
            self._handle_lifting()
        elif self.state == ArmState.RESETTING:
            self._handle_resetting()

    def _handle_idle(self):
        # could be used for future multi-arm or pause logic
        pass

    def _handle_wait_for_object(self, sim_time):
        boxes = self.last_results[0].boxes.xyxy.cpu().numpy() if len(self.last_results) > 0 and hasattr(self.last_results[0], 'boxes') else []
        confs = self.last_results[0].boxes.conf.cpu().numpy() if len(self.last_results) > 0 and hasattr(self.last_results[0], 'boxes') else []

        if self.target_info is None and len(boxes) > 0:
            idx = np.argmax(confs)
            if confs[idx] >= self.config.confidence_threshold:
                # get yolo-detected center coordinates
                center_x = int(self.last_results[0].boxes.xywh[idx][0].item())
                center_y = int(self.last_results[0].boxes.xywh[idx][1].item())
                
                # convert 2d pixel coordinates to 3d world coordinates
                detected_world_pos = self.sim_env.camera.get_pixel_world_coords(center_x, center_y)
                detected_world_pos.append(self.depth)
                logger.info(f"[DETECT] World Position: {detected_world_pos}")
                
                # get detected class name
                class_idx = int(self.last_results[0].boxes.cls[idx].cpu().numpy())
                class_name = self.sim_env.model.names[class_idx] if hasattr(self.sim_env.model, 'names') and class_idx < len(self.sim_env.model.names) else str(class_idx)
                
                if class_name not in self.config.trash_classes:
                    logger.info(f"[DETECT] Object '{class_name}' is recyclable - letting it continue")
                    # for recyclable objects, we can spawn new objects immediately since they don't get picked up
                    self.arm_processing_trash = False
                    return  # do not pick up recyclables, let them keep moving
                
                if detected_world_pos[0] > self.config.detection_line_x:
                    # set the target to the pos from model
                    logger.info(f"[DETECT] Trash object detected at {detected_world_pos} (YOLO)")
                    self.target_info = {"initial_pos": detected_world_pos, "detection_time": sim_time}
                    logger.info(f"[DETECT] Target acquired - initiating pickup sequence")
                    self.state = ArmState.PREPARE_PICK
                else:
                    logger.info(f"[DETECT] Tracking object at X={detected_world_pos[0]:.2f}, waiting to cross detection line")

    def _handle_prepare_pick(self, sim_time):
        time_offset = 0
        y = self.target_info["initial_pos"][1]
        print(f"Y value: {y}")
        if y < 0:
            time_offset = abs(y)*1.5
            print(f"Time offset: {time_offset}")
        #mess with predicted x based on y offset
        time_to_pickup = (self.config.pickup_x_coord - time_offset) / self.config.belt_velocity if self.config.belt_velocity > 0 else float('inf')
        print(f"initial time to pickup: {time_to_pickup}")
        if time_to_pickup < 0:
            time_to_pickup = 0.001

        logger.info(f"[ARM] Object {time_to_pickup:.2f}s from pickup position")

        if 0 < time_to_pickup <= self.config.arm_lead_time:
            logger.info(f"[ARM] Initiating pickup sequence")
            self.arm_processing_trash = True  # set flag to prevent new object spawning
            self.picking_start_step = self.simulation_step  # start timeout counter
            self.state = ArmState.PICKING

    def _handle_picking(self):
        # check for timeout (270 simulation steps = ~3 seconds at 90 fps)
        if self.picking_start_step and (self.simulation_step - self.picking_start_step) > self.picking_timeout_steps:
            steps_elapsed = self.simulation_step - self.picking_start_step
            logger.warning(f"[ARM] Picking operation timed out after {steps_elapsed} simulation steps, transitioning to reset")
            # reset all picking-related state variables
            self.picked = False
            self.tracking = False
            self.target_info = None
            if self.constraint_id:
                release_object(self.constraint_id)
                self.constraint_id = None
            self.arm_substate = None
            self.picking_start_step = None
            self.state = ArmState.RESETTING
            return
        offset = 0
        """
        x = self.target_info["initial_pos"][0]
        if x < -0.1:
            #offset = abs(x)
            print(f"X: {x}")
            print(f"Offset: {offset}")
        """
        pickup_pos = [self.config.pickup_x_coord + offset, self.target_info["initial_pos"][1], self.target_info["initial_pos"][2]]
        above_pos = [pickup_pos[0], pickup_pos[1], pickup_pos[2] + self.config.arm_above_offset]
        if self.arm_substate is None:
            logger.info(f"[ARM] Moving to pickup position: {pickup_pos}")
            logger.info(f"[ARM] Moving to above position: {above_pos}")
            move_arm_to(self.sim_env.kuka_id, self.sim_env.num_joints, above_pos)
            self.arm_substate = "wait_above"
        elif self.arm_substate == "wait_above":
            if wait_for_arm_to_reach(self.sim_env.kuka_id, above_pos, threshold=self.config.arm_threshold):
                logger.info(f"[ARM] Moving to final pickup position: {pickup_pos}")
                move_arm_to(self.sim_env.kuka_id, self.sim_env.num_joints, pickup_pos)
                self.arm_substate = "wait_pick"
        elif self.arm_substate == "wait_pick":
            if wait_for_arm_to_reach(self.sim_env.kuka_id, pickup_pos, threshold=self.config.arm_threshold):
                logger.info("[ARM] Grabbing object")
                self.constraint_id = grab_object(self.sim_env.kuka_id, self.object_id)
                self.picked = True
                self.tracking = True
                logger.info("[ARM] Object grabbed successfully")
                self.arm_substate = None
                self.picking_start_step = None  # Clear timeout counter
                self.state = ArmState.LIFTING

    def _handle_lifting(self):
        current_pos = p.getLinkState(self.sim_env.kuka_id, self.sim_env.num_joints - 1)[0]
        lift_pos = [current_pos[0], current_pos[1], self.config.arm_lift_height]
        if self.arm_substate is None:
            logger.info("[ARM] Lifting object")
            move_arm_to(self.sim_env.kuka_id, self.sim_env.num_joints, lift_pos)
            self.arm_substate = "wait_lift"
        elif self.arm_substate == "wait_lift":
            if wait_for_arm_to_reach(self.sim_env.kuka_id, lift_pos, threshold=self.config.arm_threshold):
                logger.info("[ARM] Moving to drop location")
                move_arm_to(self.sim_env.kuka_id, self.sim_env.num_joints, self.config.drop_position)
                self.arm_substate = "wait_drop"
        elif self.arm_substate == "wait_drop":
            # use a slightly larger threshold for drop
            if wait_for_arm_to_reach(self.sim_env.kuka_id, self.config.drop_position, threshold=self.config.arm_threshold):
                logger.info("[ARM] Releasing object")
                release_object(self.constraint_id)
                self.release_time = time.time()
                # log the detected class label for the object just picked up
                if self.last_results is not None and len(self.last_results) > 0 and hasattr(self.last_results[0], 'boxes'):
                    confs = self.last_results[0].boxes.conf.cpu().numpy()
                    if confs is not None and len(confs) > 0:
                        idx = np.argmax(confs)
                        class_idx = int(self.last_results[0].boxes.cls[idx].cpu().numpy())
                        class_name = self.sim_env.model.names[class_idx] if hasattr(self.sim_env.model, 'names') and class_idx < len(self.sim_env.model.names) else str(class_idx)
                        logger.info(f"[ARM] Successfully processed: {class_name}")
                self.wait_drop(0.25);
                self.picked = False
                self.tracking = False
                self.target_info = None
                self.constraint_id = None
                self.arm_substate = None
                # spawn new object after successful pickup
                self.object_id = self.object_loader.load_random_object()
                self.object_processed = False  # reset for new object
                self.state = ArmState.RESETTING

    def _handle_resetting(self):
        if self.arm_substate is None:
            logger.info("[ARM] Resetting arm to initial joint positions using forward kinematics")
            
            # use forward kinematics to move to initial joint positions
            move_arm_to_joint_positions(self.sim_env.kuka_id, self.sim_env.num_joints, self.sim_env.initial_joint_positions)
            self.arm_substate = "wait_joints"
            
        elif self.arm_substate == "wait_joints":
            # check if all joints have reached their initial positions
            current_joint_positions = []
            for joint_idx in range(self.sim_env.num_joints):
                current_pos = p.getJointState(self.sim_env.kuka_id, joint_idx)[0]
                current_joint_positions.append(current_pos)
            
            
            # check if joints have reached target positions
            if wait_for_joints_to_reach(self.sim_env.kuka_id, self.sim_env.initial_joint_positions, threshold=0.1):
                logger.info("[ARM] Successfully reset to initial joint positions!")
                
                # log final end effector position for verification
                final_ee_pos = p.getLinkState(self.sim_env.kuka_id, 6)[0]
                
                # clear yolo results to prevent old detections from staying in memory
                self.last_results = []
                self.last_img = None
                # clear the flag since arm finished processing trash object
                self.arm_processing_trash = False
                # don't load new object here - let the main loop handle it
                self.arm_substate = None
                self.state = ArmState.WAIT_FOR_OBJECT


if __name__ == "__main__":
    config = SimConfig()
    # enable frame capture for video generation
    controller = RobotController(config, headless=True, capture_frames=True)
    try:
        controller.run()
    except KeyboardInterrupt:
        logger.info("[SIM] Simulation stopped by user")
        # generate videos even if interrupted
        if controller.capture_frames:
            logger.info("[VIDEO] Generating videos from captured frames...")
            videos = create_videos_from_simulation(PROJECT_ROOT, fps=30, config=controller.config)
            if videos:
                logger.info("[VIDEO] Videos created:")
                for camera, video_path in videos.items():
                    logger.info(f"[VIDEO] {camera}: {video_path}")