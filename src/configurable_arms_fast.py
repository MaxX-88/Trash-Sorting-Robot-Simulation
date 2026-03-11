import pybullet as p
import numpy as np
import cv2
import time
import sys
import os
import glob
import argparse

# paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.simulation import SimConfig, ArmState, SimulationEnvironment, ObjectLoader
from src.control.pybullet_helpers import move_arm_to, wait_for_arm_to_reach, grab_object, release_object, move_arm_to_joint_positions, wait_for_joints_to_reach
from src.utils.debug_gui import DebugInterface
from src.utils.logger import get_logger
from src.utils.camera import TopDownCamera

logger = get_logger(__name__)


class ConfigurableRobotController:
    """
    N arms - configurable.
    FAST ver: yolo runs less often for speed
    """
    def __init__(self, num_arms: int, config: SimConfig, headless: bool = True, capture_frames: bool = False,
                 yolo_interval: int = 5, no_display: bool = False, render_interval: int = 1):
        """
        yolo_interval: run yolo every N frames (default 5)
        no_display: no cv2 windows for max speed
        render_interval: render cams every N frames
        """
        if num_arms < 1:
            raise ValueError("need >= 1 arm")
        
        self.num_arms = num_arms
        self.yolo_interval = yolo_interval
        self.no_display = no_display
        self.render_interval = render_interval
        
        # belt size calc based on arm count
        arm_spacing = 2.0
        total_arm_span = (num_arms - 1) * arm_spacing
        config.belt_length = max(8.0, total_arm_span + 4.0)
        config.counter_length = config.belt_length
        config.floor_size = max(10.0, config.belt_length + 2.0)
        
        config.spawn_x_position = -(config.belt_length / 2) + 1.0
        config.conveyor_start_x = -(config.belt_length / 2)
        config.conveyor_end_x = (config.belt_length / 2)
        config.yolo_zone_exit_x = (config.belt_length / 2)
        
        config.belt_position = [0, 0.1, 1.01]
        config.counter_position = [0, 0.1, 0.49]

        perspective_scale_factor = config.belt_length / 5
        top_scale_factor = config.belt_length / 6.5
        
        config.perspective_camera_position = [
            3 * perspective_scale_factor,
            -2 * perspective_scale_factor,
            4 + (perspective_scale_factor - 1) * 2
        ]
        
        config.top_camera_position = [
            0,
            0,
            6 * top_scale_factor
        ]
        
        logger.info(f"[SETUP] {num_arms} arms, belt={config.belt_length}")
        logger.info(f"[SETUP] YOLO interval: {yolo_interval} frames")
        logger.info(f"[SETUP] Render: {render_interval}, Display: {'OFF' if no_display else 'ON'}")
        logger.info(f"[SETUP] Cam scale: {perspective_scale_factor:.2f}, {top_scale_factor:.2f}")
        logger.info(f"[SETUP] Perspective: {config.perspective_camera_position}")
        logger.info(f"[SETUP] Top: {config.top_camera_position}")
        logger.info(f"[SETUP] Belt bounds: start={config.conveyor_start_x:.2f}, end={config.conveyor_end_x:.2f}")
        
        self.config = config
        self.capture_frames = capture_frames
        self.frame_count = 0
        
        self.sim_env = SimulationEnvironment(config, headless)
        self.object_loader = ObjectLoader(config)
        self.object_loader.set_belt_id(self.sim_env.belt_id)
        
        # remove defaults from sim
        if hasattr(self.sim_env, 'bin_trash'):
            p.removeBody(self.sim_env.bin_trash)
        if hasattr(self.sim_env, 'bin_recycling'):
            p.removeBody(self.sim_env.bin_recycling)
        if hasattr(self.sim_env, 'kuka_id'):
            p.removeBody(self.sim_env.kuka_id)
        if hasattr(self.sim_env, 'arm_counter_id'):
            p.removeBody(self.sim_env.arm_counter_id)
        
        # recycling bin
        recycling_bin_x = config.conveyor_end_x + 0.5
        recycling_bin_position = [recycling_bin_x, 0.1, 0.5]
        self.bin_recycling = p.loadURDF(
            config.trash_bin_urdf_path, 
            basePosition=recycling_bin_position, 
            globalScaling=config.bin_scale, 
            useFixedBase=True
        )
        for shape in p.getVisualShapeData(self.bin_recycling):
            link_index = shape[1]
            p.changeVisualShape(self.bin_recycling, link_index, rgbaColor=[0, 0, 1, 0.5])
        
        logger.info(f"[SETUP] Recycling bin X={recycling_bin_x:.2f}")
        
        # arm positions
        arm_positions = []
        if num_arms % 2 == 1:
            center = num_arms // 2
            for i in range(num_arms):
                x = (i - center) * arm_spacing
                arm_positions.append(x)
        else:
            for i in range(num_arms):
                x = (i - num_arms / 2 + 0.5) * arm_spacing
                arm_positions.append(x)
        
        logger.info(f"[SETUP] Arms at: {arm_positions}")
        
        self.arms = []
        self.objects = {}
        
        initial_joint_positions_template = self.sim_env.initial_joint_positions
        
        for i, x in enumerate(arm_positions):
            arm_position = [x, 0.6, 1]
            kuka_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=arm_position, useFixedBase=True)
            num_joints = p.getNumJoints(kuka_id)
            
            for link_index in range(-1, num_joints):  
                color = [1, 1, 1, 1] if link_index % 2 == 0 else [0, 0, 0, 1]
                p.changeVisualShape(kuka_id, link_index, rgbaColor=color)
            
            initial_joint_positions = initial_joint_positions_template
            for joint_idx in range(num_joints):
                p.resetJointState(kuka_id, joint_idx, initial_joint_positions[joint_idx])
            
            p.createMultiBody(0,
                p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.2, 0.5]),
                p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.2, 0.5], rgbaColor=[0.4, 0.4, 0.4, 1]),
                [x, 0.7, 0.5])
            
            bin_id = p.loadURDF(self.config.trash_bin_urdf_path, [x + 0.925, 1.1, 0.5], globalScaling=self.config.bin_scale, useFixedBase=True)
            for shape in p.getVisualShapeData(bin_id):
                p.changeVisualShape(bin_id, shape[1], rgbaColor=[0.7, 0.7, 0.7, 0.5])
            
            camera = TopDownCamera(self.config.detection_img_width, self.config.detection_img_height, [x, 0.03, 3], self.config.floor_plane_size)
            
            arm = {
                'id': i + 1,
                'x': x,
                'kuka_id': kuka_id,
                'num_joints': num_joints,
                'initial_joint_positions': initial_joint_positions,
                'camera': camera,
                'drop_position': [x + 0.9, 0.8, 1.5],
                'target_info': None,
                'target_object_id': None,
                'closest_object_id': None,
                'closest_object_name': None,
                'picked': False,
                'tracking': False,
                'constraint_id': None,
                'release_time': None,
                'object_processed': False,
                'arm_processing_trash': False,
                'arm_substate': None,
                'picking_start_step': None,
                'picking_timeout_steps': 70,
                'state': ArmState.WAIT_FOR_OBJECT,
                'previous_state': ArmState.IDLE,
                'last_img': None,
                'last_img_arr': None,
                'last_results': [],
                'depth': None,
            }
            self.arms.append(arm)
        
        self._setup_frame_capture()
        middle_arm = self.arms[len(self.arms) // 2]
        self.gui = DebugInterface(middle_arm['kuka_id'], middle_arm['num_joints'], self.config)
        self.simulation_step = 0

    def _setup_frame_capture(self):
        """setup frame dirs for capture"""
        self.spawn_side_arm = self.arms[0]
        
        if self.capture_frames:
            self.frames_base_dir = os.path.join(PROJECT_ROOT, "configurable_arms_vids", "frames")
            
            spawn_arm_frames_dir = os.path.join(self.frames_base_dir, f"arm_{self.spawn_side_arm['id']}")
            os.makedirs(spawn_arm_frames_dir, exist_ok=True)
            if os.path.exists(spawn_arm_frames_dir):
                for frame_file in os.listdir(spawn_arm_frames_dir):
                    if frame_file.endswith('.jpg'):
                        os.remove(os.path.join(spawn_arm_frames_dir, frame_file))
            self.spawn_side_arm['frames_dir'] = spawn_arm_frames_dir
            
            self.perspective_frames_dir = os.path.join(self.frames_base_dir, "perspective")
            os.makedirs(self.perspective_frames_dir, exist_ok=True)
            if os.path.exists(self.perspective_frames_dir):
                for frame_file in os.listdir(self.perspective_frames_dir):
                    if frame_file.endswith('.jpg'):
                        os.remove(os.path.join(self.perspective_frames_dir, frame_file))
            
            if self.config.enable_top_camera:
                self.top_frames_dir = os.path.join(self.frames_base_dir, "top")
                os.makedirs(self.top_frames_dir, exist_ok=True)
                if os.path.exists(self.top_frames_dir):
                    for frame_file in os.listdir(self.top_frames_dir):
                        if frame_file.endswith('.jpg'):
                            os.remove(os.path.join(self.top_frames_dir, frame_file))
            
            logger.info(f"[CAPTURE] Enabled for perspective + top + arm {self.spawn_side_arm['id']}")
    
    def _save_frame(self, perspective_img, arm_imgs):
        """save frames to disk"""
        if not self.capture_frames:
            return
        
        if perspective_img is not None:
            perspective_bgr = cv2.cvtColor(perspective_img, cv2.COLOR_RGB2BGR)
            perspective_filename = os.path.join(self.perspective_frames_dir, f"frame_{self.frame_count:06d}.jpg")
            cv2.imwrite(perspective_filename, perspective_bgr)
        
        if self.config.enable_top_camera and hasattr(self, 'top_img') and self.top_img is not None:
            top_bgr = cv2.cvtColor(self.top_img, cv2.COLOR_RGB2BGR)
            top_filename = os.path.join(self.top_frames_dir, f"frame_{self.frame_count:06d}.jpg")
            cv2.imwrite(top_filename, top_bgr)
        
        arm = self.spawn_side_arm
        if 'output_img' in arm and arm['output_img'] is not None:
            arm_bgr = cv2.cvtColor(arm['output_img'], cv2.COLOR_RGB2BGR)
            arm_filename = os.path.join(arm['frames_dir'], f"frame_{self.frame_count:06d}.jpg")
            cv2.imwrite(arm_filename, arm_bgr)
        
        self.frame_count += 1
    
    def _check_exit_key(self):
        if self.no_display:
            return False  # ctrl+c to exit when no display
        key = cv2.waitKey(1) & 0xFF
        return key == ord('q')
    
    def wait_drop(self, seconds):
        for _ in range(int(seconds * self.config.simulation_fps)):
            p.stepSimulation()

    def run(self):
        """
        main loop - yolo runs every yolo_interval frames
        """
        frame_count = 0
        sim_time = 0.0
        num_objects = 0
        
        last_log_time = time.time()
        
        while True:
            should_render = (frame_count % self.render_interval == 0)
            
            if should_render:
                self.perspective_img, self.perspective_img_arr = self.sim_env.perspective_camera.get_image()
                
                if self.config.enable_top_camera:
                    self.top_img, self.top_img_arr = self.sim_env.top_camera.get_image()
            
            # spawn objects
            if frame_count % (50//min(self.num_arms,5)) == 0:
                object_id = self.object_loader.load_random_object()
                _, is_trash, object_name = self.object_loader.get_last_object_info()
                obj_pos, _ = p.getBasePositionAndOrientation(object_id)
                self.objects[num_objects] = [object_id, obj_pos, is_trash, object_name]
                num_objects += 1
                obj_type = "TRASH" if is_trash else "RECYCLING"
                logger.info(f"[SPAWN] {obj_type}: {object_name} (ID: {object_id}), total: {num_objects}")

            # update obj positions
            for obj_key in list(self.objects.keys()):
                obj_id = self.objects[obj_key][0]
                updated_pos, _ = p.getBasePositionAndOrientation(obj_id)
                self.objects[obj_key][1] = updated_pos
            
            # yolo only every N frames (for speed)
            should_run_yolo = (frame_count % self.yolo_interval == 0)
            
            if should_run_yolo:
                for arm in self.arms:
                    arm['last_img'], arm['last_img_arr'] = arm['camera'].get_image()

                    if arm['last_img'] is not None:
                        try:
                            img_bgr = cv2.cvtColor(arm['last_img'], cv2.COLOR_RGB2BGR)
                            results = self.sim_env.model(img_bgr, verbose=False)
                            arm['last_results'] = results
                        except Exception as e:
                            logger.error(f"[ARM {arm['id']} YOLO] Error: {e}")
                            arm['last_results'] = []
            
            # belt velocity - push objects
            for obj_key, object in list(self.objects.items()):
                obj_id = object[0]
                obj_pos = object[1]

                is_held = any(arm['target_object_id'] == obj_id for arm in self.arms)

                if not is_held:
                    contacts = p.getContactPoints(bodyA=obj_id, bodyB=self.sim_env.belt_id)
                    if contacts:
                        p.resetBaseVelocity(obj_id, linearVelocity=[self.config.belt_velocity, 0, 0])

                # only remove when confirmed gone - past belt end or in bin (fixes early release bounce-back)
                if not is_held and (obj_pos[0] > self.config.conveyor_end_x or obj_pos[2] < 0.8):
                    del self.objects[obj_key]
            
            # fsm step
            for arm in self.arms:
                self._step_fsm(arm, sim_time)
                if arm['state'] != arm['previous_state']:
                    logger.info(f"[FSM ARM {arm['id']}] {arm['previous_state'].name} -> {arm['state'].name}")
                    arm['previous_state'] = arm['state']
            
            # display only when rendering (skip frames for perf)
            if should_render:
                if not self.no_display:
                    cv2.imshow("Perspective Camera", cv2.cvtColor(self.perspective_img, cv2.COLOR_RGB2BGR))
                
                for arm in self.arms:
                    if arm['last_img'] is not None and arm['last_img'].shape[0] > 0:
                        try:
                            output_img = arm['last_img'].copy()
                            
                            if len(arm['last_results']) > 0 and hasattr(arm['last_results'][0], 'boxes') and arm['last_results'][0].boxes is not None:
                                boxes = arm['last_results'][0].boxes.xyxy.cpu().numpy()
                                confs = arm['last_results'][0].boxes.conf.cpu().numpy()
                                classes = arm['last_results'][0].boxes.cls.cpu().numpy()
                                
                                for box, conf, cls in zip(boxes, confs, classes):
                                    x1, y1, x2, y2 = box
                                    class_idx = int(cls)
                                    class_name = self.sim_env.model.names[class_idx] if hasattr(self.sim_env.model, 'names') and class_idx < len(self.sim_env.model.names) else str(class_idx)
                                    
                                    cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                    
                                    label = f"{class_name} {conf:.2f}"
                                    cv2.putText(output_img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            debug_info = []
                            debug_info.append(f"Arm {arm['id']} (X={arm['x']:.1f})")
                            debug_info.append(f"Frame: {frame_count}")
                            debug_info.append(f"FSM: {arm['state'].name}")
                            debug_info.append(f"Closest Obj ID: {arm.get('closest_object_id')}")
                            debug_info.append(f"Target Obj ID: {arm.get('target_object_id')}")
                            debug_info.append(f"Arm Processing: {'YES' if arm['arm_processing_trash'] else 'NO'}")
                            
                            for i, info in enumerate(debug_info):
                                y_pos = 30 + i * 25
                                cv2.putText(output_img, info, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            arm['output_img'] = output_img
                            
                            if not self.no_display and arm['id'] == self.spawn_side_arm['id']:
                                cv2.imshow("Detection Camera", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
                        except Exception as e:
                            logger.error(f"[DISPLAY] Arm {arm['id']} error: {e}")
                
                if not self.no_display and self.config.enable_top_camera and hasattr(self, 'top_img') and self.top_img is not None:
                    cv2.imshow("Top Camera", cv2.cvtColor(self.top_img, cv2.COLOR_RGB2BGR))
                
                self._save_frame(self.perspective_img, None)
            
            if self._check_exit_key():
                logger.info("[SIM] q pressed")
                break
            frame_count += 1
            
            if self.config.enable_frame_logging and frame_count % self.config.frame_log_interval == 0:
                elapsed = time.time() - last_log_time
                fps = self.config.frame_log_interval / elapsed if elapsed > 0 else 0
                logger.info(f"[SIM] Frame: {frame_count}, FPS: {fps:.1f}")
                last_log_time = time.time()

            sim_time += 1.0 / self.config.simulation_fps
            for _ in range(3):
                p.stepSimulation()
                self.simulation_step += 1

    def _step_fsm(self, arm, sim_time):
        if arm['state'] == ArmState.IDLE:
            pass
        elif arm['state'] == ArmState.WAIT_FOR_OBJECT:
            self._handle_wait_for_object(arm, sim_time)
        elif arm['state'] == ArmState.PREPARE_PICK:
            self._handle_prepare_pick(arm, sim_time)
        elif arm['state'] == ArmState.PICKING:
            self._handle_picking(arm)
        elif arm['state'] == ArmState.LIFTING:
            self._handle_lifting(arm)
        elif arm['state'] == ArmState.RESETTING:
            self._handle_resetting(arm)

    def _handle_wait_for_object(self, arm, sim_time):
        """yolo detect - find target"""
        if arm['target_info'] is not None:
            logger.debug(f"[ARM {arm['id']}] Has target already")
            return
        
        if len(arm['last_results']) == 0:
            return
        
        boxes = arm['last_results'][0].boxes.xyxy.cpu().numpy() if hasattr(arm['last_results'][0], 'boxes') and arm['last_results'][0].boxes is not None else []
        confs = arm['last_results'][0].boxes.conf.cpu().numpy() if len(boxes) > 0 else []
        
        if len(boxes) == 0:
            return
        
        idx = np.argmax(confs)
        if confs[idx] < self.config.confidence_threshold:
            return
        
        center_x = int(arm['last_results'][0].boxes.xywh[idx][0].item())
        center_y = int(arm['last_results'][0].boxes.xywh[idx][1].item())

        class_idx = int(arm['last_results'][0].boxes.cls[idx].cpu().numpy())
        class_name = self.sim_env.model.names[class_idx] if hasattr(self.sim_env.model, 'names') and class_idx < len(self.sim_env.model.names) else str(class_idx)
        confidence = confs[idx]

        if class_name not in self.config.trash_classes:
            logger.info(f"[ARM {arm['id']} DETECT] '{class_name}' recyclable")
            arm['arm_processing_trash'] = False
            return

        arm_pickup_x = arm['x'] + 0.535
        arm_detection_line_x = arm['x'] - 1.0

        cam_pos = arm['camera']._camera_position
        detected_world_pos = arm['camera'].get_pixel_world_coords(center_x, center_y)
        detected_world_pos[0] += cam_pos[0]

        # depth from depth camera - only for the chosen object's bbox (like main.py)
        if arm['last_img_arr'] is not None:
            depth_buf = arm['last_img_arr'][3]
            h, w = arm['last_img'].shape[0], arm['last_img'].shape[1]
            if hasattr(depth_buf, 'reshape'):
                depth_buf = np.reshape(depth_buf, (h, w))
            else:
                depth_buf = np.array(depth_buf).reshape(h, w)
            x1, y1, x2, y2 = arm['last_results'][0].boxes.xyxy.cpu().numpy()[idx]
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
            depth_region = depth_buf[y1:y2+1, x1:x2+1]
            height_val = np.min(depth_region) if depth_region.size > 0 else 0.5
            near, far = arm['camera'].near, arm['camera'].far
            depth_z = cam_pos[2] - (far * near / (far - (far - near) * height_val))
            detected_world_pos.append(depth_z)
        else:
            detected_world_pos.append(1.3)

        logger.info(f"[ARM {arm['id']} DETECT] World: {detected_world_pos}")

        if detected_world_pos[0] > arm_detection_line_x:
            logger.info(f"[ARM {arm['id']} DETECT] Trash '{class_name}' at {detected_world_pos} (conf={confidence:.2f})")
            arm['target_info'] = {
                "initial_pos": detected_world_pos,
                "detection_time": sim_time,
                "class_name": class_name,
                "confidence": confidence
            }
            logger.info(f"[ARM {arm['id']} DETECT] Target acquired")
            arm['state'] = ArmState.PREPARE_PICK
        else:
            logger.debug(f"[ARM {arm['id']} DETECT] Tracking X={detected_world_pos[0]:.2f}")

    def _handle_prepare_pick(self, arm, sim_time):
        """pickup timing - when to go"""
        if arm['picked']:
            return
        
        if arm['target_info'] is None:
            arm['state'] = ArmState.WAIT_FOR_OBJECT
            return
        
        y = arm['target_info']['initial_pos'][1]
        obj_name = arm['target_info'].get('class_name', 'unknown')
        
        time_offset = 0
        if y < 0:
            time_offset = abs(y) * 1.5
        
        arm_pickup_x = arm['x'] + 0.535
        
        time_to_pickup = (arm_pickup_x - time_offset) / self.config.belt_velocity if self.config.belt_velocity > 0 else float('inf')
        if time_to_pickup < 0:
            time_to_pickup = 0.001
        
        logger.debug(f"[ARM {arm['id']}] {obj_name} - {time_to_pickup:.2f}s (y={y:.2f})")
        
        if 0 < time_to_pickup <= self.config.arm_lead_time:
            logger.info(f"[ARM {arm['id']}] >>> PICK: {obj_name}")
            arm['arm_processing_trash'] = True
            arm['picking_start_step'] = self.simulation_step
            arm['state'] = ArmState.PICKING
        else:
            logger.debug(f"[ARM {arm['id']}] Waiting - {time_to_pickup:.2f}s")

    def _handle_picking(self, arm):
        """picking - actually grab"""
        if arm['picking_start_step'] and (self.simulation_step - arm['picking_start_step']) > arm['picking_timeout_steps']:
            steps_elapsed = self.simulation_step - arm['picking_start_step']
            logger.warning(f"[ARM {arm['id']}] Timeout {steps_elapsed} steps")
            arm['picked'] = False
            arm['tracking'] = False
            arm['target_info'] = None
            if arm['constraint_id']:
                release_object(arm['constraint_id'])
                arm['constraint_id'] = None
            arm['arm_substate'] = None
            arm['picking_start_step'] = None
            arm['state'] = ArmState.RESETTING
            return
        
        if arm['target_info'] is None:
            arm['state'] = ArmState.RESETTING
            return
        
        arm_pickup_x = arm['x'] + 0.535
        pickup_pos = [arm_pickup_x, arm['target_info']['initial_pos'][1], arm['target_info']['initial_pos'][2]]
        above_pos = [pickup_pos[0], pickup_pos[1], pickup_pos[2] + self.config.arm_above_offset]
        
        if arm['arm_substate'] is None:
            logger.info(f"[ARM {arm['id']}] Moving: {pickup_pos}")
            logger.info(f"[ARM {arm['id']}] Above: {above_pos}")
            move_arm_to(arm['kuka_id'], arm['num_joints'], above_pos)
            arm['arm_substate'] = "wait_above"
        elif arm['arm_substate'] == "wait_above":
            if wait_for_arm_to_reach(arm['kuka_id'], above_pos, threshold=self.config.arm_threshold):
                logger.info(f"[ARM {arm['id']}] Down: {pickup_pos}")
                move_arm_to(arm['kuka_id'], arm['num_joints'], pickup_pos)
                arm['arm_substate'] = "wait_pick"
        elif arm['arm_substate'] == "wait_pick":
            if wait_for_arm_to_reach(arm['kuka_id'], pickup_pos, threshold=self.config.arm_threshold):
                closest_obj_id = None
                closest_dist = float('inf')
                for obj_key, obj_data in self.objects.items():
                    obj_id = obj_data[0]
                    obj_pos = obj_data[1]
                    dist = ((obj_pos[0] - pickup_pos[0])**2 + (obj_pos[1] - pickup_pos[1])**2)**0.5
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_obj_id = obj_id
                        arm['grabbed_obj_key'] = obj_key
                
                if closest_obj_id is not None and closest_dist < 0.5:
                    logger.info(f"[ARM {arm['id']}] Grab ID {closest_obj_id} (d={closest_dist:.2f})")
                    arm['constraint_id'] = grab_object(arm['kuka_id'], closest_obj_id)
                    arm['target_object_id'] = closest_obj_id
                    arm['picked'] = True
                    arm['tracking'] = True
                    logger.info(f"[ARM {arm['id']}] Got it")
                    arm['arm_substate'] = None
                    arm['picking_start_step'] = None
                    arm['state'] = ArmState.LIFTING
                else:
                    logger.warning(f"[ARM {arm['id']}] No obj (d={closest_dist:.2f})")
                    arm['arm_substate'] = None
                    arm['picking_start_step'] = None
                    arm['target_info'] = None
                    arm['state'] = ArmState.RESETTING

    def _handle_lifting(self, arm):
        # get current pos
        current_pos = p.getLinkState(arm['kuka_id'], arm['num_joints'] - 1)[0]
        lift_pos = [current_pos[0], current_pos[1], self.config.arm_lift_height]
        if arm['arm_substate'] is None:
            logger.info(f"[ARM {arm['id']}] Lift")
            move_arm_to(arm['kuka_id'], arm['num_joints'], lift_pos)
            arm['arm_substate'] = "wait_lift"
        elif arm['arm_substate'] == "wait_lift":
            if wait_for_arm_to_reach(arm['kuka_id'], lift_pos, threshold=self.config.arm_threshold):
                logger.info(f"[ARM {arm['id']}] To drop")
                move_arm_to(arm['kuka_id'], arm['num_joints'], arm['drop_position'])
                arm['arm_substate'] = "wait_drop"
        elif arm['arm_substate'] == "wait_drop":
            if wait_for_arm_to_reach(arm['kuka_id'], arm['drop_position'], threshold=self.config.arm_threshold):
                if arm.get('drop_reached_time') is None:
                    arm['drop_reached_time'] = time.time()
                    arm['arm_substate'] = "wait_before_drop"
        elif arm['arm_substate'] == "wait_before_drop":
            if (time.time() - arm['drop_reached_time']) >= self.config.arm_pause_before_drop:
                logger.info(f"[ARM {arm['id']}] Release {arm['target_object_id']}")
                release_object(arm['constraint_id'])
                arm['release_time'] = time.time()
                
                if arm['target_info'] is not None and 'class_name' in arm['target_info']:
                    logger.info(f"[ARM {arm['id']}] Done: {arm['target_info']['class_name']}")
                
                # don't remove from self.objects here - let cleanup do it when obj is past belt or in bin
                # (fixes early release: if obj bounces back on belt, it stays tracked and gets velocity)
                logger.info(f"[ARM {arm['id']}] Released {arm['target_object_id']}")

                arm['picked'] = False
                arm['tracking'] = False
                arm['target_info'] = None
                arm['target_object_id'] = None
                arm['constraint_id'] = None
                arm['arm_substate'] = None
                arm['drop_reached_time'] = None
                arm['state'] = ArmState.RESETTING

    def _handle_resetting(self, arm):
        # back to idle pose
        if arm['arm_substate'] is None:
            logger.info(f"[ARM {arm['id']}] Reset joints")
            move_arm_to_joint_positions(arm['kuka_id'], arm['num_joints'], arm['initial_joint_positions'])
            arm['arm_substate'] = "wait_joints"
            
        elif arm['arm_substate'] == "wait_joints":
            current_joint_positions = []
            for joint_idx in range(arm['num_joints']):
                current_pos = p.getJointState(arm['kuka_id'], joint_idx)[0]
                current_joint_positions.append(current_pos)
            
            if wait_for_joints_to_reach(arm['kuka_id'], arm['initial_joint_positions'], threshold=0.1):
                logger.info(f"[ARM {arm['id']}] Reset ok!")
                final_ee_pos = p.getLinkState(arm['kuka_id'], 6)[0]
                arm['last_results'] = []
                arm['last_img'] = None
                arm['arm_processing_trash'] = False
                arm['arm_substate'] = None
                arm['state'] = ArmState.WAIT_FOR_OBJECT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-arm sim (fast)')
    parser.add_argument('--num-arms', type=int, default=3, help='Arms (default: 3)')
    parser.add_argument('--no-video', action='store_true', help='No video')
    parser.add_argument('--yolo-interval', type=int, default=5, help='YOLO every N frames (default: 5)')
    parser.add_argument('--no-display', action='store_true', help='No windows')
    parser.add_argument('--render-interval', type=int, default=1, help='Render every N frames')
    args = parser.parse_args()
    
    num_arms = args.num_arms
    capture_frames = not args.no_video
    
    logger.info(f"[MAIN] FAST sim with {num_arms} arms")
    logger.info(f"[MAIN] Video: {'on' if capture_frames else 'off'}")
    logger.info(f"[MAIN] YOLO: {args.yolo_interval}, Render: {args.render_interval}, Display: {'off' if args.no_display else 'on'}")
    
    config = SimConfig()
    controller = ConfigurableRobotController(num_arms, config, headless=True, capture_frames=capture_frames,
                                            yolo_interval=args.yolo_interval, no_display=args.no_display,
                                            render_interval=args.render_interval)
    
    try:
        controller.run()
    except KeyboardInterrupt:
        logger.info("[SIM] Stopped")
    finally:
        if controller.capture_frames:
            from src.video_generator import create_video_from_frames
            
            logger.info("[VIDEO] Making vids...")
            
            video_fps = int(controller.config.simulation_fps / controller.render_interval)
            logger.info(f"[VIDEO] FPS: {video_fps} (sim={controller.config.simulation_fps}, render={controller.render_interval})")
            
            videos_base_dir = os.path.join(PROJECT_ROOT, "configurable_arms_vids")
            run_number = 1
            while os.path.exists(os.path.join(videos_base_dir, f"Run{run_number}")):
                run_number += 1
            
            run_dir = os.path.join(videos_base_dir, f"Run{run_number}")
            os.makedirs(run_dir, exist_ok=True)
            logger.info(f"[VIDEO] Run{run_number}...")
            
            perspective_video = os.path.join(run_dir, "perspective_simulation.mp4")
            if create_video_from_frames(controller.perspective_frames_dir, perspective_video, fps=video_fps):
                logger.info(f"[VIDEO] Perspective: {perspective_video}")
            
            if controller.config.enable_top_camera and hasattr(controller, 'top_frames_dir'):
                top_video = os.path.join(run_dir, "top_simulation.mp4")
                if create_video_from_frames(controller.top_frames_dir, top_video, fps=video_fps):
                    logger.info(f"[VIDEO] Top: {top_video}")
            
            arm = controller.spawn_side_arm
            arm_video = os.path.join(run_dir, f"arm_{arm['id']}_simulation.mp4")
            if create_video_from_frames(arm['frames_dir'], arm_video, fps=video_fps):
                logger.info(f"[VIDEO] Arm {arm['id']}: {arm_video}")
            
            logger.info("[VIDEO] Cleanup...")
            frame_dirs = [controller.perspective_frames_dir, controller.spawn_side_arm['frames_dir']]
            if controller.config.enable_top_camera and hasattr(controller, 'top_frames_dir'):
                frame_dirs.append(controller.top_frames_dir)
            for frame_dir in frame_dirs:
                if os.path.exists(frame_dir):
                    frame_files = glob.glob(os.path.join(frame_dir, "frame_*.jpg"))
                    for frame_file in frame_files:
                        os.remove(frame_file)
                    logger.info(f"[VIDEO] Cleaned {len(frame_files)} from {os.path.basename(frame_dir)}")
            
            logger.info(f"[VIDEO] Done -> {run_dir}")
