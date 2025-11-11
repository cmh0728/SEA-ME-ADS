#!/usr/bin/env python3
import os
from functools import partial
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge

# Import lane detector modules
try:
    from lane_detector_py.binary import DEFAULT_PARAMS, create_lane_mask
    from lane_detector_py.birdseye import compute_homography, warp_to_top_view
    from lane_detector_py.sliding_window import fit_polynomial, sliding_window_search
    from lane_detector_py.visualization import (
        draw_lane_overlay,
        render_sliding_window_debug,
    )
except ImportError:  # pragma: no cover
    from .binary import DEFAULT_PARAMS, create_lane_mask
    from .birdseye import compute_homography, warp_to_top_view
    from .sliding_window import fit_polynomial, sliding_window_search
    from .visualization import draw_lane_overlay, render_sliding_window_debug


class PolynomialKalmanFilter:
    """Simple constant-model Kalman filter for lane polynomial coefficients."""

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 5e-2,
        max_prediction_frames: int = 5,
    ):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.max_prediction_frames = int(max(1, max_prediction_frames))
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.missed = 0

    def reset(self):
        self.x = None
        self.P = None
        self.missed = 0

    def update(self, measurement: np.ndarray) -> np.ndarray:
        meas = np.asarray(measurement, dtype=np.float64).reshape(3, 1)
        if self.x is None or self.P is None:
            self.x = meas.copy()
            self.P = np.eye(3, dtype=np.float64)
            self.missed = 0
            return self.x.ravel()

        self._predict_covariance()
        y = meas - self.x
        S = self.P + self._measurement_cov()
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K) @ self.P
        self.missed = 0
        return self.x.ravel()

    def predict(self) -> Optional[np.ndarray]:
        if self.x is None or self.P is None:
            return None
        if self.missed >= self.max_prediction_frames:
            return None
        self._predict_covariance()
        self.missed += 1
        return self.x.ravel()

    def _predict_covariance(self):
        if self.P is None:
            return
        self.P = self.P + self._process_cov()

    def _process_cov(self) -> np.ndarray:
        return np.eye(3, dtype=np.float64) * self.process_noise

    def _measurement_cov(self) -> np.ndarray:
        return np.eye(3, dtype=np.float64) * self.measurement_noise


# Lane Detector Node Class
class LaneDetectorNode(Node):
    def __init__(self): # 노드 생성시 한번만 실행
        super().__init__('lane_detector')

        # 파라미터서버에 파라미터 등록 및 기본값 설정
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('publish_overlay_topic', '/lane/overlay')
        self.declare_parameter('publish_offset_topic', '/lane/center_offset')
        self.declare_parameter('publish_heading_topic', '/lane/heading_offset')
        self.declare_parameter('use_birdeye', True)
        self.declare_parameter('enable_visualization', False) # 디버깅용 시각화 여부 파라미터 , 기본값 False
        self.declare_parameter('lane_width_px', 650.0) # 차선폭 650px 기본 설정 
        self.declare_parameter('vehicle_center_bias_px', -40.0)  # 이미지 중심 대비 차량 중심 보정값
        self.declare_parameter('fit_smoothing_alpha', 0.2)
        self.declare_parameter('use_high_contrast_preprocess', True)
        self.declare_parameter('high_contrast_threshold', 170.0)
        self.declare_parameter('high_contrast_blur_kernel', 3)
        self.declare_parameter('high_contrast_dilate_kernel', 5)
        self.declare_parameter('high_contrast_canny_low', 100.0)
        self.declare_parameter('high_contrast_canny_high', 360.0)
        self.declare_parameter('enable_remap_ipm', False)
        self.declare_parameter('remap_map_x_path', '')
        self.declare_parameter('remap_map_y_path', '')
        self.declare_parameter('remap_map_delimiter', '\t')
        self.declare_parameter('use_kalman_smoothing', True)
        self.declare_parameter('kalman_process_noise', 1e-4)
        self.declare_parameter('kalman_measurement_noise', 5e-2)
        self.declare_parameter('kalman_max_prediction_frames', 5)
        self.declare_parameter('publish_offset_in_meters', False)
        self.declare_parameter('pixel_to_meter', 5.38e-4)
        self.crop_size = (860, 480)
        self.last_frame_shape = None
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.lane_width_px = float(self.get_parameter('lane_width_px').get_parameter_value().double_value)
        self.lane_width_cm = 35.0  # 실제 차폭 (cm)
        self._last_logged_lane_width = None # 픽셀 차폭계산용 
        self._measured_lane_width_px = self.lane_width_px
        self.vehicle_center_bias_px = float(
            self.get_parameter('vehicle_center_bias_px').get_parameter_value().double_value
        )
        self._log_lane_width_if_needed(self.lane_width_px)

        # 버드아이용 호모그래피(예시 좌표: 해상도 640x480 전제)
        # 실제 카메라 및 트랙에 맞게 보정 필요 
        # src_points : 원본 카메라 이미지에서 변환에 사용할 4개의 점 
        # dst_points : 버드아이뷰에서 대응되는 4개의 점[x0, y0, x1, y1, x2, y2, x3, y3
        self.declare_parameter('src_points', [0.0, 400.0, 212.0, 165.0, 618.0, 165.0, 860.0, 400.0]) # 11.07 수정된 기본값 최종(카메라 꺾어서)
        self.declare_parameter('dst_points', [0.0,  480.0, 0.0,   0.0, 860.0, 0.0, 860.0, 480.0])

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.subscribe_compressed = image_topic.endswith('/compressed')
        # print(self.subscribe_compressed) # check
        # overlay_topic = self.get_parameter('publish_overlay_topic').get_parameter_value().string_value
        offset_topic = self.get_parameter('publish_offset_topic').get_parameter_value().string_value
        self.use_birdeye = self.get_parameter('use_birdeye').get_parameter_value().bool_value
        self.visualize = self.get_parameter('enable_visualization').get_parameter_value().bool_value
        self.fit_smoothing_alpha = float(self.get_parameter('fit_smoothing_alpha').value)
        self.use_high_contrast = bool(self.get_parameter('use_high_contrast_preprocess').value)
        self.high_contrast_cfg = {
            'threshold': float(self.get_parameter('high_contrast_threshold').value),
            'blur_kernel': int(self.get_parameter('high_contrast_blur_kernel').value),
            'dilate_kernel': int(self.get_parameter('high_contrast_dilate_kernel').value),
            'canny_low': float(self.get_parameter('high_contrast_canny_low').value),
            'canny_high': float(self.get_parameter('high_contrast_canny_high').value),
        }
        self.enable_remap = bool(self.get_parameter('enable_remap_ipm').value)
        self.remap_map_x_path = str(self.get_parameter('remap_map_x_path').value)
        self.remap_map_y_path = str(self.get_parameter('remap_map_y_path').value)
        self.remap_map_delimiter = str(self.get_parameter('remap_map_delimiter').value or '\t')
        self.publish_offset_in_meters = bool(self.get_parameter('publish_offset_in_meters').value)
        self.pixel_to_meter = float(self.get_parameter('pixel_to_meter').value)
        if self.publish_offset_in_meters and self.pixel_to_meter <= 0.0:
            self.get_logger().warn('pixel_to_meter must be positive; disabling meter conversion.')
            self.publish_offset_in_meters = False
        self._offset_unit_logged = False

        self.src_pts = np.array(self.get_parameter('src_points').value, dtype=np.float32).reshape(4, 2)
        self.dst_pts = np.array(self.get_parameter('dst_points').value, dtype=np.float32).reshape(4, 2)

        # QoS: 센서데이터는 BestEffort/Depth=1이 지연/버퍼에 유리
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge() # change ros imgmsg <-> cv2

        # self.pub_overlay = self.create_publisher(Image, overlay_topic, qos) # 차선 오버레이 이미지 퍼블리셔 --> 필요없어
        self.pub_offset = self.create_publisher(Float32, offset_topic, 10)
        heading_topic = self.get_parameter('publish_heading_topic').get_parameter_value().string_value
        self.pub_heading = self.create_publisher(Float32, heading_topic, 10)

        if self.subscribe_compressed: # compressed image --> 실제 실행되는 부분
            self.sub = self.create_subscription(CompressedImage, image_topic, self.image_cb_compressed, qos)
        else: # raw image
            self.sub = self.create_subscription(Image, image_topic, self.image_cb_raw, qos)

        # 호모그래피 미리 계산(버드아이뷰 변환에 사용할 행렬)--> 프레임 계산을 줄이기 위해 한번만 실행 
        self.H, self.Hinv = self._compute_homography()
        self.remap_ready = False
        self.remap_map_x: Optional[np.ndarray] = None
        self.remap_map_y: Optional[np.ndarray] = None
        if self.enable_remap:
            self._load_remap_maps()

        self.use_kalman = bool(self.get_parameter('use_kalman_smoothing').value)
        process_noise = float(self.get_parameter('kalman_process_noise').value)
        measurement_noise = float(self.get_parameter('kalman_measurement_noise').value)
        max_preds = int(self.get_parameter('kalman_max_prediction_frames').value)
        if self.use_kalman:
            self.left_kalman = PolynomialKalmanFilter(process_noise, measurement_noise, max_preds)
            self.right_kalman = PolynomialKalmanFilter(process_noise, measurement_noise, max_preds)
        else:
            self.left_kalman = None
            self.right_kalman = None

        # 디버깅용 이미지 표시 창과 마우스 콜백 설정
        self.window_name = 'lane_detector_input'
        self.control_window_src = 'homography_controls_src'
        self.control_window_dst = 'homography_controls_dst'
        self.birdeye_window = 'wrapped_img'
        self.overlay_window = 'lane_overlay'
        self.binary_control_window = 'binary_controls'

        # trackbar param
        self.homography_ui_ready = False # 트랙바 한번만 생성
        self.binary_ui_ready = False
        self._trackbar_lock = False

        self.binary_params = dict(DEFAULT_PARAMS)
        self._binary_trackbar_names = {
            'clip_limit': 'clip_limit_x10',
            'tile_grid': 'tile_grid',
            'blur_kernel': 'blur_kernel',
            'gray_thresh': 'gray_thresh',
            'sat_thresh': 'sat_thresh',
            'canny_low': 'canny_low',
            'canny_high': 'canny_high',
            'white_v_min': 'white_v_min',
            'white_s_max': 'white_s_max',
        }


        sub_type = 'CompressedImage' if self.subscribe_compressed else 'Image'
        self.get_logger().info(f'LaneDetector subscribing: {image_topic} ({sub_type})')
        # self.get_logger().info(f'Publishing overlay: {overlay_topic}, center_offset: {offset_topic}')


    #####################################  homography  ####################################################################

    def _compute_homography(self):
        return compute_homography(self.src_pts, self.dst_pts, self.use_birdeye)

    # 차폭 계산 로직 
    def _log_lane_width_if_needed(self, width: float):
        if width is None: # 양쪽 둘중 하나라도 차선못잡았을때 제외 
            return
        if self._last_logged_lane_width is None or abs(width - self._last_logged_lane_width) >= 1.0:
            self._last_logged_lane_width = width
            # self.get_logger().info(f'Estimated lane width (px): {width:.2f}')

    def _load_remap_maps(self):
        if not self.remap_map_x_path or not self.remap_map_y_path:
            self.get_logger().warn('enable_remap_ipm is true but remap_map_x_path/remap_map_y_path are empty.')
            self.enable_remap = False
            return
        try:
            self.remap_map_x = self._load_map_file(self.remap_map_x_path)
            self.remap_map_y = self._load_map_file(self.remap_map_y_path)
        except Exception as exc:  # pragma: no cover - file errors are runtime only
            self.get_logger().warn(f'Failed to load remap maps: {exc}')
            self.enable_remap = False
            self.remap_ready = False
            return

        if self.remap_map_x.shape != self.remap_map_y.shape:
            self.get_logger().warn('Remap map shapes do not match; disabling remap.')
            self.enable_remap = False
            self.remap_ready = False
            return

        if self.remap_map_x.dtype != np.float32:
            self.remap_map_x = self.remap_map_x.astype(np.float32)
        if self.remap_map_y.dtype != np.float32:
            self.remap_map_y = self.remap_map_y.astype(np.float32)

        self.remap_ready = True
        self.get_logger().info(
            f'Loaded remap maps ({self.remap_map_x.shape[1]}x{self.remap_map_x.shape[0]}) for IPM.'
        )

    def _load_map_file(self, path: str) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        _, ext = os.path.splitext(path.lower())
        if ext == '.npy':
            data = np.load(path)
        else:
            delimiter = None if not self.remap_map_delimiter else self.remap_map_delimiter
            kwargs = {}
            if delimiter:
                kwargs['delimiter'] = delimiter
            data = np.loadtxt(path, **kwargs)
        return np.asarray(data, dtype=np.float32)

    def _apply_remap(self, image: np.ndarray) -> Optional[np.ndarray]:
        if not self.remap_ready or self.remap_map_x is None or self.remap_map_y is None:
            return None
        try:
            return cv2.remap(
                image,
                self.remap_map_x,
                self.remap_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        except cv2.error as exc:  # pragma: no cover
            self.get_logger().warn(f'Remap failed: {exc}. Disabling remap.')
            self.remap_ready = False
            return None

    def _center_crop(self, bgr: np.ndarray) -> np.ndarray:
        crop_w, crop_h = self.crop_size
        cur_h, cur_w, _ = bgr.shape
        if cur_w >= crop_w and cur_h >= crop_h:
            x0 = (cur_w - crop_w) // 2
            y0 = (cur_h - crop_h) // 2
            bgr = bgr[y0:y0 + crop_h, x0:x0 + crop_w]
        else:
            self.get_logger().warn(
                f'Incoming image smaller than crop size ({cur_w}x{cur_h} < {crop_w}x{crop_h}); skipping center crop.')

        cur_h, cur_w, _ = bgr.shape
        if cur_w > crop_w:
            x0 = (cur_w - crop_w) // 2
            bgr = bgr[:, x0:x0 + crop_w]
        elif cur_w < crop_w:
            self.get_logger().warn(
                f'Incoming image narrower than crop width ({cur_w} < {crop_w}); skipping horizontal crop.')
        return bgr

    def _high_contrast_binary(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur_k = max(1, int(self.high_contrast_cfg['blur_kernel']))
        if blur_k % 2 == 0:
            blur_k += 1
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        thresh = int(np.clip(self.high_contrast_cfg['threshold'], 0, 255))
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        kernel_size = max(1, int(self.high_contrast_cfg['dilate_kernel']))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        binary = cv2.dilate(binary, kernel, iterations=1)
        canny_low = int(np.clip(self.high_contrast_cfg['canny_low'], 0, 254))
        canny_high = int(np.clip(self.high_contrast_cfg['canny_high'], canny_low + 1, 255))
        edges = cv2.Canny(gray, canny_low, canny_high)
        combo = np.zeros_like(binary)
        combo[(binary == 255) | (edges == 255)] = 255
        return combo

    def _warp_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.remap_ready:
            return mask
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        warped = warp_to_top_view(mask, self.H) if self.H is not None else mask
        if warped.ndim == 3:
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        return warped

    def _filter_fit(
        self,
        raw_fit: Optional[np.ndarray],
        prev_fit: Optional[np.ndarray],
        kalman_filter: Optional[PolynomialKalmanFilter],
    ) -> Optional[np.ndarray]:
        if raw_fit is None:
            if kalman_filter is not None:
                prediction = kalman_filter.predict()
                if prediction is not None:
                    return prediction.copy()
            return prev_fit.copy() if prev_fit is not None else None

        measurement = np.asarray(raw_fit, dtype=np.float64)
        if kalman_filter is not None:
            return kalman_filter.update(measurement)

        if prev_fit is None:
            return measurement
        alpha = np.clip(self.fit_smoothing_alpha, 0.0, 1.0)
        return (1.0 - alpha) * prev_fit + alpha * measurement

    def _convert_offset_units(self, offset_px: float) -> float:
        if not self.publish_offset_in_meters or not np.isfinite(offset_px):
            return offset_px
        if not self._offset_unit_logged:
            self.get_logger().info(
                f'Publishing offset in meters (pixel_to_meter={self.pixel_to_meter:.3e}).'
            )
            self._offset_unit_logged = True
        return offset_px * self.pixel_to_meter
    
    def _ensure_homography_ui(self):
        if not self.use_birdeye or self.homography_ui_ready or self.last_frame_shape is None:
            return

        ref_w, ref_h = self.last_frame_shape
        ref_w = max(1, ref_w)
        ref_h = max(1, ref_h)

        cv2.namedWindow(self.control_window_src, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.control_window_dst, cv2.WINDOW_AUTOSIZE)

        for idx in range(4):
            self._create_homography_trackbar('src', idx, 0, ref_w)
            self._create_homography_trackbar('src', idx, 1, ref_h)
            self._create_homography_trackbar('dst', idx, 0, ref_w)
            self._create_homography_trackbar('dst', idx, 1, ref_h)

        self.homography_ui_ready = True

    def _ensure_binary_ui(self):
        if not self.visualize or self.binary_ui_ready:
            return

        cv2.namedWindow(self.binary_control_window, cv2.WINDOW_AUTOSIZE)

        def _create(name: str, value: int, max_val: int, key: str):
            value = int(np.clip(value, 0, max_val))
            cv2.createTrackbar(
                name,
                self.binary_control_window,
                value,
                max_val,
                partial(self._on_binary_trackbar, key)
            )
            self._on_binary_trackbar(key, value)

        _create(self._binary_trackbar_names['clip_limit'], int(round(self.binary_params['clip_limit'] * 10)), 100, 'clip_limit')
        _create(self._binary_trackbar_names['tile_grid'], int(self.binary_params['tile_grid']), 40, 'tile_grid')
        _create(self._binary_trackbar_names['blur_kernel'], int(self.binary_params['blur_kernel']), 31, 'blur_kernel')
        _create(self._binary_trackbar_names['gray_thresh'], int(self.binary_params['gray_thresh']), 255, 'gray_thresh')
        _create(self._binary_trackbar_names['sat_thresh'], int(self.binary_params['sat_thresh']), 255, 'sat_thresh')
        _create(self._binary_trackbar_names['canny_low'], int(self.binary_params['canny_low']), 255, 'canny_low')
        _create(self._binary_trackbar_names['canny_high'], int(self.binary_params['canny_high']), 255, 'canny_high')
        _create(self._binary_trackbar_names['white_v_min'], int(self.binary_params['white_v_min']), 255, 'white_v_min')
        _create(self._binary_trackbar_names['white_s_max'], int(self.binary_params['white_s_max']), 255, 'white_s_max')

        self.binary_ui_ready = True

    def _create_homography_trackbar(self, point_type: str, idx: int, axis: int, max_val: int):
        arr = self.src_pts if point_type == 'src' else self.dst_pts
        track_name = f'{point_type}_{"x" if axis == 0 else "y"}{idx}'
        max_slider = max(1, max_val - 1)
        initial = int(np.clip(arr[idx, axis], 0, max_slider))
        arr[idx, axis] = float(initial)
        window_name = self.control_window_src if point_type == 'src' else self.control_window_dst
        cv2.createTrackbar(
            track_name,
            window_name,
            initial,
            max_slider,
            partial(self._on_homography_trackbar, point_type, idx, axis, track_name, window_name)
        )

    def _on_homography_trackbar(
        self,
        point_type: str,
        idx: int,
        axis: int,
        track_name: str,
        window_name: str,
        value: int,
    ):
        if self._trackbar_lock:
            return

        arr = self.src_pts if point_type == 'src' else self.dst_pts
        if self.last_frame_shape:
            ref_w, ref_h = self.last_frame_shape
        else:
            ref_w, ref_h = self.crop_size
        max_val = (ref_w - 1) if axis == 0 else (ref_h - 1)
        clipped = float(np.clip(value, 0, max_val))

        if clipped != value:
            try:
                self._trackbar_lock = True
                cv2.setTrackbarPos(track_name, window_name, int(clipped))
            finally:
                self._trackbar_lock = False

        arr[idx, axis] = clipped
        self.H, self.Hinv = self._compute_homography()

    def _set_binary_trackbar(self, key: str, value: int):
        name = self._binary_trackbar_names[key]
        try:
            self._trackbar_lock = True
            cv2.setTrackbarPos(name, self.binary_control_window, int(value))
        finally:
            self._trackbar_lock = False

    def _on_binary_trackbar(self, key: str, value: int):
        if self._trackbar_lock:
            return

        if key == 'clip_limit':
            value = max(1, value)
            if value != int(round(self.binary_params['clip_limit'] * 10)):
                self.binary_params['clip_limit'] = value / 10.0
            if value != cv2.getTrackbarPos(self._binary_trackbar_names[key], self.binary_control_window):
                self._set_binary_trackbar(key, value)
            return

        if key == 'tile_grid':
            value = max(2, value)
            self.binary_params['tile_grid'] = value
            if value != cv2.getTrackbarPos(self._binary_trackbar_names[key], self.binary_control_window):
                self._set_binary_trackbar(key, value)
            return

        if key == 'blur_kernel':
            value = max(1, value)
            if value % 2 == 0:
                value = value + 1 if value < 31 else value - 1
            self.binary_params['blur_kernel'] = value
            if value != cv2.getTrackbarPos(self._binary_trackbar_names[key], self.binary_control_window):
                self._set_binary_trackbar(key, value)
            return

        if key in ('gray_thresh', 'sat_thresh', 'white_v_min', 'white_s_max'):
            self.binary_params[key] = int(np.clip(value, 0, 255))
            return

        if key == 'canny_low':
            value = int(np.clip(value, 0, 254))
            self.binary_params['canny_low'] = value
            high = int(self.binary_params['canny_high'])
            if high <= value:
                high = min(255, value + 1)
                self.binary_params['canny_high'] = high
                self._set_binary_trackbar('canny_high', high)
            return

        if key == 'canny_high':
            low = int(self.binary_params['canny_low'])
            value = int(np.clip(value, low + 1, 255))
            self.binary_params['canny_high'] = value
            if value != cv2.getTrackbarPos(self._binary_trackbar_names[key], self.binary_control_window):
                self._set_binary_trackbar(key, value)
            return

    ####################################  image change cv <--> ROS ############################################################

    # cv image bridge raw
    def image_cb_raw(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return
        self._process_frame(bgr)

    # cv image bridge compressed 
    def image_cb_compressed(self, msg: CompressedImage):
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge (compressed) error: {e}')
            return
        
        # cv2.imshow('raw compressed img', bgr) # check image 
        # print(bgr.shape[0], bgr.shape[1])  # default size is 720, 1280 --> 1 is wide 0 is height
        self._process_frame(bgr)

    ################################   image processing main function ########################################################

    def _process_frame(self, bgr: np.ndarray, *, visualize: bool = None):
        viz_enabled = self.visualize if visualize is None else visualize

        # 1) 영상 준비 (리맵 사용 시 원본 그대로 / 아니면 중앙 크롭)
        working_bgr = bgr if self.remap_ready else self._center_crop(bgr)
        remapped_bgr = self._apply_remap(working_bgr)
        processing_bgr = remapped_bgr if remapped_bgr is not None else working_bgr

        # track bar options (옵션으로만 활성화)
        # self._ensure_homography_ui()
        # self._ensure_binary_ui()

        # 2) 전처리 → 이진 마스크
        mask = create_lane_mask(processing_bgr, self.binary_params)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if self.use_high_contrast:
            high_contrast = self._high_contrast_binary(processing_bgr)
            mask = cv2.bitwise_or(mask, high_contrast)

        # 3) 버드아이뷰 변환 (리맵 사용 시 그대로 활용)
        top = self._warp_mask(mask)
        if top.ndim == 3:
            top = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        lane_h, lane_w = top.shape[:2]
        self.last_frame_shape = (lane_w, lane_h)
        
        # 6) 차선 검출 부분 
        # 슬라이딩 윈도우 → 피팅
        (lx, ly), (rx, ry), window_records = sliding_window_search(top)
        left_fit_raw = fit_polynomial((lx, ly))
        right_fit_raw = fit_polynomial((rx, ry))
        left_fit = self._filter_fit(left_fit_raw, self.prev_left_fit, self.left_kalman)
        right_fit = self._filter_fit(right_fit_raw, self.prev_right_fit, self.right_kalman)

        if left_fit is not None:
            self.prev_left_fit = left_fit.copy()
        if right_fit is not None:
            self.prev_right_fit = right_fit.copy()

        if self.lane_width_px is not None:
            if left_fit is not None and right_fit is None:
                right_fit = left_fit.copy()
                right_fit[2] += self.lane_width_px
            elif right_fit is not None and left_fit is None:
                left_fit = right_fit.copy()
                left_fit[2] -= self.lane_width_px

        # 차 폭/센터 오프셋 계산 (픽셀 기준 → 미터 변환은 사용자 설정)
        center_offset_px = float("nan")
        y_eval = lane_h - 1

        def _eval_fit(fit):
            if fit is None:
                return None
            return float(fit[0]*y_eval*y_eval + fit[1]*y_eval + fit[2])

        lane_center = None
        img_center = lane_w / 2.0 + self.vehicle_center_bias_px
        have_left = left_fit is not None
        have_right = right_fit is not None

        def _eval_slope(fit):
            if fit is None:
                return None
            return float(2.0 * fit[0] * y_eval + fit[1])

        if have_left and have_right:
            x_left = _eval_fit(left_fit)
            x_right = _eval_fit(right_fit)
            if x_left is not None and x_right is not None:
                lane_center = (x_left + x_right) / 2.0
                self._measured_lane_width_px = float(x_right - x_left)
                self._log_lane_width_if_needed(self._measured_lane_width_px)
        elif self._measured_lane_width_px is not None:
            half_width = self._measured_lane_width_px / 2.0
            if have_left:
                x_left = _eval_fit(left_fit)
                if x_left is not None:
                    lane_center = x_left + half_width
            elif have_right:
                x_right = _eval_fit(right_fit)
                if x_right is not None:
                    lane_center = x_right - half_width

        lane_center_point_top = (lane_center, y_eval) if lane_center is not None else None

        if lane_center is not None:
            center_offset_px = float(img_center - lane_center)

        lane_slope = None
        left_slope = _eval_slope(left_fit) if have_left else None
        right_slope = _eval_slope(right_fit) if have_right else None
        if left_slope is not None and right_slope is not None:
            lane_slope = 0.5 * (left_slope + right_slope)
        else:
            lane_slope = left_slope if left_slope is not None else right_slope
        heading_offset_rad = float(np.arctan(lane_slope)) if lane_slope is not None else float("nan")

        if viz_enabled:
            # cv2.imshow(self.window_name, bgr)
            debug_view = render_sliding_window_debug(
                top, window_records, (lx, ly), (rx, ry), lane_center_point=lane_center_point_top
            )
            # cv2.imshow("mask",mask)
            cv2.imshow(self.birdeye_window, debug_view)

            if not self.remap_ready:
                fill_overlay = have_left and have_right
                overlay = draw_lane_overlay(
                    working_bgr,
                    top,
                    self.Hinv,
                    left_fit,
                    right_fit,
                    fill=fill_overlay,
                    lane_center_point=lane_center_point_top,
                    vehicle_center_px=img_center,
                )
                cv2.imshow(self.overlay_window, overlay)

            # if you want to publish overlay image, uncomment below
            # self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8'))
            cv2.waitKey(1)

        # 퍼블리시
        offset_value = self._convert_offset_units(center_offset_px)
        self.pub_offset.publish(Float32(data=float(offset_value)))
        self.pub_heading.publish(Float32(data=heading_offset_rad))


    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.get_logger().info(f'Mouse click at ({x}, {y})')

def main():
    rclpy.init()
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# 실제 차폭 --> 픽셀 차폭 계산필요 
# 실제 차폭 : 35cm 정도 
