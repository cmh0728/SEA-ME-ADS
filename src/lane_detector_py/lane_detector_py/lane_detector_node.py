#!/usr/bin/env python3
from functools import partial
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
    from lane_detector_py.binary import create_lane_mask
    from lane_detector_py.birdseye import compute_homography, warp_to_top_view
    from lane_detector_py.sliding_window import fit_polynomial, sliding_window_search
    from lane_detector_py.visualization import (
        draw_lane_overlay,
        render_sliding_window_debug,
    )
except ImportError:  # pragma: no cover
    from .binary import create_lane_mask
    from .birdseye import compute_homography, warp_to_top_view
    from .sliding_window import fit_polynomial, sliding_window_search
    from .visualization import draw_lane_overlay, render_sliding_window_debug


# Lane Detector Node Class
class LaneDetectorNode(Node):
    def __init__(self): # 노드 생성시 한번만 실행
        super().__init__('lane_detector')

        # 파라미터서버에 파라미터 등록 및 기본값 설정
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('publish_overlay_topic', '/lane/overlay')
        self.declare_parameter('publish_offset_topic', '/lane/center_offset')
        self.declare_parameter('use_birdeye', True)
        self.declare_parameter('enable_visualization', True) # 디버깅용 시각화 여부 파라미터 , 기본값 False
        self.crop_size = (860, 480)
        self.last_frame_shape = None
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.lane_width_px = None

        # 버드아이용 호모그래피(예시 좌표: 해상도 640x480 전제)
        # 실제 카메라 및 트랙에 맞게 보정 필요 
        # src_points : 원본 카메라 이미지에서 변환에 사용할 4개의 점 
        # dst_points : 버드아이뷰에서 대응되는 4개의 점[x0, y0, x1, y1, x2, y2, x3, y3
        self.declare_parameter('src_points', [55.0, 480.0, 235.0, 300.0, 477.0, 300.0, 633.0, 480.0]) # 06.11 수정된 기본값, 카메라 위치 바꿔서 다시 체크 
        self.declare_parameter('dst_points', [0.0,  480.0, 0.0,   0.0, 640.0, 0.0, 640.0, 480.0])

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.subscribe_compressed = image_topic.endswith('/compressed')
        # print(self.subscribe_compressed) # check
        # overlay_topic = self.get_parameter('publish_overlay_topic').get_parameter_value().string_value
        offset_topic = self.get_parameter('publish_offset_topic').get_parameter_value().string_value
        self.use_birdeye = self.get_parameter('use_birdeye').get_parameter_value().bool_value
        self.visualize = self.get_parameter('enable_visualization').get_parameter_value().bool_value

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

        if self.subscribe_compressed: # compressed image --> 실제 실행되는 부분
            self.sub = self.create_subscription(CompressedImage, image_topic, self.image_cb_compressed, qos)
        else: # raw image
            self.sub = self.create_subscription(Image, image_topic, self.image_cb_raw, qos)

        # 호모그래피 미리 계산(버드아이뷰 변환에 사용할 행렬)--> 프레임 계산을 줄이기 위해 한번만 실행 
        self.H, self.Hinv = self._compute_homography()

        # 디버깅용 이미지 표시 창과 마우스 콜백 설정
        self.window_name = 'lane_detector_input'
        self.control_window = 'homography_controls'
        self.birdeye_window = 'wrapped_img'
        self.overlay_window = 'lane_overlay'

        # trackbar param
        self.homography_ui_ready = False # 트랙바 한번만 생성
        self._trackbar_lock = False


        sub_type = 'CompressedImage' if self.subscribe_compressed else 'Image'
        self.get_logger().info(f'LaneDetector subscribing: {image_topic} ({sub_type})')
        # self.get_logger().info(f'Publishing overlay: {overlay_topic}, center_offset: {offset_topic}')


    #####################################  homography  ####################################################################

    def _compute_homography(self):
        return compute_homography(self.src_pts, self.dst_pts, self.use_birdeye)
    
    def _ensure_homography_ui(self):
        if not self.use_birdeye or self.homography_ui_ready or self.last_frame_shape is None:
            return

        ref_w, ref_h = self.last_frame_shape
        ref_w = max(1, ref_w)
        ref_h = max(1, ref_h)

        cv2.namedWindow(self.control_window, cv2.WINDOW_AUTOSIZE)

        for idx in range(4):
            self._create_homography_trackbar('src', idx, 0, ref_w)
            self._create_homography_trackbar('src', idx, 1, ref_h)
            self._create_homography_trackbar('dst', idx, 0, ref_w)
            self._create_homography_trackbar('dst', idx, 1, ref_h)

        self.homography_ui_ready = True

    def _create_homography_trackbar(self, point_type: str, idx: int, axis: int, max_val: int):
        arr = self.src_pts if point_type == 'src' else self.dst_pts
        track_name = f'{point_type}_{"x" if axis == 0 else "y"}{idx}'
        max_slider = max(1, max_val - 1)
        initial = int(np.clip(arr[idx, axis], 0, max_slider))
        arr[idx, axis] = float(initial)
        cv2.createTrackbar(
            track_name,
            self.control_window,
            initial,
            max_slider,
            partial(self._on_homography_trackbar, point_type, idx, axis, track_name)
        )

    def _on_homography_trackbar(self, point_type: str, idx: int, axis: int, track_name: str, value: int):
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
                cv2.setTrackbarPos(track_name, self.control_window, int(clipped))
            finally:
                self._trackbar_lock = False

        arr[idx, axis] = clipped
        self.H, self.Hinv = self._compute_homography()

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

        # 1) 중앙 크롭: 640x480 기준으로 중앙 영역만 사용
        crop_w, crop_h = self.crop_size
        cur_h, cur_w, _ = bgr.shape
        if cur_w >= crop_w and cur_h >= crop_h:
            x0 = (cur_w - crop_w) // 2
            y0 = (cur_h - crop_h) // 2
            bgr = bgr[y0:y0 + crop_h, x0:x0 + crop_w]
        else:
            self.get_logger().warn(
                f'Incoming image smaller than crop size ({cur_w}x{cur_h} < {crop_w}x{crop_h}); skipping center crop.')

        # # 2) 상단 1/3 제거하여 하단 2/3만 사용
        # cur_h, cur_w, _ = bgr.shape
        # top_cut = cur_h // 3
        # if top_cut > 0:
        #     bgr = bgr[top_cut:, :]

        # 3) 가로가 넓을 경우 다시 중앙 정렬
        cur_h, cur_w, _ = bgr.shape
        if cur_w > crop_w:
            x0 = (cur_w - crop_w) // 2
            bgr = bgr[:, x0:x0 + crop_w]
        elif cur_w < crop_w:
            self.get_logger().warn(
                f'Incoming image narrower than crop width ({cur_w} < {crop_w}); skipping horizontal crop.')

        h, w, _ = bgr.shape
        self.last_frame_shape = (w, h)
        # self._ensure_homography_ui()

        # 4) 전처리 → 이진 마스크
        mask = create_lane_mask(bgr)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        

        # 5) 버드아이뷰 변환 (이진 마스크 기준)
        top = warp_to_top_view(mask, self.H) if self.H is not None else mask
        if top.ndim == 3:
            top = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        
        # 6) 차선 검출 부분 
        # 슬라이딩 윈도우 → 피팅
        (lx, ly), (rx, ry), window_records = sliding_window_search(top)
        left_fit_raw = fit_polynomial((lx, ly))
        right_fit_raw = fit_polynomial((rx, ry))
        left_detected = left_fit_raw is not None
        right_detected = right_fit_raw is not None

        alpha = getattr(self, 'fit_smoothing_alpha', 0.2)

        def _smooth(prev, new):
            if prev is None:
                return new.copy()
            return (1.0 - alpha) * prev + alpha * new

        left_fit = None
        if left_detected:
            raw = np.array(left_fit_raw, dtype=float)
            left_fit = _smooth(self.prev_left_fit, raw)
            self.prev_left_fit = left_fit.copy()
        elif self.prev_left_fit is not None:
            left_fit = self.prev_left_fit.copy()

        right_fit = None
        if right_detected:
            raw = np.array(right_fit_raw, dtype=float)
            right_fit = _smooth(self.prev_right_fit, raw)
            self.prev_right_fit = right_fit.copy()
        elif self.prev_right_fit is not None:
            right_fit = self.prev_right_fit.copy()

        if left_detected and right_detected:
            y_eval_width = h - 1
            x_left_raw = (left_fit_raw[0]*y_eval_width*y_eval_width +
                          left_fit_raw[1]*y_eval_width + left_fit_raw[2])
            x_right_raw = (right_fit_raw[0]*y_eval_width*y_eval_width +
                           right_fit_raw[1]*y_eval_width + right_fit_raw[2])
            width_px = float(x_right_raw - x_left_raw)
            if width_px > 0.0:
                if self.lane_width_px is None:
                    self.lane_width_px = width_px
                else:
                    beta = getattr(self, 'lane_width_alpha', 0.2)
                    self.lane_width_px = (1.0 - beta) * self.lane_width_px + beta * width_px
        else:
            if self.lane_width_px is not None:
                if left_fit is not None and right_fit is None:
                    right_fit = left_fit.copy()
                    right_fit[2] += self.lane_width_px
                elif right_fit is not None and left_fit is None:
                    left_fit = right_fit.copy()
                    left_fit[2] -= self.lane_width_px

        # 차 폭/센터 오프셋 계산 (픽셀 기준 → 미터 변환은 사용자 설정)
        center_offset_px = 0.0
        if left_fit is not None and right_fit is not None:
            y_eval = h - 1
            x_left = left_fit[0]*y_eval*y_eval + left_fit[1]*y_eval + left_fit[2]
            x_right = right_fit[0]*y_eval*y_eval + right_fit[1]*y_eval + right_fit[2]
            lane_center = (x_left + x_right) / 2.0
            img_center = w / 2.0
            center_offset_px = float(img_center - lane_center)

        if viz_enabled:
            cv2.imshow(self.window_name, bgr)
            debug_view = render_sliding_window_debug(top, window_records, (lx, ly), (rx, ry))
            cv2.imshow(self.birdeye_window, debug_view)

            fill_overlay = left_detected and right_detected
            overlay = draw_lane_overlay(bgr, top, self.Hinv, left_fit, right_fit, fill=fill_overlay)
            cv2.imshow(self.overlay_window, overlay)

            # if you want to publish overlay image, uncomment below
            # self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8'))
            cv2.waitKey(1)

        # 퍼블리시
        self.pub_offset.publish(Float32(data=center_offset_px))


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
