#!/usr/bin/env python3
import math
import time
from functools import partial
import inspect
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge


def _hist_peaks(binary_topdown, margin_bottom=0.4):
    """하단부 히스토그램 피크(좌/우 베이스 위치) 찾기"""
    h, w = binary_topdown.shape
    y0 = int(h * (1.0 - margin_bottom))
    region = binary_topdown[y0:, :]
    histogram = np.sum(region // 255, axis=0)

    mid = w // 2
    leftx_base = np.argmax(histogram[:mid]) if histogram[:mid].any() else None
    rightx_base = np.argmax(histogram[mid:]) + mid if histogram[mid:].any() else None
    return leftx_base, rightx_base


def _sliding_window(binary_topdown, nwindows=9, window_width=80, minpix=50):
    """
    슬라이딩 윈도우로 좌/우 차선 픽셀 인덱스 수집.

    tunable parameters:
        nwindows     : 전체 윈도우 개수 (세로 방향 분할 수)
        window_width : 각 윈도우의 가로 폭(픽셀 단위)
        minpix       : 윈도우 안에 최소 몇 개의 유효 픽셀이 있어야
                       윈도우 중앙을 갱신할지 결정하는 기준
    """
    h, w = binary_topdown.shape
    nonzero = binary_topdown.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_base, rightx_base = _hist_peaks(binary_topdown)
    leftx_current, rightx_current = leftx_base, rightx_base

    window_height = h // nwindows
    left_lane_inds = []
    right_lane_inds = []
    window_records = {'left': [], 'right': []}

    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height

        def _gather(x_center):
            if x_center is None:
                return None, None, None, None, np.array([], dtype=int)
            win_x_low = max(0, x_center - window_width // 2)
            win_x_high = min(w, x_center + window_width // 2)
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            if len(good_inds) > minpix:
                x_center = int(np.mean(nonzerox[good_inds]))
            return win_x_low, win_x_high, win_y_low, win_y_high, good_inds, x_center

        # left
        if leftx_current is not None:
            lx0, lx1, ly0, ly1, good, leftx_current = _gather(leftx_current)
            if good.size:
                left_lane_inds.append(good)
            if lx0 is not None and lx1 is not None:
                window_records['left'].append((lx0, lx1, ly0, ly1))

        # right
        if rightx_current is not None:
            rx0, rx1, ry0, ry1, good, rightx_current = _gather(rightx_current)
            if good.size:
                right_lane_inds.append(good)
            if rx0 is not None and rx1 is not None:
                window_records['right'].append((rx0, rx1, ry0, ry1))

    left_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([], dtype=int)
    right_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([], dtype=int)

    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]
    return (leftx, lefty), (rightx, righty), window_records


def _fit_poly(points):
    """y기준 x(y) = ay^2 + by + c 2차 피팅"""
    x, y = points
    if len(x) < 50:
        return None
    return np.polyfit(y, x, 2)  # [a, b, c]


def _draw_overlay(orig_bgr, binary_topdown, Hinv, left_fit, right_fit):
    h, w = binary_topdown.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    ploty = np.arange(h)
    if left_fit is not None:
        leftx = (left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]).astype(np.int32)
        pts_left = np.stack([leftx, ploty], axis=1)
    else:
        pts_left = None

    if right_fit is not None:
        rightx = (right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]).astype(np.int32)
        pts_right = np.stack([rightx, ploty], axis=1)
    else:
        pts_right = None

    if pts_left is not None and pts_right is not None:
        pts = np.vstack([pts_left, pts_right[::-1]])
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
    elif pts_left is not None:
        cv2.polylines(overlay, [pts_left], False, (255, 0, 0), 5)
    elif pts_right is not None:
        cv2.polylines(overlay, [pts_right], False, (0, 0, 255), 5)

    overlay_warped = cv2.warpPerspective(overlay, Hinv, (orig_bgr.shape[1], orig_bgr.shape[0]))
    out = cv2.addWeighted(orig_bgr, 1.0, overlay_warped, 0.4, 0)
    return out


class LaneDetectorNode(Node):
    def __init__(self): # 노드 생성시 한번만 실행
        super().__init__('lane_detector')

        # 파라미터서버에 파라미터 등록 및 기본값 설정
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('publish_overlay_topic', '/lane/overlay')
        self.declare_parameter('publish_offset_topic', '/lane/center_offset')
        self.declare_parameter('use_birdeye', True)
        self.crop_size = (640, 480)
        self.last_frame_shape = None
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.lane_width_px = None

        # 버드아이용 호모그래피(예시 좌표: 해상도 640x480 전제)
        # 실제 카메라 및 트랙에 맞게 보정 필요 
        # src_points : 원본 카메라 이미지에서 변환에 사용할 4개의 점 
        # dst_points : 버드아이뷰에서 대응되는 4개의 점[x0, y0, x1, y1, x2, y2, x3, y3
        self.declare_parameter('src_points', [54.0, 480.0, 219.0, 300.0, 456.0, 300.0, 633.0, 480.0])
        self.declare_parameter('dst_points', [0.0,  480.0, 0.0,   0.0, 640.0, 0.0, 640.0, 480.0])

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.subscribe_compressed = image_topic.endswith('/compressed')
        # print(self.subscribe_compressed) # check
        overlay_topic = self.get_parameter('publish_overlay_topic').get_parameter_value().string_value
        offset_topic = self.get_parameter('publish_offset_topic').get_parameter_value().string_value
        self.use_birdeye = self.get_parameter('use_birdeye').get_parameter_value().bool_value

        self.src_pts = np.array(self.get_parameter('src_points').value, dtype=np.float32).reshape(4, 2)
        self.dst_pts = np.array(self.get_parameter('dst_points').value, dtype=np.float32).reshape(4, 2)

        # QoS: 센서데이터는 BestEffort/Depth=1이 지연/버퍼에 유리
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge() # change ros imgmsg <-> cv2

        self.pub_overlay = self.create_publisher(Image, overlay_topic, qos)
        self.pub_offset = self.create_publisher(Float32, offset_topic, 10)

        if self.subscribe_compressed: # compressed image
            self.sub = self.create_subscription(CompressedImage, image_topic, self.image_cb_compressed, qos)
        else: # raw image
            self.sub = self.create_subscription(Image, image_topic, self.image_cb_raw, qos)

        # 호모그래피 미리 계산(버드아이뷰 변환에 사용할 행렬)--> 프레임 계산을 줄이기 위해 한번만 실행 
        self.H, self.Hinv = self._compute_homography()

        # 디버깅용 이미지 표시 창과 마우스 콜백 설정
        self.window_name = 'lane_detector_input'
        #cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        #cv2.setMouseCallback(self.window_name, self._on_mouse)
        self.control_window = 'homography_controls'
        self.birdeye_window = 'wrapped_img'
        self.overlay_window = 'lane_overlay'
        self.homography_ui_ready = False
        self._trackbar_lock = False

        #cv2.namedWindow(self.overlay_window, cv2.WINDOW_NORMAL)
        # if self.use_birdeye:
        #     cv2.namedWindow(self.birdeye_window, cv2.WINDOW_NORMAL)

        sub_type = 'CompressedImage' if self.subscribe_compressed else 'Image'
        self.get_logger().info(f'LaneDetector subscribing: {image_topic} ({sub_type})')
        self.get_logger().info(f'Publishing overlay: {overlay_topic}, center_offset: {offset_topic}')

    def _compute_homography(self):
        if not self.use_birdeye:
            return np.eye(3), np.eye(3)

        src = self.src_pts.astype(np.float32)
        dst = self.dst_pts.astype(np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        Hinv = cv2.getPerspectiveTransform(dst, src)
        return H, Hinv # 행렬 및 역행렬 

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

    def _render_sliding_window_debug(self, binary_topdown, windows, left_points, right_points):
        if binary_topdown.ndim == 2:
            vis = cv2.cvtColor(binary_topdown, cv2.COLOR_GRAY2BGR)
        else:
            vis = binary_topdown.copy()

        for x0, x1, y0, y1 in windows.get('left', []):
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 1)
        for x0, x1, y0, y1 in windows.get('right', []):
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)

        lx, ly = left_points
        if lx.size:
            lx_clip = np.clip(lx.astype(int), 0, vis.shape[1]-1)
            ly_clip = np.clip(ly.astype(int), 0, vis.shape[0]-1)
            vis[ly_clip, lx_clip] = (0, 255, 255)
        rx, ry = right_points
        if rx.size:
            rx_clip = np.clip(rx.astype(int), 0, vis.shape[1]-1)
            ry_clip = np.clip(ry.astype(int), 0, vis.shape[0]-1)
            vis[ry_clip, rx_clip] = (0, 255, 255)

        return vis

    def _binarize(self, bgr):
        """HSV + Sobel 혼합 간단 임계처리"""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 조명 보정(가벼운 CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v2 = clahe.apply(v)
        hsv2 = cv2.merge([h, s, v2])
        bgr2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

        gray = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러로 노이즈 완화 후 엣지/임계값 계산
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        sobelx = cv2.Sobel(gray_blur, cv2.CV_16S, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)

        # 색 기반(흰/노란) + 엣지 기반 + 흰색 구간 강화(HSV)
        _, binary_gray = cv2.threshold(gray_blur, 210, 255, cv2.THRESH_BINARY)  # 밝은 선
        _, sat_mask = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY)         # 채도 약간
        edges = cv2.Canny(gray_blur, 80, 160)

        # 흰색(고밝기 + 낮은 채도) 픽셀 강조
        white_mask = cv2.inRange(hsv2, np.array([0, 0, 180]), np.array([180, 80, 255]))

        combo = np.zeros_like(gray_blur)
        combo[(binary_gray == 255) | (sat_mask == 255) | (edges == 255) | (white_mask == 255)] = 255
        return combo

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


    # image processing main function
    def _process_frame(self, bgr: np.ndarray):
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

        cv2.imshow(self.window_name, bgr) # input image

        h, w, _ = bgr.shape
        self.last_frame_shape = (w, h)
        # self._ensure_homography_ui()

        # 4) 전처리 → 이진 마스크
        mask = self._binarize(bgr)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cv2.imshow('binary mask',mask)

        # 5) 버드아이뷰 변환 (이진 마스크 기준)
        top = cv2.warpPerspective(mask, self.H, (w, h)) if self.H is not None else mask
        if top.ndim == 3:
            top = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)

        # 슬라이딩 윈도우 → 피팅
        (lx, ly), (rx, ry), window_records = _sliding_window(top)
        left_fit_raw = _fit_poly((lx, ly))
        right_fit_raw = _fit_poly((rx, ry))
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

        debug_view = self._render_sliding_window_debug(top, window_records, (lx, ly), (rx, ry))
        # cv2.imshow(self.birdeye_window, debug_view)

        # 오버레이 이미지
        fill_overlay = left_detected and right_detected
        draw_kwargs = {}
        try:
            if 'fill' in inspect.signature(_draw_overlay).parameters:
                draw_kwargs['fill'] = fill_overlay
        except (ValueError, TypeError):
            pass
        overlay = _draw_overlay(bgr, top, self.Hinv, left_fit, right_fit, **draw_kwargs)
        
        
        cv2.imshow(self.overlay_window, overlay)
         

        # 퍼블리시
        # self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8'))
        self.pub_offset.publish(Float32(data=center_offset_px))

        cv2.waitKey(1)


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

