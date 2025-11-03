#!/usr/bin/env python3
import math
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
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
    """슬라이딩 윈도우로 좌/우 차선 픽셀 인덱스 수집"""
    h, w = binary_topdown.shape
    nonzero = binary_topdown.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_base, rightx_base = _hist_peaks(binary_topdown)
    leftx_current, rightx_current = leftx_base, rightx_base

    window_height = h // nwindows
    left_lane_inds = []
    right_lane_inds = []

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

        # right
        if rightx_current is not None:
            rx0, rx1, ry0, ry1, good, rightx_current = _gather(rightx_current)
            if good.size:
                right_lane_inds.append(good)

    left_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([], dtype=int)
    right_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([], dtype=int)

    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]
    return (leftx, lefty), (rightx, righty)


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
    def __init__(self):
        super().__init__('lane_detector')

        # 파라미터
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('publish_overlay_topic', '/lane/overlay')
        self.declare_parameter('publish_offset_topic', '/lane/center_offset')
        self.declare_parameter('use_birdeye', True)

        # 버드아이용 호모그래피(예시 좌표: 해상도 640x480 전제)
        # 실제 트랙 기준으로 꼭 보정하세요!
        self.declare_parameter('src_points', [ # 이미지 상 4점 (좌상, 우상, 우하, 좌하)
            [200.0, 300.0], [440.0, 300.0], [620.0, 470.0], [20.0, 470.0]
        ])
        self.declare_parameter('dst_points', [ # 탑뷰 상 4점(직사각형)
            [100.0, 0.0], [540.0, 0.0], [540.0, 480.0], [100.0, 480.0]
        ])

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        overlay_topic = self.get_parameter('publish_overlay_topic').get_parameter_value().string_value
        offset_topic = self.get_parameter('publish_offset_topic').get_parameter_value().string_value

        # QoS: 센서데이터는 BestEffort/Depth=1이 지연/버퍼에 유리
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()

        self.pub_overlay = self.create_publisher(Image, overlay_topic, qos)
        self.pub_offset = self.create_publisher(Float32, offset_topic, 10)

        self.sub = self.create_subscription(Image, image_topic, self.image_cb, qos)

        # 호모그래피 미리 계산
        self.H, self.Hinv = self._compute_homography()

        self.get_logger().info(f'LaneDetector subscribing: {image_topic}')
        self.get_logger().info(f'Publishing overlay: {overlay_topic}, center_offset: {offset_topic}')

    def _compute_homography(self):
        use_be = self.get_parameter('use_birdeye').get_parameter_value().bool_value
        if not use_be:
            return np.eye(3), np.eye(3)
        src = np.array(self.get_parameter('src_points').value, dtype=np.float32)
        dst = np.array(self.get_parameter('dst_points').value, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        Hinv = cv2.getPerspectiveTransform(dst, src)
        return H, Hinv

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
        sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)

        # 색 기반(흰/노란) + 엣지 기반을 OR
        _, binary_gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # 밝은 선
        _, sat_mask = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY)         # 채도 약간
        edges = cv2.Canny(gray, 80, 160)

        combo = np.zeros_like(gray)
        combo[(binary_gray == 255) | (sat_mask == 255) | (edges == 255)] = 255
        return combo

    def image_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        h, w, _ = bgr.shape

        # 전처리 → 이진 마스크
        mask = self._binarize(bgr)

        # 탑뷰(옵션)
        top = cv2.warpPerspective(mask, self.H, (w, h)) if self.H is not None else mask

        # 슬라이딩 윈도우 → 피팅
        (lx, ly), (rx, ry) = _sliding_window(top)
        left_fit = _fit_poly((lx, ly))
        right_fit = _fit_poly((rx, ry))

        # 차 폭/센터 오프셋 계산 (픽셀 기준 → 미터 변환은 사용자 설정)
        center_offset_px = 0.0
        if left_fit is not None and right_fit is not None:
            y_eval = h - 1
            x_left = left_fit[0]*y_eval*y_eval + left_fit[1]*y_eval + left_fit[2]
            x_right = right_fit[0]*y_eval*y_eval + right_fit[1]*y_eval + right_fit[2]
            lane_center = (x_left + x_right) / 2.0
            img_center = w / 2.0
            center_offset_px = float(img_center - lane_center)  # +면 우측, -면 좌측으로 해석 가능

        # 오버레이 이미지
        overlay = _draw_overlay(bgr, top, self.Hinv, left_fit, right_fit)

        # 퍼블리시
        self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8'))
        self.pub_offset.publish(Float32(data=center_offset_px))


def main():
    rclpy.init()
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
