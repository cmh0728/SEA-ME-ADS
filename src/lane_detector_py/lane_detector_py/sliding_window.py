from typing import Dict, List, Tuple

import numpy as np

WindowRecords = Dict[str, List[Tuple[int, int, int, int]]]
LanePoints = Tuple[np.ndarray, np.ndarray]

# histogram 피크찾기
def _hist_peaks(binary_topdown: np.ndarray, margin_bottom: float = 0.4):
    """Find histogram peaks (base positions) near the bottom of the binary image."""
    # 바닥 영역 히스토그램을 통해 왼쪽/오른쪽 차선의 시작 x좌표를 추정
    h, w = binary_topdown.shape
    y0 = int(h * (1.0 - margin_bottom))
    region = binary_topdown[y0:, :]
    histogram = np.sum(region // 255, axis=0)

    mid = w // 2
    leftx_base = np.argmax(histogram[:mid]) if histogram[:mid].any() else None
    rightx_base = np.argmax(histogram[mid:]) + mid if histogram[mid:].any() else None
    return leftx_base, rightx_base


def sliding_window_search(
    binary_topdown: np.ndarray,
    nwindows: int = 9,
    window_width: int = 100,  # 윈도우 가로폭 픽셀값
    minpix: int = 80,  # 윈도우 내 유효 픽셀 수 최솟값
    ) -> Tuple[LanePoints, LanePoints, WindowRecords]:
    """Collect left/right lane pixels with a histogram-guided sliding-window search."""
    # 바이너리 버드뷰 이미지 좌표를 준비
    h, w = binary_topdown.shape
    nonzero = binary_topdown.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 히스토그램으로 첫 윈도우의 중앙 좌표를 잡음
    leftx_base, rightx_base = _hist_peaks(binary_topdown)
    print(f"Left base: {leftx_base}, Right base: {rightx_base}")
    leftx_current, rightx_current = leftx_base, rightx_base

    # 슬라이딩 윈도우 높이 및 결과 저장 리스트 초기화
    window_height = h // nwindows if nwindows else h
    left_lane_inds = []
    right_lane_inds = []
    window_records: WindowRecords = {"left": [], "right": []}

    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height

        def _gather(x_center: int):
            # 현재 윈도우 중심을 기준으로 유효한 픽셀을 모으고 중심을 새로 계산
            if x_center is None:
                return None, None, None, None, np.array([], dtype=int), None
            win_x_low = max(0, x_center - window_width // 2)
            win_x_high = min(w, x_center + window_width // 2)
            good_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_x_low)
                & (nonzerox < win_x_high)
            ).nonzero()[0]
            new_center = x_center
            if len(good_inds) > minpix:
                new_center = int(np.mean(nonzerox[good_inds]))
            return win_x_low, win_x_high, win_y_low, win_y_high, good_inds, new_center

        if leftx_current is not None:
            # 왼쪽 차선 윈도우 추적
            lx0, lx1, ly0, ly1, good, leftx_current = _gather(leftx_current)
            if good.size:
                left_lane_inds.append(good)
            if lx0 is not None and lx1 is not None:
                window_records["left"].append((lx0, lx1, ly0, ly1))

        if rightx_current is not None:
            # 오른쪽 차선 윈도우 추적
            rx0, rx1, ry0, ry1, good, rightx_current = _gather(rightx_current)
            if good.size:
                right_lane_inds.append(good)
            if rx0 is not None and rx1 is not None:
                window_records["right"].append((rx0, rx1, ry0, ry1))

    left_inds = (
        np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([], dtype=int)
    )
    right_inds = (
        np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([], dtype=int)
    )

    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]
    return (leftx, lefty), (rightx, righty), window_records


def fit_polynomial(points: LanePoints):
    """Fit a second-order polynomial x(y) = ay^2 + by + c if enough points exist."""
    # 충분한 포인트가 있을 때만 2차 곡선을 피팅
    x, y = points
    if len(x) < 50:
        return None
    return np.polyfit(y, x, 2)
