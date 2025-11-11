#ifndef CAMERA_H
#define CAMERA_H

#include "Global/global.h"
#include <sensor_msgs/CompressedImage.h>

void CameraProcessing(RAW_CAMERA_DATA_t *pst_RawData, CAMERA_DATA_t *pst_CameraData);
void LoadParam(CAMERA_DATA_t *pst_CameraData);
void LoadMappingParam(CAMERA_DATA_t *pst_CameraData);

// Lane Detection
void FindTop5MaxIndices(const int32_t* ps32_Histogram, int32_t s32_MidPoint, int32_t ars32_TopIndices[5], bool& b_NoLane);
int32_t FindClosestToMidPoint(const int32_t points[5], int32_t s32_MidPoint);
void FindLaneStartPositions(const cv::Mat& st_Edge, int32_t& s32_WindowCentorLeft, int32_t& s32_WindowCentorRight, bool& b_NoLaneLeft, bool& b_NoLaneRight);
void SlidingWindow(const cv::Mat& st_EdgeImage, const cv::Mat& st_NonZeroPosition, CAMERA_LANEINFO_t& st_LaneInfoLeft ,CAMERA_LANEINFO_t& st_LaneInfoRight ,int32_t& s32_LeftWindowCentor, int32_t& s32_RightWindowCentor, cv::Mat& st_ResultImage);
LANE_COEFFICIENT_t FitModel(const Point& st_Point1, const Point& st_Point2, bool& b_Flag);
void CalculateLaneCoefficient(CAMERA_LANEINFO_t& st_LaneInfo, int32_t s32_Iteration, int64_t s64_Threshold);
void InitializeKalmanObject(LANE_KALMAN_t& st_KalmanObject);
KALMAN_STATE_t CalculateKalmanState(const LANE_COEFFICIENT_t& st_LaneCoef, float32_t& f32_Distance, float32_t& f32_Angle);
void UpdateObservation(LANE_KALMAN_t& st_KalmanObject, const KALMAN_STATE_t st_KalmanState);
void SetInitialX(LANE_KALMAN_t& st_KalmanObject);
void PredictState(LANE_KALMAN_t& st_KalmanObject);
void UpdateMeasurement(LANE_KALMAN_t& st_KalmanObject);
void CheckSameKalmanObject(LANE_KALMAN_t& st_KalmanObject, KALMAN_STATE_t st_KalmanStateLeft);
void DeleteKalmanObject(CAMERA_DATA_t &pst_CameraData, int32_t& s32_KalmanObjectNum, int32_t s32_I);
void DrawDrivingLane(cv::Mat& st_ResultImage, const LANE_COEFFICIENT_t st_LaneCoef, cv::Scalar st_Color);
void MakeKalmanStateBasedLaneCoef(const LANE_KALMAN_t& st_KalmanObject, LANE_COEFFICIENT_t& st_LaneCoefficient);

#endif