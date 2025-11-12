#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <vector>
#include <cstdlib>
#include <tuple>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <eigen3/Eigen/Dense> // Eigen Library : 선형대수 템플릿 라이브러리 
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <Eigen/Eigen>
#include <Eigen/Dense>

#include "opencv2/opencv.hpp"       

// 제어 최적화 라이브러리 casadi
// #include "casadi/casadi.hpp" 
// #include "casadi/core/sparsity_interface.hpp"

// #include "Integration/GNSSInfo.h"
// #include "Integration/object_msg_arr.h"

#include <yaml-cpp/yaml.h>
// #include <mlpack.hpp> // 머신러닝 라이브러리 통합 헤더 


#define NONE -1e9

typedef unsigned char uint8_t;
typedef unsigned short uint16_t; 
typedef unsigned int uint32_t;
typedef short int16_t;
typedef float float32_t;
typedef double float64_t;

using namespace std;
using namespace Eigen;
using namespace chrono;
using namespace cv;


const int32_t c_PARSING_UDP = 0;
const int32_t c_PARSING_ROS = 1;
const int32_t c_PARSING_SERIAL = 2;
const int32_t c_PARSING_CAN = 3;

const int32_t c_LIDAR_BUFFER_SIZE = 2000000;
const int32_t c_CAMERA_BUFFER_SIZE = 2000000;
const int32_t c_GPSINS_BUFFER_SIZE = 100;
const int32_t c_IMU_BUFFER_SIZE = 100;
const int32_t c_CAN_BUFFER_SIZE = 100;
const int32_t c_MAX_FRAME_SIZE = 5000;





const int32_t c_VIEWER_NULL = -1;
const int32_t c_VIEWER_MENU = 0;
const int32_t c_VIEWER_LOCAL = 1;
const int32_t c_VIEWER_GLOBAL = 2;
const int32_t c_VIEWER_FRAME = 3;
const int32_t c_VIEWER_CAMERA = 4;

const int32_t c_VIEWER_SIMUL_MODE = 0;
const int32_t c_VIEWER_REAL_MODE = 1;

const int32_t c_VIEWER_BUTTON_NUM_SIMUL = 5;
const int32_t c_VIEWER_BUTTON_NUM_REAL = 3;

const int32_t c_VIEWER_BUTTON_NULL = -1;

const int32_t c_VIEWER_BUTTON_SIMUL_MODE = 0;
const int32_t c_VIEWER_BUTTON_SIMUL_SENSOR_STATE = 1;
const int32_t c_VIEWER_BUTTON_SIMUL_EGO_STATE = 2;
const int32_t c_VIEWER_BUTTON_SIMUL_LOGGING = 3;
const int32_t c_VIEWER_BUTTON_SIMUL_PLAY = 4;


const int32_t c_VIEWER_BUTTON_REAL_MODE = 0;
const int32_t c_VIEWER_BUTTON_REAL_LOGGING = 1;
const int32_t c_VIEWER_BUTTON_REAL_AUTOMODE_ON = 2;

const int32_t c_VIEWER_BUTTON_EXIT = 6;

const int32_t c_STATE_MODE_LOGGING_ON = 0;
const int32_t c_STATE_MODE_LOGGING_OFF = 1;
const int32_t c_STATE_MODE_PLAY = 2;
const int32_t c_STATE_MODE_STOP = 3; 
const int32_t c_STATE_MODE_NULL = 4;

const int32_t c_STATE_1XZOOMFACTOR = 5;
const int32_t c_STATE_2XZOOMFACTOR = 6;
const int32_t c_STATE_3XZOOMFACTOR = 7;

const int32_t c_TOTAL_POINT_NUM = 240000;

const uint8_t c_POINT_INVALID = 0;
const uint8_t c_POINT_VALID = 255;
const uint8_t c_POINT_GROUND = 128;
const uint8_t c_CLUSTER_CAR = 255;
const uint8_t c_CLUSTER_NONCAR = 0;

const float32_t c_LEAF_SIZE = 1.f;

const float32_t c_DISTANCE_LEAF_SIZE = 1.f;
const float32_t c_AZIMUTH_LEAF_SIZE = 1.f;
const float32_t c_ELEVATION_LEAF_SIZE = 2.5f;

const int32_t c_VOXEL_DISTANCE_MIN = 0;
const int32_t c_VOXEL_DISTANCE_MAX = 150;
const int32_t c_VOXEL_AZIMUTH_MIN = 0;
const int32_t c_VOXEL_AZIMUTH_MAX = 360;
const int32_t c_VOXEL_ELEVATION_MIN = -15;
const int32_t c_VOXEL_ELEVATION_MAX = 5;

const uint32_t c_GRID_DISTANCE_SIZE = (c_VOXEL_DISTANCE_MAX - c_VOXEL_DISTANCE_MIN) / c_DISTANCE_LEAF_SIZE;
const uint32_t c_GRID_AZIMUTH_SIZE = (c_VOXEL_AZIMUTH_MAX - c_VOXEL_AZIMUTH_MIN) / c_AZIMUTH_LEAF_SIZE;
const uint32_t c_GRID_ELEVATION_SIZE = (c_VOXEL_ELEVATION_MAX - c_VOXEL_ELEVATION_MIN) / c_ELEVATION_LEAF_SIZE;


const int32_t c_UNCLUSTERED = -1;
const int32_t c_TOTAL_CLUSTER_NUM = 500;
const int32_t c_CLUSTER_POINT_NUM = 100000;

const uint8_t c_OBJECT_STATIC = 0;
const uint8_t c_OBJECT_DYNAMIC = 1;
const uint8_t c_TOTAL_TRACKING_NUM = 32;

extern float64_t c_ORIGIN_LATITUDE_DEG;
extern float64_t c_ORIGIN_LONGITUDE_DEG;
extern float64_t c_ORIGIN_LATITUDE_RAD;
extern float64_t c_ORIGIN_LONGITUDE_RAD;
extern float64_t c_ORIGIN_ALTITUDE;
extern float64_t c_ORIGIN_REFERENCE_X;
extern float64_t c_ORIGIN_REFERENCE_Y;
extern float64_t c_ORIGIN_REFERENCE_Z;


const float64_t c_LLA2ENU_A = 6378137.0;
const float64_t c_LLA2ENU_FLAT_RATIO = 1 / 298.257223563;
const float64_t c_LLA2ENU_N_2 = (2 * c_LLA2ENU_FLAT_RATIO) - pow(c_LLA2ENU_FLAT_RATIO, 2);


const int32_t c_PLANNING_MAX_PATH_NUM = 10000;
const int32_t c_PLANNING_MAX_SPLINE_NUM = 100000;
const int32_t c_PLANNING_MAX_FRENET_NUM = 1000;
const int32_t c_PLANNING_MAX_FRENET_PATH_NUM = 1000;


const uint8_t c_CONTROL_FLAG_NONE = 0;
const uint8_t c_CONTROL_FLAG_ACC = 1;
const uint8_t c_CONTROL_FLAG_AEB = 2;
const uint8_t c_CONTROL_FLAG_OVERTAKING = 3;

const int32_t c_CONTROL_HORIZON = 10;



typedef struct _RAW_LIDAR_DATA
{
  uint64_t u64_Timestamp;
  int32_t s32_Num;
  char arc_Buffer[c_LIDAR_BUFFER_SIZE];
  int32_t s32_LiDARHeader;
} RAW_LIDAR_DATA_t;

typedef struct _RAW_CAMERA_DATA
{
  uint64_t u64_Timestamp;
  int32_t s32_Num;
  char arc_Buffer[c_CAMERA_BUFFER_SIZE];
  int32_t s32_CameraHeader;
} RAW_CAMERA_DATA_t;

typedef struct _RAW_GPSINS_DATA
{
  uint64_t u64_Timestamp;
  int32_t s32_Num;
  char arc_Buffer[c_GPSINS_BUFFER_SIZE];
  int32_t s32_GPSINSHeader;
} RAW_GPSINS_DATA_t;

typedef struct _RAW_IMU_DATA
{
  uint64_t u64_Timestamp;
  int32_t s32_Num;
  char arc_Buffer[c_IMU_BUFFER_SIZE];
  int32_t s32_IMUHeader;
} RAW_IMU_DATA_t;


typedef struct _RAW_CAN_DATA
{
  uint64_t u64_Timestamp;
  int32_t s32_Num;
  char arc_Buffer[c_CAN_BUFFER_SIZE];
  int32_t s32_CANHeader;
} RAW_CAN_DATA_t;




typedef struct _SENSOR_DATA
{
  uint64_t u64_Timestamp;
  
  RAW_LIDAR_DATA_t st_RawLIDAR;
  RAW_CAMERA_DATA_t st_RawCamera;
  RAW_GPSINS_DATA_t st_RawGPSINS;
  RAW_IMU_DATA_t st_RawIMU;
  RAW_CAN_DATA_t st_RawCAN;

  pthread_mutex_t st_MutexLIDAR;
  pthread_mutex_t st_MutexCamera;
  pthread_mutex_t st_MutexGPSINS;
  pthread_mutex_t st_MutexIMU;
  pthread_mutex_t st_MutexCAN;

} SENSOR_DATA_t;




typedef struct _VIEWER_PARAMETER
{
  int32_t s32_ScreenX = 0;
  int32_t s32_ScreenY = 0;
  int32_t s32_ScreenWidth = 0;
  int32_t s32_ScreenHeight = 0;

  int32_t s32_LocalX = 0;
  int32_t s32_LocalY = 0;
  int32_t s32_LocalWidth = 0;
  int32_t s32_LocalHeight = 0;

  int32_t s32_GlobalX = 0;
  int32_t s32_GlobalY = 0;
  int32_t s32_GlobalWidth = 0;
  int32_t s32_GlobalHeight = 0;

  int32_t s32_MenuX = 0;
  int32_t s32_MenuY = 0;
  int32_t s32_MenuWidth = 0;
  int32_t s32_MenuHeight = 0;

  int32_t s32_LoggingX = 0;
  int32_t s32_LoggingY = 0; //constant
  int32_t s32_LoggingWidth = 0;
  int32_t s32_LoggingHeight = 0;

  int32_t s32_CameraX = 0;
  int32_t s32_CameraY = 0;
  int32_t s32_CameraWidth = 0;
  int32_t s32_CameraHeight = 0;

  int32_t s32_FrameX = 0;
  int32_t s32_FrameY = 0;
  int32_t s32_FrameWidth = 0;
  int32_t s32_FrameHeight = 0;
  int32_t s32_FrameBarX = 0;
  
  float32_t f32_GlobalZoomFactor = 3.4f;
  float32_t f32_LocalZoomFactor = 3.4f;

  bool b_MoveFrame = false;
  bool b_TransLocalMap = false;
  bool b_TransGlobalMap = false;
  bool b_RotateLocalMap = false;
  bool b_RotateGlobalMap = false;

  float32_t f32_PrevX;
  float32_t f32_PrevY;

  int32_t s32_CurrentMode;
  int32_t s32_CurrentWindow = c_VIEWER_NULL;
  int32_t s32_CurrentButton = c_VIEWER_BUTTON_NULL;

  int32_t s32_State = c_STATE_MODE_NULL;
  int32_t s32_ZoomFactorState = c_STATE_1XZOOMFACTOR;
  int32_t s32_GlobalWindowState = 0;

} VIEWER_PARAMETER_t;




typedef struct _POINT
{
    float32_t f32_X;
    float32_t f32_Y;
    float32_t f32_Z;

    float32_t f32_Azimuth_deg;
    float32_t f32_Azimuth_rad;
    float32_t f32_Elevation_deg;
    float32_t f32_Elevation_rad;
    float32_t f32_Distance;
    
    uint8_t u8_Intensity;
    uint8_t u8_Channel;
    uint8_t u8_Flag;
    uint16_t aru16_VoxelIdx[3];

} POINT_t;

typedef struct _VOXEL
{
  uint16_t aru16_Grid[c_GRID_DISTANCE_SIZE][c_GRID_AZIMUTH_SIZE][c_GRID_ELEVATION_SIZE];
  int16_t ars16_ClusterIdx[c_GRID_DISTANCE_SIZE][c_GRID_AZIMUTH_SIZE][c_GRID_ELEVATION_SIZE];

} VOXEL_t;

typedef struct _CLUSTER
{
  uint16_t aru16_PointIdx[c_CLUSTER_POINT_NUM];
  int32_t s32_PointNum;

  float32_t f32_X;
  float32_t f32_Y;
  float32_t f32_Z;

  float32_t f32_MaxX;
  float32_t f32_MaxY;
  float32_t f32_MaxZ;
  float32_t f32_MinX;
  float32_t f32_MinY;
  float32_t f32_MinZ;

  float32_t f32_Volume;
  float32_t f32_Distance;
  float32_t f32_Azimuth;
  float32_t f32_Elevation;
  uint8_t u8_Class;


  // Test Data for Planning Processing
  float32_t f32_VelocityX_m_s;
  float32_t f32_VelocityY_m_s;
  float32_t f32_Velocity_m_s;
  float32_t f32_Yaw_rad_ENU;
  float32_t f32_Yaw_rad_NED;
  // Test Data for Planning Processing

  bool b_State = false;

} CLUSTER_t;

typedef struct _TRACKING
{
    float32_t f32_X;
    float32_t f32_Y;
    float32_t f32_VelocityX_m_s;
    float32_t f32_VelocityY_m_s;

    float32_t f32_ClusterX;
    float32_t f32_ClusterY;
    float32_t f32_PrevClusterX;
    float32_t f32_PrevClusterY;

    uint32_t u32_EraseCnt = 0;
    bool b_InitlzFlag = false;
    bool b_UpdateFlag = false;

    steady_clock::time_point st_PredictionTimeStart;
    steady_clock::time_point st_PredictionTimeEnd;
    float32_t f32_PredictionDt;

    steady_clock::time_point st_UpdateTimeStart;
    steady_clock::time_point st_UpdateTimeEnd;
    float32_t f32_UpdateDt;

    MatrixXf st_A = MatrixXf(4, 4);
    VectorXf st_X = VectorXf(4);
    VectorXf st_PrevX = VectorXf(4);
    VectorXf st_Z = VectorXf(4);
    MatrixXf st_H = Matrix4f::Identity();
    MatrixXf st_K = MatrixXf(4, 4);
    MatrixXf st_P = MatrixXf(4, 4);
    MatrixXf st_Q = MatrixXf(4, 4);
    MatrixXf st_R = MatrixXf(4, 4);

    // float32_t f32_Heading_rad;
    // float32_t f32_Heading_deg;
    // float32_t f32_SizeW;
    // float32_t f32_SizeH;

} TRACKING_t;


typedef struct _LIDAR_DATA
{
    uint64_t u64_Timestamp;

    POINT_t arst_Point[c_TOTAL_POINT_NUM];
    VOXEL_t st_Voxel;
    CLUSTER_t arst_Cluster[c_TOTAL_CLUSTER_NUM];
    CLUSTER_t arst_PrevCluster[c_TOTAL_CLUSTER_NUM];
    TRACKING_t arst_Tracking[c_TOTAL_TRACKING_NUM];

    int32_t s32_PointNum;
    int32_t s32_ClusterNum = 0;
    int32_t s32_PrevClusterNum = 0;
    int32_t s32_TrackingNum = 0;

} LIDAR_DATA_t;

typedef struct _KALMAN_STATE
{
  
  float64_t f64_Distance;
  float64_t f64_Angle;
  float64_t f64_DeltaDistance;
  float64_t f64_DeltaAngle;

} KALMAN_STATE;

typedef struct _LANE_COEFFICIENT
{
  float64_t f64_Slope;
  float64_t f64_Intercept;
} LANE_COEFFICIENT_t;

typedef struct _LANE_KALMAN
{
  MatrixXf st_A = MatrixXf(4,4);          // 시스템 모델 행렬
  VectorXf st_X = VectorXf(4);            // 추정값
  VectorXf st_PrevX = VectorXf(4);        // 이전 추정값
  VectorXf st_Z = VectorXf(4);            // 측정값   
  MatrixXf st_H = Matrix4f::Identity();   // 상태 전이 행렬   
  MatrixXf st_K = MatrixXf(4, 4);         // Kalman Gain
  MatrixXf st_P = MatrixXf(4, 4);         // 오차 공분산
  MatrixXf st_Q = MatrixXf(4, 4);         // 시스템 노이즈 행렬
  MatrixXf st_R = MatrixXf(4, 4);         // 관측(센서) 노이즈 행렬
  
  bool b_InitializeFlag = false;
  bool b_MeasurementUpdateFlag = false;
  bool b_IsLeft = false;
  int32_t s32_CntNoMatching = 0;

  LANE_COEFFICIENT_t st_LaneCoefficient;
  KALMAN_STATE st_LaneState;

} LANE_KALMAN_t;


typedef struct _CAMERA_PARAM
{
    std::string s_IPMParameterX;
    std::string s_IPMParameterY;

    int32_t s32_MarginX;
    int32_t s32_MarginY;

    int64_t s64_IntervalX;
    int64_t s64_IntervalY;

    int32_t s32_RemapHeight;
    int32_t s32_RemapWidth;

} CAMERA_PARAM_t;

typedef struct _CMAMERA_LANEINFO
{
  cv::Point arst_LaneSample[40];
  int32_t s32_SampleCount;
  LANE_COEFFICIENT_t st_LaneCoefficient;
  bool b_IsLeft = false;

} CAMERA_LANEINFO;


// 카메라 데이터 구조체 선언 
struct CAMERA_DATA {
    uint64_t u64_Timestamp{};
    CAMERA_PARAM_t st_CameraParameter{};
    float32_t arf32_LaneCenterX[1000]{};
    float32_t arf32_LaneCenterY[1000]{};
    int32_t s32_LaneCenterNum{};
    LANE_KALMAN_t arst_KalmanObject[10]{};
    int32_t s32_KalmanObjectNum{};
    float32_t f32_LastDistanceLeft{};
    float32_t f32_LastAngleLeft{};
    float32_t f32_LastDistanceRight{};
    float32_t f32_LastAngleRight{};
    bool b_ThereIsLeft = false;
    bool b_ThereIsRight = false;
};

typedef struct _GPS_DATA
{
    float64_t f64_Latitude;
    float64_t f64_Longitude;
    float64_t f64_Altitude;
} GPS_DATA_t;







typedef struct _IMU_DATA
{
    float32_t f32_AccelX;
    float32_t f32_AccelY;
    float32_t f32_AccelZ;

    float32_t f32_GyroX;
    float32_t f32_GyroY;
    float32_t f32_GyroZ;

    float32_t f32_QuaternionX;
    float32_t f32_QuaternionY;
    float32_t f32_QuaternionZ;
    float32_t f32_QuaternionW;
    float32_t f32_Roll_rad;
    float32_t f32_Pitch_rad;
    float32_t f32_Yaw_rad;

    float32_t f32_Roll_deg;
    float32_t f32_Pitch_deg;
    float32_t f32_Yaw_deg;


} IMU_DATA_t;







typedef struct _CAN_DATA
{
    float32_t f32_Speed_kph;
    float32_t f32_Speed_m_s;
    float32_t f32_SteerAngle_deg;
    float32_t f32_SteerAngle_rad;
    float32_t f32_Heading_deg;
    float32_t f32_Heading_rad;

} CAN_DATA_t;


typedef struct _GPS_IMU_EKF_MATRIX
{
  MatrixXf st_Q= MatrixXf(15, 15), st_H= MatrixXf(15, 15), st_K= MatrixXf(15, 15), st_P= MatrixXf(15, 15), 
            st_F= MatrixXf(15, 15), st_F2= MatrixXf(15, 15), st_A= MatrixXf(15, 15), st_R= MatrixXf(15, 15);

  MatrixXf st_X= MatrixXf(1, 9);

  MatrixXf st_dX = MatrixXf(15, 1), st_dX0 = MatrixXf(15, 1), st_m_7 = MatrixXf(15, 1);
  MatrixXf st_Cbn = MatrixXf(3, 3), st_f_ned = MatrixXf(3, 1), st_V = MatrixXf(3, 1),
            st_V_ned = MatrixXf(3, 1), st_c3 = MatrixXf(3, 1), st_Vt = MatrixXf(3, 1), st_Vt2 = MatrixXf(3, 1),
            st_Pt = MatrixXf(3, 1), st_Pt2 = MatrixXf(3, 1), st_av = MatrixXf(3, 3), st_g = MatrixXf(3, 1),
            st_accel_b = MatrixXf(3,1), st_omega_b = MatrixXf(3,1);

  MatrixXf st_w_ibb = MatrixXf(1, 3), st_w_enn = MatrixXf(1, 3), st_w_ien = MatrixXf(1, 3), st_w_inn = MatrixXf(1, 3),
                st_c1 = MatrixXf(1, 3), st_c2 = MatrixXf(1, 3), st_c31 = MatrixXf(1, 3), st_c32 = MatrixXf(1, 3), 
                st_c33 = MatrixXf(1, 3), st_c34 = MatrixXf(1, 3), st_c35 = MatrixXf(1, 3);

  MatrixXf st_acc = MatrixXf(3, 1), st_av_exp = MatrixXf(3, 3), st_av_2 = MatrixXf(3, 3), st_Rotation = MatrixXf(3, 3), st_F_pp = MatrixXf(3, 3), 
            st_F_pv = MatrixXf(3, 3), st_F_vp = MatrixXf(3, 3), st_F_vv = MatrixXf(3, 3), st_F_vphi = MatrixXf(3, 3), st_F_phip = MatrixXf(3, 3), 
            st_F_phiv = MatrixXf(3, 3), st_F_phiphi = MatrixXf(3, 3), st_Cbn_minus = MatrixXf(3, 3), st_C_B2N = MatrixXf(3,3), minCbn = MatrixXf(3, 3);
  
  VectorXf euler_angles = VectorXf(3);


} GPS_IMU_EKF_MATRIX_t;


typedef struct _EKF_DATA
{
    float64_t lat, lon, alt, h;
    float64_t gps_lat, gps_lon, gps_alt;
    float64_t lat_origin, lon_origin, alt_origin;                                                         // origin LLA(rad)
    float64_t wgs84_f, wgs84_e2;                      // wgs84 model
    float64_t ecc, R0, Ome, gravity, ecc_2; // earth value
    float64_t INSTime, lastINST, GPSTime, lastGPST;
    float64_t cos_lat, sin_lat, tan_lat;

    float32_t cur_E_prev, cur_N_prev, cur_E_aft, cur_N_aft;
    float32_t av_x, av_y, av_z;
    float32_t la_x, la_y, la_z;
    float32_t qu_x, qu_y, qu_z, qu_w;
    float32_t t0, t1, t2, t3, t4, t5;
    float32_t setting_yaw;
    float32_t roll, pitch, yaw, delta_yaw;
    float32_t q_roll, q_pitch, q_yaw;
    float32_t rho_n, rho_e, rho_d;
    float32_t f_n, f_e, f_d;
    float32_t x, y, z;
    float32_t Rm, Rt, Rmm, Rtt;
    float32_t v_e, v_n, v_u;
    float32_t gps_v_e, gps_v_n, gps_v_u;
    float32_t cur_E, cur_N, cur_U, prev_E, prev_N, prev_U;
    float32_t gps_E, gps_N, gps_U, d_E, d_N, d_U;
    float32_t dt, dt_gps;
    float32_t prev_roll, prev_pitch, prev_yaw, prev_lat, prev_lon, prev_alt, prev_v_e, prev_v_n, prev_v_u;
    
    int32_t TAU_A;
    int32_t SIG_G_D;
    int32_t TAU_G;

    bool flag0 , flag1, flag2, IMU_flag, GPS_flag, Ref_flag;

} EKF_DATA_t;


typedef struct _DEAD_RECKONING_DATA
{
    uint64_t u64_Timestamp;

    GPS_DATA_t st_GPS;
    IMU_DATA_t st_IMU;
    CAN_DATA_t st_CAN;

    float32_t f32_Latitude;
    float32_t f32_Longitude;
    float32_t f32_Altitude;

    float32_t f32_Roll_rad;
    float32_t f32_Pitch_rad;
    float32_t f32_Yaw_rad;

    float32_t f32_AccelX;
    float32_t f32_AccelY;
    float32_t f32_AccelZ;
    
    float32_t f32_GyroX;
    float32_t f32_GyroY;
    float32_t f32_GyroZ;

    float32_t f32_VehicleSpeed_m_s;
    float32_t f32_VehicleSpeed_kph;

    EKF_DATA_t st_EKF_data; 
    GPS_IMU_EKF_MATRIX_t st_EKF_Matrix;



} DEAD_RECKONING_DATA_t;

typedef struct FILE_INFO
{
    FILE* pFile;
    string st_FileName;
    int32_t s32_CntFrame;
    int32_t s32_AllSensorBufferSize;
    uint64_t ars64_Frame[c_MAX_FRAME_SIZE];
} FILE_INFO_t;



typedef struct _PATH
{
    float32_t arf32_X[c_PLANNING_MAX_PATH_NUM];
    float32_t arf32_Y[c_PLANNING_MAX_PATH_NUM];
    float32_t arf32_D[c_PLANNING_MAX_PATH_NUM];
    int32_t s32_Num;
} PATH_t;




typedef struct _FRENET_PARAM
{

    float32_t f32_WheelBase; 

    float32_t f32_MaxSpeed_kph;
    float32_t f32_MaxAccel;
    float32_t f32_MaxCurvature; 
    float32_t f32_DT;  
    float32_t f32_MaxT; 
    float32_t f32_MinT;  
    float32_t f32_TargetSpeed_m_s;  
    float32_t f32_SpeedSamplingTime; 

    int32_t s32_SampleNum; 

    float32_t f32_RobotRadius; 
    float32_t f32_KJ;
    float32_t f32_KT;
    float32_t f32_KD;
    float32_t f32_KLat;
    float32_t f32_KLon;

    //////////////////////////////////////////////////////////////////////////



    //////////////////////////////////////////////////////////////////////////
    float32_t f32_CurV;
    float32_t f32_CurA;
    float32_t f32_CurD0;
    float32_t f32_CurD1;
    float32_t f32_CurD2;
    float32_t f32_CurS0;
    float32_t f32_CurS1;
    float32_t f32_CurS2;
    /////////////////////////////////////////////////////////////////////////


} FRENET_PARAM_t;



typedef struct _QUINTIC{
    float32_t f32_XS;
    float32_t f32_VXS;
    float32_t f32_AXS;
    float32_t f32_XE;
    float32_t f32_VXE;
    float32_t f32_AXE;
    float32_t f32_A0;
    float32_t f32_A1;
    float32_t f32_A2;
    float32_t f32_A3;
    float32_t f32_A4;
    float32_t f32_A5;

    void Init(float32_t f32_tXS, float32_t f32_tVXS, float32_t f32_tAXS, float32_t f32_tXE, float32_t f32_tVXE, float32_t f32_tAXE, float32_t f32_T);
    float32_t CalcPoint(float32_t f32_T);
    float32_t CalcFirstDerivative(float32_t f32_T);
    float32_t CalcSecondDerivative(float32_t f32_T);
    float32_t CalcThirdDerivative(float32_t f32_T);
} QUINTIC_t;





typedef struct _QUARTIC{
    
    float32_t f32_XS;
    float32_t f32_VXS;
    float32_t f32_AXS;
    float32_t f32_VXE;
    float32_t f32_AXE;
    float32_t f32_A0;
    float32_t f32_A1;
    float32_t f32_A2;
    float32_t f32_A3;
    float32_t f32_A4;
    
    void Init(float32_t f32_tXS, float32_t f32_tVXS, float32_t f32_tAXS, float32_t f32_tVXE, float32_t f32_tAXE, float32_t f32_T);
    float32_t CalcPoint(float32_t f32_T);
    float32_t CalcFirstDerivative(float32_t f32_T);
    float32_t CalcSecondDerivative(float32_t f32_T);
    float32_t CalcThirdDerivative(float32_t f32_T);

} QUARTIC_t;







typedef struct _SPLINE_PARAM
{
  // y = a*x^3 + b*x^2 + c*x + d 

  float32_t arf32_A[c_PLANNING_MAX_SPLINE_NUM];
  float32_t arf32_B[c_PLANNING_MAX_SPLINE_NUM];
  float32_t arf32_C[c_PLANNING_MAX_SPLINE_NUM];
  float32_t arf32_D[c_PLANNING_MAX_SPLINE_NUM];

  float32_t arf32_X[c_PLANNING_MAX_SPLINE_NUM];
  float32_t arf32_Y[c_PLANNING_MAX_SPLINE_NUM];

  int32_t s32_NX;  


  void Init(float32_t *pf32_X, float32_t *pf32_Y, int32_t s32_Num);
  int32_t SearchNearestIndex(float32_t f32_P);
  Eigen::MatrixXd CalcA(float32_t* pf32_H);
  Eigen::MatrixXd CalcB(float32_t* pf32_H);
  float32_t Calc(float32_t f32_T);
  float32_t CalcD(float32_t f32_T);
  float32_t CalcDD(float32_t f32_T);


} SPLINE_PARAM_t;






typedef struct _SPLINE2D_PARAM
{
  SPLINE_PARAM_t st_SplineX;
  SPLINE_PARAM_t st_SplineY;

  float32_t arf32_X[c_PLANNING_MAX_SPLINE_NUM];
  float32_t arf32_Y[c_PLANNING_MAX_SPLINE_NUM];
  float32_t arf32_S[c_PLANNING_MAX_SPLINE_NUM];
  float32_t arf32_DS[c_PLANNING_MAX_SPLINE_NUM];

  int32_t s32_NX;

  void Init(PATH_t *pst_Path);
  void CalcS();
  void CalcPosition(float32_t f32_T, float32_t &f32_X, float32_t &f32_Y);
  float32_t CalcCurvature(float32_t f32_T);
  float32_t CalcYawRad(float32_t f32_T);
  float32_t GetLastS();

} SPLINE2D_PARAM_t;






typedef struct _SPLINE_PATH
{
    float32_t arf32_X[c_PLANNING_MAX_SPLINE_NUM];
    float32_t arf32_Y[c_PLANNING_MAX_SPLINE_NUM];
    float32_t arf32_Yaw_rad_ENU[c_PLANNING_MAX_SPLINE_NUM];
    float32_t arf32_Yaw_rad_NED[c_PLANNING_MAX_SPLINE_NUM];    
    float32_t arf32_Curvature[c_PLANNING_MAX_SPLINE_NUM];
    float32_t arf32_Speed_kph[c_PLANNING_MAX_SPLINE_NUM];
    float32_t arf32_S[c_PLANNING_MAX_SPLINE_NUM];
    float32_t arf32_D[c_PLANNING_MAX_SPLINE_NUM];
    int32_t s32_Num;
    int32_t s32_NearestIndex=0; 

    int32_t GetNearestIndex(float32_t f32_X, float32_t f32_Y);

} SPLINE_PATH_t;




typedef struct _FRENET_PATH {
    
    float32_t arf32_T[c_PLANNING_MAX_FRENET_NUM];
    
    float32_t arf32_D0[c_PLANNING_MAX_FRENET_NUM];
    float32_t arf32_D1[c_PLANNING_MAX_FRENET_NUM];
    float32_t arf32_D2[c_PLANNING_MAX_FRENET_NUM];
    float32_t arf32_D3[c_PLANNING_MAX_FRENET_NUM];

    float32_t arf32_S0[c_PLANNING_MAX_FRENET_NUM];
    float32_t arf32_S1[c_PLANNING_MAX_FRENET_NUM];
    float32_t arf32_S2[c_PLANNING_MAX_FRENET_NUM];
    float32_t arf32_S3[c_PLANNING_MAX_FRENET_NUM];
    
    float32_t f32_CostD;
    float32_t f32_CostV;
    float32_t f32_CostF;
    
    float32_t arf32_X[c_PLANNING_MAX_FRENET_NUM];
    float32_t arf32_Y[c_PLANNING_MAX_FRENET_NUM];
    float32_t arf32_Yaw_rad_ENU[c_PLANNING_MAX_FRENET_NUM];
    float32_t arf32_Yaw_rad_NED[c_PLANNING_MAX_FRENET_NUM];
    float32_t arf32_DS[c_PLANNING_MAX_FRENET_NUM];
    float32_t arf32_C[c_PLANNING_MAX_FRENET_NUM];
    int32_t s32_CurrentLane;
    int32_t s32_Num;

} FRENET_PATH_t;





typedef struct _VEHICLE_DATA {

    int32_t s32_VehicleIdx;
    float32_t f32_X;
    float32_t f32_Y;
    float32_t f32_Z;
    float32_t f32_VelocityX_m_s;
    float32_t f32_VelocityY_m_s;
    float32_t f32_Velocity_m_s;
    float32_t f32_AccelX;
    float32_t f32_AccelY;
    float32_t f32_Accel;

    float32_t f32_S;
    float32_t f32_D;
    float32_t f32_DS;

    float32_t f32_Yaw_rad_ENU;
    float32_t f32_Yaw_rad_NED;

    int32_t s32_CurrentLane;
    float32_t f32_TimeToCollision;
    float32_t f32_SafetyDistance;

    bool b_IsCar = false;
    bool b_IsAhead = false;
    bool b_IsBack = false;
    bool b_IsTracked = false;

} VEHICLE_DATA_t;






typedef struct _PLANNER {

    bool b_VehicleAhead;
    bool b_VehicleLeft;
    bool b_VehicleRight;

    int32_t s32_TargetLane;

    int8_t s8_AccFlag;
    int8_t s8_AEBFlag;
    int8_t s8_OverTakingFlag;
    int8_t s8_OverTakingStart;
    int8_t s8_TargetVehicleIdx;
    int8_t s8_OverTakingTry;

    int32_t s32_AccCnt;
    int32_t s32_OverTakingCnt;

    float32_t f32_SafetyDistance;

} PLANNER_t;



typedef struct _LANE_FILTER {

    int32_t s32_MemorySize = 20;
    int32_t s32_LaneHistoryIdx;
    int32_t s32_LaneHistoryCnt;
    int32_t s32_LaneHistory[20] = {0};

} LANE_FILTER_t;



typedef struct _PLANNING_DATA
{
    uint64_t u64_Timestamp;
    
    bool b_Stanley;
    bool b_PurePursuit;
    bool b_MPC;

    bool b_FixedPath;
    bool b_NonFixedPath;
    int32_t s32_BasicLane;

    PATH_t st_ReferenceLine1_global;
    PATH_t st_ReferenceLine2_global;
    PATH_t st_ReferenceLine3_global;
    PATH_t st_JokerLine_global;
    PATH_t st_LaneDistance_1_2;
    PATH_t st_LaneDistance_3_2;


    SPLINE2D_PARAM_t st_Spline2DParam;    
    SPLINE_PATH_t st_SplinePath_global;
    int32_t s32_NearestSplinePathIndex;
    float32_t f32_LastS;


    FRENET_PARAM_t st_FrenetParam;
    FRENET_PATH_t arst_FrenetPath_global[c_PLANNING_MAX_FRENET_PATH_NUM];
    int32_t s32_FrenetPathNum;
    int32_t s32_BestPath;
    int32_t s32_FrenetNearIdx;
    float32_t f32_FrenetNearDist;


    VEHICLE_DATA_t arst_VehicleData[c_TOTAL_CLUSTER_NUM];
    VEHICLE_DATA_t st_EgoVehicleData;
    int32_t s32_ClassifiedVehicleNum;

    PLANNER_t st_Planner;
    LANE_FILTER_t st_VehicleLaneFilter;
    LANE_FILTER_t st_EgoLaneFilter;

} PLANNING_DATA_t;





typedef struct _State
{
  // Current State
  float32_t f32_X;
  float32_t f32_Y;
  float32_t f32_Velocity_m_s;
  float32_t f32_Yaw_rad;
} State_t;






typedef struct _Mpc_State
{
  // Current State
  State_t st_OriginState;
  State_t st_LocalState;

  // MPC Tmp State
  State_t st_MPCState;

  // reference State
  // casadi::MX MX_refState;

  // predict State
  float32_t arf32_PredictX[c_CONTROL_HORIZON + 1];
  float32_t arf32_PredictY[c_CONTROL_HORIZON + 1];
  float32_t arf32_PredictV[c_CONTROL_HORIZON + 1];
  float32_t arf32_PredictYaw[c_CONTROL_HORIZON + 1];

  // Solve State
  float64_t arf32_SolX[c_CONTROL_HORIZON] = {0,};
  float64_t arf32_SolY[c_CONTROL_HORIZON] = {0,};
  float64_t arf32_SolV[c_CONTROL_HORIZON] = {0,};
  float64_t arf32_SolYaw[c_CONTROL_HORIZON]  = {0,};
  float64_t arf32_SolAccel[c_CONTROL_HORIZON] = {0,};
  float64_t arf32_SolSteer[c_CONTROL_HORIZON] = {0,};

} MPC_State_t;





typedef struct _MPC_Param
{
  // MPC Parameters
  const uint32_t u32_NX = 4;
  const uint32_t u32_NU = 2;
  const uint32_t u32_T = c_CONTROL_HORIZON;
  const float32_t f32_DT = 0.4;

  // vehicle parameters
  const float32_t f32_WB = 3.010;
  const float32_t f32_LF = 1.408;
  const float32_t f32_LR = 1.602;

  // Dynamics
  const float32_t f32_MAX_SPEED_m_s = 140.0 / 3.6;  // [m/s]
  const float32_t f32_MIN_SPEED_m_s = 0.0 / 3.6;    // [m/s]

  const float32_t f32_MAX_STEER_rad = 32.1 * M_PI / 180.f;      // [rad]
  const float32_t f32_MAX_DSTEER_rad_s = 27.9 * M_PI / 180.f;   // [rad/s] 
  const float32_t f32_MAX_ACCEL_m_ss = 5.6;                     // [m/ss]
  const float32_t f32_MAX_YAWRATE_rad_s = 20.0 * M_PI / 180.f;  // [rad/s]

} MPC_Param_t;






typedef struct _CONTROL_DATA
{
    float32_t f32_TargetSteer_deg;
    float32_t f32_TargetSteer_rad;

    float32_t f32_TargetSpeed_kph;
    float32_t f32_TargetSpeed_m_s;

    float32_t f32_KP;
    float32_t f32_KI;
    float32_t f32_KD;

    float32_t f32_TargetAccel;
    float32_t f32_TargetBrake;

    float32_t f32_PreError = NONE;
    float32_t f32_I_Error = 0;

    float32_t f32_LocalYaw_rad = 0;
    float32_t f32_PreYaw = 0;
    uint32_t u32_LocalYawCnt = 0;

    bool b_SpeedProfileFlag = false;

    

    MPC_Param_t st_MPCPARAM;
    MPC_State_t st_State;

} CONTROL_DATA_t;

typedef struct _LOGIC_HZ
{
  uint64_t u64_LiDARHz;
  uint64_t u64_CameraHz;
  uint64_t u64_DeadReckoningHz;
  uint64_t u64_PlanningHz;
  uint64_t u64_ControlHz;
} LOGIC_HZ_t;



uint64_t getMillisecond();
float32_t deg2rad(float32_t f32_Degree);
float32_t rad2deg(float32_t f32_Radian);

float64_t deg2rad(float64_t f64_Degree);
float64_t rad2deg(float64_t f64_Radian);

float32_t ms2kph(float32_t f32_Speed);
float32_t kph2ms(float32_t f32_Speed);

float64_t ms2kph(float64_t f64_Speed);
float64_t kph2ms(float64_t f64_Speed);

float32_t getDistance3d(float32_t f32_X1, float32_t f32_Y1, float32_t f32_Z1, float32_t f32_X2, float32_t f32_Y2, float32_t f32_Z2);
float64_t getDistance3d(float64_t f64_X1, float64_t f64_Y1, float64_t f64_Z1, float64_t f64_X2, float64_t f64_Y2, float64_t f64_Z2);

float32_t getDistance2d(float32_t f32_X1, float32_t f32_Y1, float32_t f32_X2, float32_t f32_Y2);
float64_t getDistance2d(float64_t f64_X1, float64_t f64_Y1, float64_t f64_X2, float64_t f64_Y2);

float32_t pi2pi(float32_t f32_Angle);
float64_t pi2pi(float64_t f64_Angle);

void GetModData(float32_t min, float32_t max, float32_t &data);
void GetModData(float64_t min, float64_t max, float64_t &data);

void CalcNearestDistIdx(float64_t f64_X, float64_t f64_Y, float64_t* f64_MapX, float64_t* f64_MapY, int32_t s32_MapLength, float64_t& f64_NearDist, int32_t& s32_NearIdx);
void CalcNearestDistIdx(float32_t f32_X, float32_t f32_Y, float32_t* f32_MapX, float32_t* f32_MapY, int32_t s32_MapLength, float32_t& f32_NearDist, int32_t& s32_NearIdx);

void lla2enu(float64_t f64_Lat_deg, float64_t f64_Lon_deg, float64_t f64_Alt, float32_t &f32_E, float32_t &f32_N, float32_t &f32_U);

void CreateBagFileName(char* datetime);
void CreateFile();
void LoggingFirstRead();
bool CompareFilenames(const std::string& a, const std::string& b);
int extractNumberFromFilename(const std::string& filename);
void rotationMatrix(float32_t roll, float32_t pitch, float32_t yaw, float32_t matrix[3][3]);
void rotatePoint(float32_t f32_X, float32_t f32_Y, float32_t f32_Z, float32_t roll, float32_t pitch, float32_t yaw, float32_t &f32_RX, float32_t &f32_RY, float32_t &f32_RZ);


#endif