// lane detection v2 with cpp 
// main flow : 

#include "perception/CamNode.hpp"

//################################################## Global parameter ##################################################//


cv::Mat st_IPMX;
cv::Mat st_IPMY;

//버퍼 미리 생성 
cv::Mat g_IpmImg;       // IPM 결과 (컬러)
cv::Mat g_TempImg;      // 그레이 + 이후 전처리
cv::Mat g_ResultImage;  // 시각화용
bool b_NoLaneLeft = false;
bool b_NoLaneRight = false;
CAMERA_LANEINFO st_LaneInfoLeftMain{};
CAMERA_LANEINFO st_LaneInfoRightMain{};
bool visualize = true;
static CAMERA_DATA static_camera_data;

//################################################## CameraProcessing class functions ##################################################//

// RealSense 이미지 토픽을 구독하고 시각화 창을 준비하는 ROS 노드 생성자
CameraProcessing::CameraProcessing() : rclcpp::Node("CameraProcessing_node") // rclcpp node 상속 클래스 
{
  declare_parameter<std::string>("image_topic", "/camera/camera/color/image_raw/compressed");
  const auto image_topic = get_parameter("image_topic").as_string();

  LoadParam(&static_camera_data);          // cameardata param load
  LoadMappingParam(&static_camera_data);   // cameradata IPM 맵 로드

  //img subscriber
  image_subscription_ = create_subscription<sensor_msgs::msg::CompressedImage>(image_topic, rclcpp::SensorDataQoS(),
    std::bind(&CameraProcessing::on_image, this, std::placeholders::_1)); // 콜백 함수 바인딩 : on_image


  RCLCPP_INFO(get_logger(), "Perception node subscribing to %s", image_topic.c_str()); //debug msg
}

// OpenCV 창을 정리하는 소멸자
CameraProcessing::~CameraProcessing()
{
  if (!window_name_.empty()) // window있을때 
  {
    cv::destroyWindow(window_name_); // 모든 window제거 
  }
}

//################################################## CameraProcessing img sub function ##################################################//


// 압축 이미지를 디코딩하고 프레임을 미리보기로 띄우는 콜백
void CameraProcessing::on_image(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
{
  //예외처리 
  try
  {
    cv::Mat img = cv::imdecode(msg->data, cv::IMREAD_COLOR); // cv:Mat 형식 디코딩 

    ImgProcessing(img,&static_camera_data); // img processing main pipeline function
    
  }
  catch (const cv::Exception & e) // cv에러 예외처리 
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000, "OpenCV exception during decode: %s", e.what());
  }
}


//################################################## img processing functions ##################################################//
// RAW 카메라 버퍼에서 차선 정보까지 계산하는 메인 파이프라인
void ImgProcessing(const cv::Mat& img_frame, CAMERA_DATA* camera_data)
{
    // kalman filter variables
    CAMERA_LANEINFO st_LaneInfoLeft, st_LaneInfoRight; // sliding window에서 검출된 차선 정보 담는 구조체 

    //왼쪽 차선 구분 
    st_LaneInfoLeft.b_IsLeft = true; // 왼쪽 차선 표시 --> 칼만 객체 생성에서 구분 
    st_LaneInfoLeft.st_LaneCoefficient.b_IsLeft = true ;
    KALMAN_STATE st_KalmanStateLeft, st_KalmanStateRight; // 새로 계산된 좌우 차선 거리 , 각도 저장 
    int32_t s32_I, s32_J, s32_KalmanStateCnt = 0;
    KALMAN_STATE arst_KalmanState[2] = {0};

    // 매 프레임마다 초기화
    b_NoLaneLeft  = false;
    b_NoLaneRight = false;

    // 1) IPM 결과 버퍼 준비 (카메라 해상도가 바뀔 수 있으니 create 사용)
    g_IpmImg.create(
        camera_data->st_CameraParameter.s32_RemapHeight,
        camera_data->st_CameraParameter.s32_RemapWidth,
        img_frame.type()          // CV_8UC3
    );

    // 2) Temp_Img (gray) 버퍼 준비
    g_TempImg.create(
        camera_data->st_CameraParameter.s32_RemapHeight,
        camera_data->st_CameraParameter.s32_RemapWidth,
        CV_8UC1
    );

    // 3) 결과 이미지 버퍼 준비
    g_ResultImage.create(
        camera_data->st_CameraParameter.s32_RemapHeight,
        camera_data->st_CameraParameter.s32_RemapWidth,
        CV_8UC3
    );
    g_ResultImage.setTo(cv::Scalar(0,0,0)); // 매 프레임 초기화

    // =======================  원본 → IPM remapping =======================
    cv::remap(img_frame, g_IpmImg, st_IPMX, st_IPMY,
              cv::INTER_NEAREST, cv::BORDER_CONSTANT);

    // =======================  IPM → Gray + Blur  =======================
    cv::cvtColor(g_IpmImg, g_TempImg, cv::COLOR_BGR2GRAY); //tempImg 사용
    cv::GaussianBlur(g_TempImg, g_TempImg, cv::Size(3,3), 0);

    // =======================  이진화 + 팽창 + Canny  ===================
    cv::Mat st_K = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // 이진화
    cv::threshold(g_TempImg, g_TempImg, 170, 255, cv::THRESH_BINARY);   // 170 보다 크면 255 아니면 0

    // 팽창
    cv::dilate(g_TempImg, g_TempImg, st_K);

    // Canny Edge
    cv::Canny(g_TempImg, g_TempImg, 100, 360);

    // =======================  슬라이딩 윈도우 준비  ====================
    cv::Mat st_Tmp;
    cv::findNonZero(g_TempImg, st_Tmp);    // 0이 아닌 Pixel 추출 --> 차선후보픽셀 

    int32_t s32_WindowCentorLeft  = 0; // 인덱스 0으로 초기화 
    int32_t s32_WindowCentorRight = 0;

    // (옵션) 아래 영역의 EdgeSum을 보고 디버깅하고 싶으면 사용
    // cv::Mat HalfImage = g_TempImg(cv::Range(700, g_TempImg.rows), cv::Range(0, g_TempImg.cols));
    // double totalSum = cv::sum(HalfImage)[0];

    // =======================  히스토그램 기반 시작 위치 ================== --> 중앙과 가까운 인덱스 찾는과정 (차선 시작점 )
    FindLaneStartPositions(g_TempImg,
                           s32_WindowCentorLeft, //0
                           s32_WindowCentorRight, // 0
                           b_NoLaneLeft, // false
                           b_NoLaneRight);

    //히스토그램 로직에서 중앙에 있는 엉뚱한거를 차선으로 안 잡게 로직 추가하기 

    // =======================  슬라이딩 윈도우로 좌/우 차선 탐색 ==========
    SlidingWindow(g_TempImg,
                  st_Tmp,
                  st_LaneInfoLeft,
                  st_LaneInfoRight,
                  s32_WindowCentorLeft,
                  s32_WindowCentorRight,
                  g_ResultImage);

    // ======================= Kalman Filter 단계 ========================

    if (!camera_data->b_ThereIsLeft || !camera_data->b_ThereIsRight)
    {
        // ---- 왼쪽 칼만 객체 새로 생성 ----
        if (!camera_data->b_ThereIsLeft && !b_NoLaneLeft)
        {
            st_KalmanStateLeft = CalculateKalmanState(
                st_LaneInfoLeft.st_LaneCoefficient,
                camera_data->f32_LastDistanceLeft,
                camera_data->f32_LastAngleLeft
            );

            LANE_KALMAN st_KalmanObject;
            InitializeKalmanObject(st_KalmanObject);
            UpdateObservation(st_KalmanObject, st_KalmanStateLeft);
            SetInitialX(st_KalmanObject);
            st_KalmanObject.st_LaneCoefficient = st_LaneInfoLeft.st_LaneCoefficient;

            // 좌/우 판단
            if (-(st_KalmanObject.st_LaneCoefficient.f64_Intercept /
                  st_KalmanObject.st_LaneCoefficient.f64_Slope) < 300)
                st_KalmanObject.b_IsLeft = true;
            else
                st_KalmanObject.b_IsLeft = false;

            st_KalmanObject.st_LaneState = st_KalmanStateLeft;
            camera_data->b_ThereIsLeft = true;
            camera_data->arst_KalmanObject[camera_data->s32_KalmanObjectNum] = st_KalmanObject;
            camera_data->s32_KalmanObjectNum += 1;

            DrawDrivingLane(g_ResultImage,
                            st_KalmanObject.st_LaneCoefficient,
                            cv::Scalar(255, 0, 255));
        }

        // ---- 오른쪽 칼만 객체 새로 생성 ----
        if (!camera_data->b_ThereIsRight && !b_NoLaneRight)
        {
            st_KalmanStateRight = CalculateKalmanState(
                st_LaneInfoRight.st_LaneCoefficient,
                camera_data->f32_LastDistanceRight,
                camera_data->f32_LastAngleRight
            );

            LANE_KALMAN st_KalmanObject;
            InitializeKalmanObject(st_KalmanObject);
            UpdateObservation(st_KalmanObject, st_KalmanStateRight);
            SetInitialX(st_KalmanObject);
            st_KalmanObject.st_LaneCoefficient = st_LaneInfoRight.st_LaneCoefficient;

            if (-(st_KalmanObject.st_LaneCoefficient.f64_Intercept /
                  st_KalmanObject.st_LaneCoefficient.f64_Slope) < 300)
                st_KalmanObject.b_IsLeft = true;
            else
                st_KalmanObject.b_IsLeft = false;

            st_KalmanObject.st_LaneState = st_KalmanStateRight;
            camera_data->b_ThereIsRight = true;
            camera_data->arst_KalmanObject[camera_data->s32_KalmanObjectNum] = st_KalmanObject;
            camera_data->s32_KalmanObjectNum += 1;

            DrawDrivingLane(g_ResultImage,
                            st_KalmanObject.st_LaneCoefficient,
                            cv::Scalar(255, 255, 255));
        }

        // 둘 다 없는 경우는 여기서 끝
    }
    else
    {
        // ---- 이미 Kalman Object가 있는 경우: 업데이트 ----
        // 감지된 왼쪽 차선이 있는 경우 상태 업데이트
        if (!b_NoLaneLeft)
        {
            arst_KalmanState[0] = CalculateKalmanState(
                st_LaneInfoLeft.st_LaneCoefficient,
                camera_data->f32_LastDistanceLeft,
                camera_data->f32_LastAngleLeft
            );
        }

        if (!b_NoLaneRight)
        {
            arst_KalmanState[1] = CalculateKalmanState(
                st_LaneInfoRight.st_LaneCoefficient,
                camera_data->f32_LastDistanceRight,
                camera_data->f32_LastAngleRight
            );
        }

        for (s32_I = 0; s32_I < camera_data->s32_KalmanObjectNum; s32_I++)
        {
            bool b_SameObj = false;
            // 칼만 객체와 새 관측을 비교해 동일 차선인지 판별
            for (s32_J = 0; s32_J < 2; s32_J++)
            {
                CheckSameKalmanObject(camera_data->arst_KalmanObject[s32_I],
                                      arst_KalmanState[s32_J]);  // 동일 차선인지 비교

                if (camera_data->arst_KalmanObject[s32_I].b_MeasurementUpdateFlag)
                {
                    UpdateObservation(camera_data->arst_KalmanObject[s32_I],
                                      arst_KalmanState[s32_J]);
                    PredictState(camera_data->arst_KalmanObject[s32_I]);

                    if (s32_J == 0)
                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient =
                            st_LaneInfoLeft.st_LaneCoefficient;
                    else if (s32_J == 1)
                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient =
                            st_LaneInfoRight.st_LaneCoefficient;

                    UpdateMeasurement(camera_data->arst_KalmanObject[s32_I]);
                    MakeKalmanStateBasedLaneCoef(
                        camera_data->arst_KalmanObject[s32_I],
                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient
                    );

                    if (s32_J == 0)
                        DrawDrivingLane(g_ResultImage,
                                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient,
                                        cv::Scalar(255, 0, 255));
                    else if (s32_J == 1)
                        DrawDrivingLane(g_ResultImage,
                                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient,
                                        cv::Scalar(255, 255, 255));

                    b_SameObj = true;
                    break;
                }
            }

            if (!b_SameObj)
            {
                if (camera_data->arst_KalmanObject[s32_I].s32_CntNoMatching < 20)
                {
                    camera_data->arst_KalmanObject[s32_I].s32_CntNoMatching += 1;

                    PredictState(camera_data->arst_KalmanObject[s32_I]);
                    MakeKalmanStateBasedLaneCoef(
                        camera_data->arst_KalmanObject[s32_I],
                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient
                    );
                    DrawDrivingLane(g_ResultImage,
                                    camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient,
                                    cv::Scalar(255, 255, 0));
                }
                else
                {
                    DeleteKalmanObject(*camera_data,
                                       camera_data->s32_KalmanObjectNum,
                                       s32_I);
                }
            }
        }
    }

    // 새로 계산한 직선 Coef 저장 (Kalman 이전 RANSAC 결과)
    if (!b_NoLaneLeft)
    {
        CalculateLaneCoefficient(st_LaneInfoLeft, 1000, 1);
        st_LaneInfoLeftMain = st_LaneInfoLeft;
    }
    if (!b_NoLaneRight)
    {
        CalculateLaneCoefficient(st_LaneInfoRight, 1000, 1);
        st_LaneInfoRightMain = st_LaneInfoRight;
    }

    // 필요하면 여기서 그냥 RANSAC 결과만 그릴 수도 있음
    // if (!b_NoLaneLeft) {
    //     DrawDrivingLane(g_ResultImage, st_LaneInfoLeftMain.st_LaneCoefficient, cv::Scalar(255, 0, 0));
    // }
    // if (!b_NoLaneRight) {
    //     DrawDrivingLane(g_ResultImage, st_LaneInfoRightMain.st_LaneCoefficient, cv::Scalar(0, 0, 255));
    // }

    // =======================  (H) Debug GUI ================================
    if (visualize)
    {
        cv::imshow("IPM", g_IpmImg);          // 탑뷰
        cv::imshow("Temp_Img", g_TempImg);    // 현재는 Edge 결과
        cv::imshow("st_ResultImage", g_ResultImage); // 차선 + Kalman 결과
        cv::waitKey(1);
    }
}

// 칼만 상태를 선형 차선 계수로 변환
void MakeKalmanStateBasedLaneCoef(const LANE_KALMAN& st_KalmanObject, LANE_COEFFICIENT& st_LaneCoefficient)
{
    float64_t f64_Theta_Radian;

    // printf("---------------MakeKalmanStateBasedLaneCoef-----------\n");
    f64_Theta_Radian = st_KalmanObject.st_X[2]* M_PI / 180.0;
    if (std::abs(std::cos(f64_Theta_Radian)) < 1e-6) 
    { // Use a small epsilon to check for zero
        // std::cout << "The line is vertical. Equation of the line: x = " << d / std::sin(f64_Theta_Radian) << std::endl;
    } else {
        // Calculate the slope (m) and y-intercept (c)
        st_LaneCoefficient.f64_Slope = -std::tan(f64_Theta_Radian);
        st_LaneCoefficient.f64_Intercept = st_KalmanObject.st_X[0] / std::cos(f64_Theta_Radian);

        // printf("y=%f x + %f\n",st_LaneCoefficient.f64_Slope,st_LaneCoefficient.f64_Intercept);
        // std::cout << "y = " << st_LaneCoefficient.f64_Slope << "x + " << st_LaneCoefficient.f64_Intercept << std::endl;
    }
}
 
// 매칭 실패한 칼만 차선 객체를 제거
void DeleteKalmanObject(CAMERA_DATA &pst_CameraData, int32_t& s32_KalmanObjectNum, int32_t s32_I)
{
    int32_t s32_J;
    if (s32_KalmanObjectNum == 1)
    {
        s32_KalmanObjectNum -= 1;
        if (pst_CameraData.arst_KalmanObject[0].b_IsLeft)
            pst_CameraData.b_ThereIsLeft = false;
        else
            pst_CameraData.b_ThereIsRight = false;
    }
    else
    {
        if (pst_CameraData.arst_KalmanObject[s32_I].b_IsLeft)
            pst_CameraData.b_ThereIsLeft = false;
        else
            pst_CameraData.b_ThereIsRight = false;

        for(s32_J = s32_I; s32_J<s32_KalmanObjectNum-1;s32_J++)
        {
            pst_CameraData.arst_KalmanObject[s32_J] = pst_CameraData.arst_KalmanObject[s32_J+1];
        }
        s32_KalmanObjectNum -= 1;

    }
}

// 새 관측 차선이 기존 칼만 객체와 동일한지 여부 판단
void CheckSameKalmanObject(LANE_KALMAN& st_KalmanObject, KALMAN_STATE st_KalmanStateLeft)
{
    st_KalmanObject.b_MeasurementUpdateFlag = false;
    // printf("Distance: Kalman Lane: %f, Real Time Liane: %f\n",st_KalmanObject.st_LaneState.f64_Distance,st_KalmanStateLeft.f64_Distance);
    // printf("Angle   : Kalman Lane: %f, Real Time Lane: %f\n",st_KalmanObject.st_LaneState.f64_Angle,st_KalmanStateLeft.f64_Angle);

    // Parameter yaml에서 끌어오도록 수정 필요
    if (abs(st_KalmanObject.st_LaneState.f64_Distance - st_KalmanStateLeft.f64_Distance) < 40)
    {
        if(abs(st_KalmanObject.st_LaneState.f64_Angle - st_KalmanStateLeft.f64_Angle) < 10)
        {
            st_KalmanObject.b_MeasurementUpdateFlag = true;
        }
    }
}

// 칼만 필터 예측 단계
void PredictState(LANE_KALMAN& st_KalmanObject)
{
    st_KalmanObject.st_PrevX = st_KalmanObject.st_X;

    st_KalmanObject.st_X = st_KalmanObject.st_A * st_KalmanObject.st_X;
    st_KalmanObject.st_P = st_KalmanObject.st_A * st_KalmanObject.st_P * st_KalmanObject.st_A.transpose() + st_KalmanObject.st_Q;

}

// 칼만 필터 측정 업데이트 단계
void UpdateMeasurement(LANE_KALMAN& st_KalmanObject)
{
    st_KalmanObject.st_K = st_KalmanObject.st_P * st_KalmanObject.st_H.transpose() * (st_KalmanObject.st_H * st_KalmanObject.st_P * st_KalmanObject.st_H.transpose() + st_KalmanObject.st_R).inverse();
    st_KalmanObject.st_P = st_KalmanObject.st_P - st_KalmanObject.st_K * st_KalmanObject.st_H * st_KalmanObject.st_P;
    st_KalmanObject.st_X = st_KalmanObject.st_X + st_KalmanObject.st_K * (st_KalmanObject.st_Z - st_KalmanObject.st_H * st_KalmanObject.st_X);

    st_KalmanObject.s32_CntNoMatching = 0;

}

// 관측 기반으로 상태 벡터 초기화
void SetInitialX(LANE_KALMAN& st_KalmanObject)
{
    st_KalmanObject.st_X(0) = st_KalmanObject.st_Z(0);
    st_KalmanObject.st_X(1) = st_KalmanObject.st_Z(1);
    st_KalmanObject.st_X(2) = st_KalmanObject.st_Z(2);
    st_KalmanObject.st_X(3) = st_KalmanObject.st_Z(3);
}


// 관측 벡터에 거리/각도 값을 기록
void UpdateObservation(LANE_KALMAN& st_KalmanObject, const KALMAN_STATE st_KalmanState)
{
    st_KalmanObject.st_Z(0) = st_KalmanState.f64_Distance;
    st_KalmanObject.st_Z(1) = st_KalmanState.f64_DeltaDistance;
    st_KalmanObject.st_Z(2) = st_KalmanState.f64_Angle;
    st_KalmanObject.st_Z(3) = st_KalmanState.f64_DeltaAngle;
}

// 직선 모델을 거리·각도 형태의 칼만 상태로 변환
KALMAN_STATE CalculateKalmanState(const LANE_COEFFICIENT& st_LaneCoef, float32_t& f64_Distance, float32_t& f64_Angle) 
{

    KALMAN_STATE st_KalmanState;
    float64_t s64_X, s64_Y;

    s64_X = -st_LaneCoef.f64_Slope * st_LaneCoef.f64_Intercept / (st_LaneCoef.f64_Slope*st_LaneCoef.f64_Slope + 1);
    s64_Y = st_LaneCoef.f64_Slope * s64_X + st_LaneCoef.f64_Intercept;

    st_KalmanState.f64_Distance = sqrt(pow(s64_X,2)+pow(s64_Y,2));
    st_KalmanState.f64_Angle = 90 - atan2(s64_Y,s64_X) * (180.0 / M_PI);
    // st_KalmanState.f64_Angle = atan2(s64_Y,s64_X) * (180.0 / M_PI);
    st_KalmanState.f64_DeltaDistance =  st_KalmanState.f64_Distance - f64_Distance;
    st_KalmanState.f64_DeltaAngle =  st_KalmanState.f64_Angle - f64_Angle;

    // Update Last Distance, Angle
    f64_Distance = st_KalmanState.f64_Distance;
    f64_Angle = st_KalmanState.f64_Angle;
    return st_KalmanState;
}

// 칼만 필터 행렬 및 공분산 초기화
void InitializeKalmanObject(LANE_KALMAN& st_KalmanObject)
{
    st_KalmanObject.st_A << 1, 1, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 1,
                            0, 0, 0, 1;

    st_KalmanObject.st_P << 0.01, 0   , 0   , 0,
                            0   , 0.01, 0   , 0,
                            0   , 0   , 0.01, 0,
                            0   , 0   , 0   , 0.01;

    st_KalmanObject.st_Q << 0.0001 , 0   , 0   , 0,
                            0   , 0.0001 , 0   , 0,
                            0   , 0   , 0.0001 , 0,
                            0   , 0   , 0   , 0.0001;

    st_KalmanObject.st_R << 1   , 0   , 0   , 0,
                            0   , 1   , 0   , 0,
                            0   , 0   , 1   , 0,
                            0   , 0   , 0   , 1;

    st_KalmanObject.st_X.setZero();
    st_KalmanObject.st_PrevX.setZero();
    st_KalmanObject.st_Z.setZero();

}

//############################################ sliding window function ##################################################//


// 에지 이미지에서 슬라이딩 윈도로 좌/우 차선 포인트를 추출
void SlidingWindow(const cv::Mat& st_EdgeImage, const cv::Mat& st_NonZeroPosition, CAMERA_LANEINFO& st_LaneInfoLeft, 
                        CAMERA_LANEINFO& st_LaneInfoRight, int32_t& s32_WindowCentorLeft, int32_t& s32_WindowCentorRight, cv::Mat& st_ResultImage )
{
    // const cv::Mat& st_EdgeImage,        // Canny까지 끝난 에지 이미지 (이진, 0/255)
    // const cv::Mat& st_NonZeroPosition,  // findNonZero 결과 (non-zero 픽셀들의 좌표 모음)
    // CAMERA_LANEINFO& st_LaneInfoLeft,   // 왼쪽 차선 샘플/계수 저장용 구조체 (출력)
    // CAMERA_LANEINFO& st_LaneInfoRight,  // 오른쪽 차선 샘플/계수 저장용 구조체 (출력)
    // int32_t& s32_WindowCentorLeft,      // 왼쪽 슬라이딩 윈도우의 초기 x중심 (입력 + 갱신)
    // int32_t& s32_WindowCentorRight,     // 오른쪽 슬라이딩 윈도우의 초기 x중심 (입력 + 갱신)
    // cv::Mat& st_ResultImage,            // 결과 그릴 이미지 (칼만/RANSAC 선 그릴 때 사용)
    // int32_t ImgHeight                   // 이미지 높이(세로) → 슬라이딩 시작 y

    int32_t s32_WindowHeight = st_EdgeImage.rows; // 현재 윈도우의 y (이미지 맨 아래 시작 )
    bool b_ValidWindowLeft = true; // 처음부터 Valid하다고 가정 , 윈도우를 올리지 말지 여부 판단 
    bool b_ValidWindowRight = true;

    // image size
    int cols = st_EdgeImage.cols;
    int rows = st_EdgeImage.rows;
    int32_t ImgHeight = rows - 1; 

    // x : 윈도우 가로 반폭 (50픽셀) , y : 윈도우 세로 높이 (50픽셀)
    int32_t s32_MarginX = 50, s32_MarginY = 50, s32_I,s32_J, s32_ClosestPnt, s32_CentorX, s32_CentorY;

    // 왼쪽 차선중에 최종적으로 Valid한 차선임이 판단된 Pixel Point 저장 
    std::vector<cv::Point> st_LeftWindowInds; //x,y의 포인트 
    std::vector<cv::Point> st_LeftLanePoint;
    cv::Mat st_DetectedLane(st_EdgeImage.size(), CV_8UC3, Scalar(0,0,0));

    // Lane Data reset
    st_LaneInfoLeft.s32_SampleCount = 0;
    st_LaneInfoRight.s32_SampleCount = 0;

    // Sliding Window Search 과정에서 매번 재할당. 비효율적이긴 한데 일단 구현부터. 상태관리용 변수 
    std::vector<vector<cv::Point>> st_Contours;
    bool b_CheckValidWindowLeft = false; // 이번 윈도우에서 Valid한 차선이 검출되었는지 여부
    bool b_CheckValidWindowRight = false;
    int32_t s32_CountAnvalidLeft = 0; // 연속으로 Valid하지 않은 윈도우 카운트
    int32_t s32_CountAnvalidRight = 0;
    int32_t s32_Width;
    cv::Mat st_WindowMask;
    cv::Rect st_LeftWindow;
    cv::Rect st_RightWindow;

    // 슬라이딩 윈도우 반복 (한층당 한번 반복 )
    while(s32_WindowHeight>0) 
    {
        //윈도우 좌우 경계 계산 이미지 폭 800 기준 
        // int32_t s32_WindowMinWidthLeft = s32_WindowCentorLeft - s32_MarginX >= 0 ? s32_WindowCentorLeft - s32_MarginX : 0;
        // int32_t s32_WindowMaxWidthLeft = s32_WindowCentorLeft + s32_MarginX < 800 ? s32_WindowCentorLeft + s32_MarginX : 799;
        // int32_t s32_WindowMinWidthRight = s32_WindowCentorRight - s32_MarginX >= 0 ? s32_WindowCentorRight - s32_MarginX : 0;
        // int32_t s32_WindowMaxWidthRight = s32_WindowCentorRight + s32_MarginX < 800 ? s32_WindowCentorRight + s32_MarginX : 799;
        int32_t s32_WindowMinWidthLeft  = std::max(s32_WindowCentorLeft  - s32_MarginX, 0);
        int32_t s32_WindowMaxWidthLeft  = std::min(s32_WindowCentorLeft  + s32_MarginX, cols - 1);
        int32_t s32_WindowMinWidthRight = std::max(s32_WindowCentorRight - s32_MarginX, 0);
        int32_t s32_WindowMaxWidthRight = std::min(s32_WindowCentorRight + s32_MarginX, cols - 1);
        int32_t s32_WindowMinHeight = s32_WindowHeight - s32_MarginY;
        if (s32_WindowMinHeight < 0) s32_WindowMinHeight = 0;

        int32_t s32_WidthLeft  = s32_WindowMaxWidthLeft  - s32_WindowMinWidthLeft  + 1;
        int32_t s32_WidthRight = s32_WindowMaxWidthRight - s32_WindowMinWidthRight + 1;
        // ---------------------------- Left Lane Detection ----------------------------------------
        if(!b_NoLaneLeft) //왼쪽 차선이 있을 때 
        {
            if(b_ValidWindowLeft) //윈도우를 올리기로 했을 때 
            {
                s32_Width = s32_WindowMinWidthLeft + 2*s32_MarginX < 800 ? 2*s32_MarginX : 799-s32_WindowMinWidthLeft;
                // st_LeftWindow = cv::Rect(s32_WindowMinWidthLeft,s32_WindowMinHeight,s32_Width,s32_MarginY);

                st_LeftWindow  = cv::Rect(s32_WindowMinWidthLeft,  s32_WindowMinHeight,
                          s32_WidthLeft, s32_MarginY);

                st_WindowMask = st_EdgeImage(st_LeftWindow);

                //차선 조각 검출 
                cv::findContours(st_WindowMask, st_Contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

                // Calculate Next Window Position
                for (s32_I = 0; s32_I < st_Contours.size(); s32_I++) {
                    double m00, m10, m01;
                    cv::Moments M = cv::moments(st_Contours[s32_I]);
                    m00 = M.m00; m10 = M.m10; m01 = M.m01;
                    if (m00 != 0) {
                        s32_CentorX = int(m10 / m00);
                        s32_CentorY = int(m01 / m00);
                        s32_WindowCentorLeft = s32_WindowMinWidthLeft + s32_CentorX;
                        b_CheckValidWindowLeft = true; //이번 윈도우에서 Valid한 차선 검출됨
                    }
                }

                // Draw Window 
                cv::rectangle(st_EdgeImage, cv::Point(s32_WindowMinWidthLeft, s32_WindowHeight), 
                                        cv::Point(s32_WindowMaxWidthLeft, s32_WindowMinHeight), cv::Scalar(255, 255, 255), 2);
            }

            // Check Valid Window
            if (b_CheckValidWindowLeft == false)
            {
                s32_CountAnvalidLeft += 1;
                if (s32_CountAnvalidLeft == 7) // 7개 연속으로 Valid하지 않으면 윈도우 올리기 중지
                {
                    b_ValidWindowLeft = false;
                }
            }
            else
            {
                s32_CountAnvalidLeft = 0;
                s32_ClosestPnt = INT_MAX;
                for(s32_I = 0; s32_I<st_NonZeroPosition.total();s32_I++)
                {
                    cv::Point st_Position = st_NonZeroPosition.at<Point>(s32_I);
                    if(st_Position.y >= s32_WindowMinHeight && st_Position.y <= s32_WindowHeight &&
                        st_Position.x >=  s32_WindowMinWidthLeft && st_Position.x <= s32_WindowMaxWidthLeft)
                    {
                        // st_LeftWindowInds.push_back(st_Position);
                        st_DetectedLane.at<Vec3b>(st_Position.y, st_Position.x) = Vec3b(255,0,0);
                        if(s32_ClosestPnt>cols/2-st_Position.x)
                        {
                            st_Position.y = ImgHeight + st_Position.y*(-1);
                            st_LaneInfoLeft.arst_LaneSample[st_LaneInfoLeft.s32_SampleCount] = st_Position;
                        }
                    }
                }
                st_LaneInfoLeft.s32_SampleCount += 1;
            }
            st_Contours.clear();
        }

        

        // ---------------------------- Right Lane Detection ----------------------------------------
        if(!b_NoLaneRight) // 오른쪽 차선이 있을 때
        {
            if(b_ValidWindowRight)
            {
                s32_Width = s32_WindowMinWidthRight + 2*s32_MarginX < 800 ? 2*s32_MarginX : 799-s32_WindowMinWidthRight;
                // st_RightWindow = cv::Rect(s32_WindowMinWidthRight,s32_WindowMinHeight,s32_Width,s32_MarginY);
                st_RightWindow = cv::Rect(s32_WindowMinWidthRight, s32_WindowMinHeight,
                          s32_WidthRight, s32_MarginY);
                st_WindowMask = st_EdgeImage(st_RightWindow);

                cv::findContours(st_WindowMask, st_Contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

                // Calculate Next Window Position
                for (s32_I = 0; s32_I < st_Contours.size(); s32_I++) {
                    double m00, m10, m01;
                    cv::Moments M = cv::moments(st_Contours[s32_I]);
                    m00 = M.m00; m10 = M.m10; m01 = M.m01;
                    if (m00 != 0) {
                        s32_CentorX = int(m10 / m00);
                        s32_CentorY = int(m01 / m00);
                        s32_WindowCentorRight = s32_WindowMinWidthRight + s32_CentorX;
                        b_CheckValidWindowRight = true;
                    }
                }

                // Draw Window 
                cv::rectangle(st_EdgeImage, cv::Point(s32_WindowMinWidthRight, s32_WindowHeight), cv::Point(s32_WindowMaxWidthRight, s32_WindowMinHeight), cv::Scalar(255, 255, 255), 2);
            }
            // Check Valid Window
            if (b_CheckValidWindowRight == false)
            {
                s32_CountAnvalidRight += 1;
                if (s32_CountAnvalidRight == 7)
                {
                    b_ValidWindowRight = false;
                }
            }
            else
            {
                s32_CountAnvalidRight = 0;
                s32_ClosestPnt = INT_MAX;
                for(s32_I = 0; s32_I<st_NonZeroPosition.total();s32_I++)
                {
                    cv::Point st_Position = st_NonZeroPosition.at<Point>(s32_I);
                    if(st_Position.y >= s32_WindowMinHeight && st_Position.y <= s32_WindowHeight &&
                        st_Position.x >=  s32_WindowMinWidthRight && st_Position.x <= s32_WindowMaxWidthRight)
                    {
                        // st_LeftWindowInds.push_back(st_Position);
                        st_DetectedLane.at<Vec3b>(st_Position.y, st_Position.x) = Vec3b(0,0,255);
                        if(s32_ClosestPnt>st_Position.x)
                        {
                            st_Position.y = ImgHeight + st_Position.y*(-1);
                            st_LaneInfoRight.arst_LaneSample[st_LaneInfoRight.s32_SampleCount] = st_Position;
                            s32_ClosestPnt = st_Position.x;
                        }
                    }
                }
                st_LaneInfoRight.s32_SampleCount += 1;
            }
        }


        // // Reset Parameter
        s32_WindowHeight -= s32_MarginY; // Move to Next Window Height
        b_CheckValidWindowLeft = false;
        b_CheckValidWindowRight = false;
        st_Contours.clear();
    }

    // RANSAC으로 차선 계수 산출 , 셈플 5개 미만이면 차선 없는걸로 간주 
    if (st_LaneInfoLeft.s32_SampleCount < 5)
        b_NoLaneLeft = true;
    if (st_LaneInfoRight.s32_SampleCount < 5)
        b_NoLaneRight = true;

    // 차선이 있다고 판단한 경우 ransac 수행 
    if(!b_NoLaneLeft)
    {
        CalculateLaneCoefficient(st_LaneInfoLeft,1000,1);
        st_LaneInfoLeftMain = st_LaneInfoLeft;
    }
    if(!b_NoLaneRight)
    {
        CalculateLaneCoefficient(st_LaneInfoRight,1000,1);
        st_LaneInfoRightMain = st_LaneInfoRight;
    }

}

// 추정된 차선 계수로 결과 영상에 선을 그린다
void DrawDrivingLane(cv::Mat& st_ResultImage, const LANE_COEFFICIENT st_LaneCoef, cv::Scalar st_Color)
{
    int rows = st_ResultImage.rows;
    int32_t x0 = int(-st_LaneCoef.f64_Intercept / st_LaneCoef.f64_Slope);
    int32_t x1 = int(((rows - 1) - st_LaneCoef.f64_Intercept) / st_LaneCoef.f64_Slope);
    cv::line(st_ResultImage,
            cv::Point(x0, rows - 1),
            cv::Point(x1, 0),
            st_Color, 2);    
}


// 두 포인트를 이용해 직선 모델을 구성
LANE_COEFFICIENT FitModel(const Point& st_Point1, const Point& st_Point2, bool& b_Flag)
{
    LANE_COEFFICIENT st_TmpModel;
    
    if((st_Point2.x != st_Point1.x))
    {
        st_TmpModel.f64_Slope = float((st_Point2.y - st_Point1.y) / (st_Point2.x - st_Point1.x));
        st_TmpModel.f64_Intercept = int32_t(st_Point1.y - st_TmpModel.f64_Slope * st_Point1.x);
    }
    else
        b_Flag = false;

    return st_TmpModel;
}

// 슬라이딩 윈도우로 수집한 점들에서 RANSAC으로 차선 계수를 산출
void CalculateLaneCoefficient(CAMERA_LANEINFO& st_LaneInfo, int32_t s32_Iteration, int64_t s64_Threshold)
{
    // 양쪽 차선 데이터 중 중앙선에 가장 가까운 안촉 차선 기준으로 RANSAC을 활용한 기울기 계산
    srand(time(0)); // 난수 초기화
    int32_t s32_BestInlierCount = 0, s32_I, s32_Idx1, s32_Idx2, s32_InlierCount, s32_J;
    bool b_Flag = true;
    LANE_COEFFICIENT st_Temp;

    // Coef Reset
    st_LaneInfo.st_LaneCoefficient.f64_Slope = 0;
    st_LaneInfo.st_LaneCoefficient.f64_Intercept = 0;

    for(s32_I=0;s32_I<s32_Iteration;s32_I++)
    {
        s32_Idx1 = rand() % st_LaneInfo.s32_SampleCount;
        s32_Idx2 = rand() % st_LaneInfo.s32_SampleCount;
        while(s32_Idx1 == s32_Idx2)
        {
            s32_Idx2 = rand() % st_LaneInfo.s32_SampleCount;
        }

        st_Temp = FitModel(st_LaneInfo.arst_LaneSample[s32_Idx1],st_LaneInfo.arst_LaneSample[s32_Idx2], b_Flag);

        if (b_Flag)
        {
            // Calculate Inlier
            s32_InlierCount = 0;

            for(s32_J = 0; s32_J < st_LaneInfo.s32_SampleCount; s32_J++)
            {
                if(abs(-st_Temp.f64_Slope * st_LaneInfo.arst_LaneSample[s32_J].x + st_LaneInfo.arst_LaneSample[s32_J].y - st_Temp.f64_Intercept)
                        / sqrt(st_Temp.f64_Slope*st_Temp.f64_Slope + 1) < s64_Threshold)
                {
                    s32_InlierCount += 1;
                } 
            }

            // Best Model
            if (s32_InlierCount > s32_BestInlierCount)
            {
                s32_BestInlierCount = s32_InlierCount;
                st_LaneInfo.st_LaneCoefficient = st_Temp;
            }
        }

        b_Flag = true;
    }

    // cout<<"y = "<<st_LaneInfo.st_LaneCoefficient.f64_Slope<<" * x + "<<st_LaneInfo.st_LaneCoefficient.f64_Intercept<<endl;
}

// 히스토그램에서 가장 누적 픽셀이 높은  다섯 개의 열 인덱스를 구하기 
void FindTop5MaxIndices(const int32_t* ps32_Histogram, int32_t s32_MidPoint, int32_t ars32_resultIdxs[5], bool& b_NoLane) 
{

    int32_t s32_I, s32_Cnt=0;
    std::pair<int32_t, int32_t> topValues[5];                 // (value, index)
    std::fill_n(topValues, 5, std::make_pair(0, -1));         // 0으로 초기화

    //히스토그램 순회 하면서 픽셀이 가장 높은 5개 
    for (s32_I = 0; s32_I < s32_MidPoint; ++s32_I) {
        if (ps32_Histogram[s32_I] > topValues[4].first) { //0이면 값이 안 들어감 --> lane x
            topValues[4] = std::make_pair(ps32_Histogram[s32_I], s32_I);
            std::sort(topValues, topValues + 5, std::greater<>());
        }
    }

    //결과 인덱스를 복사 
    for (s32_I = 0; s32_I < 5; ++s32_I) {
        ars32_resultIdxs[s32_I] = topValues[s32_I].second;
        if (topValues[s32_I].second == -1) // 초깃값일 경우 
        {
            s32_Cnt += 1;
        }
    }
    
    if(s32_Cnt == 5)  //초깃값 5개 --> 차선이 없다
        b_NoLane = true;
}

// 인덱스 중 중앙선과 가장 가까운 값을 반환
int32_t FindClosestToMidPoint(const int32_t points[5], int32_t s32_MidPoint) 
{
    int32_t s32_MinDistance = std::abs(points[0] - s32_MidPoint); 
    int32_t s32_ClosestIndex = points[0];
    int32_t s32_I;
    
    for (s32_I = 1; s32_I < 5; ++s32_I) {
        if (points[s32_I] == -1) continue; // 유효하지 않은 인덱스 건너뛰기

        int32_t currentDistance = std::abs(points[s32_I] - s32_MidPoint);
        if (currentDistance < s32_MinDistance) {
            s32_MinDistance = currentDistance;
            s32_ClosestIndex = points[s32_I];
        }
    }

    return s32_ClosestIndex;
}


// 히스토그램 분석으로 좌·우 슬라이딩 윈도 시작 위치를 계산
void FindLaneStartPositions(const cv::Mat& st_Edge, int32_t& s32_WindowCentorLeft, int32_t& s32_WindowCentorRight, bool& b_NoLaneLeft, bool& b_NoLaneRight) 
{

    int32_t s32_col, s32_row, s32_I; //col : y , row : x

    // Histogram 계산
    int32_t* ps32_Histogram = new int32_t[st_Edge.cols](); // 동적 할당, cols는 가로방향 픽셀 개수 만큼 배열 생성 , 모두 0으로 초기화 ; cols가 x

    // 이미지 하단 30프로  열에 해당하는 행 데이터들을 각 열별로 다 더한 후 최대가 되는 x좌표(행) 추출 --> height가 700이상이여야 작동한다. 
    for (s32_row = 0; s32_row < st_Edge.cols; ++s32_row) {
        for (s32_col = st_Edge.rows*0.7 ; s32_col < st_Edge.rows; ++s32_col) {
            ps32_Histogram[s32_row] += st_Edge.at<uchar>(s32_col, s32_row) > 0 ? 1 : 0; //검정색이 아닌 픽셀을 카운트 
        }
    }

    int32_t ars32_LeftCandidate[5], ars32_RightCandidate[5];

    //왼쪽 차선 시작점 
    // 왼쪽 및 오른쪽 최대 5개 인덱스 찾기
    FindTop5MaxIndices(ps32_Histogram, st_Edge.cols / 2, ars32_LeftCandidate, b_NoLaneLeft);
    if(!b_NoLaneLeft) // 왼쪽 차선이 감지된 경우
    {
        //가장 가까운 히스토그램 인덱스를 반환  int32_t type
        s32_WindowCentorLeft = FindClosestToMidPoint(ars32_LeftCandidate, st_Edge.cols / 2);
    }

    //오른쪽 차선 시작점 : 절반 부터 시작 
    FindTop5MaxIndices(ps32_Histogram + st_Edge.cols / 2, st_Edge.cols - st_Edge.cols / 2, ars32_RightCandidate, b_NoLaneRight);
    if(!b_NoLaneRight) //오른쪽 차선 감지된 경우 
    {
        // 오른쪽 인덱스 보정
        for (s32_I = 0; s32_I < 5; ++s32_I) {
            if (ars32_RightCandidate[s32_I] != -1) {
                ars32_RightCandidate[s32_I] += st_Edge.cols / 2; // 절반 오프셋 추가 -->원래 좌표계로 보정 
            }
        }

        s32_WindowCentorRight = FindClosestToMidPoint(ars32_RightCandidate, st_Edge.cols / 2);
    }

    delete[] ps32_Histogram; // 동적 할당 해제 
}

//################################################## load parameter  ##################################################//

// YAML 카메라 설정을 로드하고 기본 상태를 초기화
void LoadParam(CAMERA_DATA *CameraData)
{
    YAML::Node st_CameraParam = YAML::LoadFile("src/Params/Camera.yaml");
    std::cout << "Loading Camera Parameter from YAML File..." << std::endl;

    CameraData->st_CameraParameter.s_IPMParameterX = st_CameraParam["IPMParameterX"].as<std::string>();
    CameraData->st_CameraParameter.s_IPMParameterY = st_CameraParam["IPMParameterY"].as<std::string>();
    CameraData->st_CameraParameter.s32_RemapHeight = st_CameraParam["RemapHeight"].as<int32_t>();
    CameraData->st_CameraParameter.s32_RemapWidth  = st_CameraParam["RemapWidth"].as<int32_t>();

    std::cout << "Sucess to Load Camera Parameter!" << std::endl;
    // Kalman Object InitialLize
    CameraData->s32_KalmanObjectNum = 0;
    CameraData->f32_LastDistanceLeft = 0;
    CameraData->f32_LastAngleLeft = 0;
    CameraData->f32_LastDistanceRight = 0;
    CameraData->f32_LastAngleRight = 0;
}  

// IPM 맵핑 테이블을 파일에서 읽어 cv::Mat으로 구성
void LoadMappingParam(CAMERA_DATA *pst_CameraData) 
{

    // cv::Mat에서 원하는 DataType이 있기 때문에 s64_Value는 float형으로 설정해야 함
    float s64_Value;
    int32_t s32_Columns, s32_Rows;

    std::ifstream st_IPMParameters(pst_CameraData->st_CameraParameter.s_IPMParameterX);
    if (!st_IPMParameters.is_open()) {
        std::cerr << "Failed to open file: " << pst_CameraData->st_CameraParameter.s_IPMParameterX << std::endl;
        return;
    }
    st_IPMX.create(pst_CameraData->st_CameraParameter.s32_RemapHeight, pst_CameraData->st_CameraParameter.s32_RemapWidth, CV_32FC1);
    for (s32_Columns = 0; s32_Columns < pst_CameraData->st_CameraParameter.s32_RemapHeight; ++s32_Columns) {
        for (s32_Rows = 0; s32_Rows < pst_CameraData->st_CameraParameter.s32_RemapWidth; ++s32_Rows) {
            st_IPMParameters >> s64_Value;
            st_IPMX.at<float>(s32_Columns, s32_Rows) = s64_Value;
        }
    }
    st_IPMParameters.close();

    st_IPMParameters.open(pst_CameraData->st_CameraParameter.s_IPMParameterY);
    st_IPMY.create(pst_CameraData->st_CameraParameter.s32_RemapHeight, pst_CameraData->st_CameraParameter.s32_RemapWidth, CV_32FC1);
    for (s32_Columns = 0; s32_Columns < pst_CameraData->st_CameraParameter.s32_RemapHeight; ++s32_Columns) {
        for (s32_Rows = 0; s32_Rows <  pst_CameraData->st_CameraParameter.s32_RemapWidth; ++s32_Rows) {
            st_IPMParameters >> s64_Value;
            st_IPMY.at<float>(s32_Columns, s32_Rows) = s64_Value;
        }
    }
    st_IPMParameters.close();
}

// RANSAC 구현 및 Coefficient 추출 완료
// Data를 다루는 구조를 다시한번 생각 후 Kalman Filter까지 결합 진행


//################################################## Camera node main function ##################################################//

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraProcessing>()); // 객체 생성 및 spin(이벤트 루프 홀출)
  rclcpp::shutdown();
  return 0;
}
