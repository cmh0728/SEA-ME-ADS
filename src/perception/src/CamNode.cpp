// lane detection v2 with cpp 
// main flow : 

#include "perception/CamNode.hpp"

//################################################## Global parameter ##################################################//


cv::Mat st_ProcessedImage;
SENSOR_DATA_t st_SensorData{};
cv::Mat st_IPMX;
cv::Mat st_IPMY;
bool b_NoLaneLeft = false;
bool b_NoLaneRight = false;
CAMERA_LANEINFO_t st_LaneInfoLeftMain{};
CAMERA_LANEINFO_t st_LaneInfoRightMain{};
bool visualize = true;
static CAMERA_DATA_t g_camera_data;

//################################################## CameraProcessing class functions ##################################################//

// RealSense 이미지 토픽을 구독하고 시각화 창을 준비하는 ROS 노드 생성자
CameraProcessing::CameraProcessing() : rclcpp::Node("CameraProcessing_node") // rclcpp node 상속 클래스 
{
  declare_parameter<std::string>("image_topic", "/camera/camera/color/image_raw/compressed");
  const auto image_topic = get_parameter("image_topic").as_string();

  LoadParam(&g_camera_data);          // 파라미터 로드
  LoadMappingParam(&g_camera_data);   // IPM 맵 로드

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

    if(visualize) // 시각화 옵션 on 일때
    {
        cv::imshow("input img", img);
        cv::waitKey(1);  // allow OpenCV to process window events
    }
    ImgProcessing(img,&g_camera_data); // img processing main pipeline function
    
  }
  catch (const cv::Exception & e) // cv에러 예외처리 
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000, "OpenCV exception during decode: %s", e.what());
  }
}


//################################################## img processing functions ##################################################//


// RAW 카메라 버퍼에서 차선 정보까지 계산하는 메인 파이프라인
void ImgProcessing(const cv::Mat& frame, CAMERA_DATA_t* camera_data)
{

    // kalman fileter variables
    CAMERA_LANEINFO_t st_LaneInfoLeft, st_LaneInfoRight; // sliding window에서 검출된 차선 정보 담는 구조체 
    st_LaneInfoLeft.b_IsLeft = true; // 왼쪽 차선 표시 --> 칼만 객체 생성에서 구분 
    KALMAN_STATE_t st_KalmanStateLeft, st_KalmanStateRight; // 새로 계산된 좌우 차선 거리 , 각도 저장 
    int32_t s32_I, s32_J, s32_KalmanStateCnt = 0;
    KALMAN_STATE_t arst_KalmanState[2] = {0};

    // 이미지 전처리 진행할 임시 행렬 생성 
    cv::Mat st_TmpImage(camera_data->st_CameraParameter.s32_RemapHeight, camera_data->st_CameraParameter.s32_RemapWidth, CV_32FC1);
    cv::Mat st_NoneZero,st_Tmp, st_ResultImage(st_TmpImage.size(), CV_8UC3, Scalar(0,0,0)); //결과랑 중간계산 mat, 최종시각화 검정색으로 초기화 

    // 1) 콜백 frame 얕은 복사  
    cv::Mat ori_img = frame; // 원본 이미지 자체를 같은 버퍼로 사용 

    // 2) IPM 맵을 이용해 780x600 영역으로 리맵 (맵 자체는 LoadMappingParam에서 로드) --> IPM 정보 어디에 ? 
    cv::remap(ori_img,ori_img,st_IPMX, st_IPMY, cv::INTER_NEAREST, cv::BORDER_CONSTANT);

    // 3) 전처리: 그레이 변환 + 블러 (필요 시 커널 크기/유형 변경)
    cv::cvtColor(ori_img, st_TmpImage, COLOR_BGR2GRAY);
    cv::GaussianBlur(st_TmpImage, st_TmpImage, Size(1,1), 0);

    // 4) 이진화 + 팽창 + 캐니 엣지로 차선 후보 강조 (threshold, Canny 범위가 튜닝 포인트)
    cv::Mat st_K = cv::getStructuringElement(MORPH_RECT, Size(5, 5));
    cv::threshold(st_TmpImage,st_TmpImage,170,255,cv::THRESH_BINARY);   // 170 보다 크면 255 아니면 0
    cv::dilate(st_TmpImage,st_TmpImage,st_K);
    cv::Canny(st_TmpImage, st_TmpImage, 100, 360);

    // 5) 슬라이딩 윈도우 준비: 0이 아닌 픽셀 좌표 추출
    cv::findNonZero(st_TmpImage,st_Tmp);    // 0이 아닌 Pixel 추출

    // _1. 히스토그램으로 좌/우 차선의 시작 x 위치 계산 (FindLaneStartPositions 내부 로직 수정 가능)
    cv::Mat HalfImage = st_TmpImage(cv::Range(700, st_TmpImage.rows), cv::Range(0, st_TmpImage.cols));
    double totalSum = cv::sum(HalfImage)[0]; // 그레이스케일 이미지의 경우
    int32_t s32_WindowCentorLeft, s32_WindowCentorRight;
    FindLaneStartPositions(st_TmpImage, s32_WindowCentorLeft, s32_WindowCentorRight, b_NoLaneLeft, b_NoLaneRight);

    // _2. 슬라이딩 윈도우로 좌/우 차선 포인트 추출
    SlidingWindow(st_TmpImage, st_Tmp, st_LaneInfoLeft, st_LaneInfoRight, s32_WindowCentorLeft, s32_WindowCentorRight, st_ResultImage);
    // _3. Kalman Filter 단계: 기존 추적 객체와 비교해 갱신/추가 여부 결정

    if(!camera_data->b_ThereIsLeft or !camera_data->b_ThereIsRight)
    {
        if(!camera_data->b_ThereIsLeft && !b_NoLaneLeft)
        {
            // 좌측 차선이 새로 검출된 경우 칼만 객체 생성
            st_KalmanStateLeft = CalculateKalmanState(st_LaneInfoLeft.st_LaneCoefficient, camera_data->f32_LastDistanceLeft, camera_data->f32_LastAngleLeft);

            LANE_KALMAN_t st_KalmanObject;
            InitializeKalmanObject(st_KalmanObject);
            UpdateObservation(st_KalmanObject,st_KalmanStateLeft);
            SetInitialX(st_KalmanObject);
            st_KalmanObject.st_LaneCoefficient = st_LaneInfoLeft.st_LaneCoefficient;

            // Check Which Lane is
            if(-(st_KalmanObject.st_LaneCoefficient.f64_Intercept/st_KalmanObject.st_LaneCoefficient.f64_Slope) < 300)
                st_KalmanObject.b_IsLeft = true;
            else
                st_KalmanObject.b_IsLeft = false;

            st_KalmanObject.st_LaneState = st_KalmanStateLeft;
            camera_data->b_ThereIsLeft = true;
            camera_data->arst_KalmanObject[camera_data->s32_KalmanObjectNum] = st_KalmanObject;
            camera_data->s32_KalmanObjectNum += 1;
            DrawDrivingLane(st_ResultImage,st_KalmanObject.st_LaneCoefficient, cv::Scalar(255,0,255));
        }

        if(!camera_data->b_ThereIsRight && !b_NoLaneRight)
        {
            // 우측 차선이 새로 검출된 경우 칼만 객체 생성
            st_KalmanStateRight = CalculateKalmanState(st_LaneInfoRight.st_LaneCoefficient, camera_data->f32_LastDistanceRight, camera_data->f32_LastAngleRight);

            LANE_KALMAN_t st_KalmanObject;
            InitializeKalmanObject(st_KalmanObject);
            UpdateObservation(st_KalmanObject,st_KalmanStateRight);
            SetInitialX(st_KalmanObject);
            st_KalmanObject.st_LaneCoefficient = st_LaneInfoRight.st_LaneCoefficient;

            // Check Which Lane is
            if(-(st_KalmanObject.st_LaneCoefficient.f64_Intercept/st_KalmanObject.st_LaneCoefficient.f64_Slope) < 300)
                st_KalmanObject.b_IsLeft = true;
            else
                st_KalmanObject.b_IsLeft = false;

            st_KalmanObject.st_LaneState = st_KalmanStateRight;
            camera_data->b_ThereIsRight = true; 
            camera_data->arst_KalmanObject[camera_data->s32_KalmanObjectNum] = st_KalmanObject;
            camera_data->s32_KalmanObjectNum += 1;
            DrawDrivingLane(st_ResultImage,st_KalmanObject.st_LaneCoefficient, cv::Scalar(255,255,255));
        }
        
        // For Checking
        if(b_NoLaneLeft && b_NoLaneLeft)
        {
            // cout<<"--------------------Threr is No Lane--------------------"<<endl;

        }
        
    }
    // 현재 Kalman Object가 있는 경우
    else
    {   
        // 감지된 왼쪽 차선이 있는 경우 상태 업데이트
        if(!b_NoLaneLeft)
        {
            arst_KalmanState[0] = CalculateKalmanState(st_LaneInfoLeft.st_LaneCoefficient, camera_data->f32_LastDistanceLeft, camera_data->f32_LastAngleLeft);
            // s32_KalmanStateCnt ++;
        }
        if(!b_NoLaneRight)
        {
            arst_KalmanState[1] = CalculateKalmanState(st_LaneInfoRight.st_LaneCoefficient, camera_data->f32_LastDistanceRight, camera_data->f32_LastAngleRight);
            // s32_KalmanStateCnt ++;
        }
        for(s32_I = 0;s32_I<camera_data->s32_KalmanObjectNum;s32_I++)
        {

            bool b_SameObj = false;

            // 칼만 객체와 새 관측을 비교해 동일 차선인지 판별
            for(s32_J = 0; s32_J < 2; s32_J++)
            {
                CheckSameKalmanObject(camera_data->arst_KalmanObject[s32_I], arst_KalmanState[s32_J]);  // 동일 차선인지 비교
                if(camera_data->arst_KalmanObject[s32_I].b_MeasurementUpdateFlag)
                {
                    UpdateObservation(camera_data->arst_KalmanObject[s32_I],arst_KalmanState[s32_J]);
                    PredictState(camera_data->arst_KalmanObject[s32_I]);
                    if (s32_J == 0)
                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient = st_LaneInfoLeft.st_LaneCoefficient;        // 주행 차선 정보 Update
                    else if(s32_J == 1)
                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient = st_LaneInfoRight.st_LaneCoefficient;        // 주행 차선 정보 Update
                    UpdateMeasurement(camera_data->arst_KalmanObject[s32_I]);
                    MakeKalmanStateBasedLaneCoef(camera_data->arst_KalmanObject[s32_I], camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient);
                    if (s32_J == 0)
                        DrawDrivingLane(st_ResultImage,camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient, cv::Scalar(255,0,255));
                    else if(s32_J == 1)
                        DrawDrivingLane(st_ResultImage,camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient, cv::Scalar(255,255,255));

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
                    MakeKalmanStateBasedLaneCoef(camera_data->arst_KalmanObject[s32_I], camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient);
                    DrawDrivingLane(st_ResultImage,camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient, cv::Scalar(255,255,0));
                
                    // printf("------------- No Matched Kalman Lane Object & s32_CntNoMatching: %d-------------\n",camera_data->arst_KalmanObject[s32_I].s32_CntNoMatching);
                }
                else
                {
                    // printf("------------- You should Delete Kalman object & s32_CntNoMatching: %d-------------\n",camera_data->arst_KalmanObject[s32_I].s32_CntNoMatching);
                    DeleteKalmanObject(*camera_data, camera_data->s32_KalmanObjectNum, s32_I);
                }

            }

        }
    }
    // printf("------------- s32_KalmanObjectNum: %d-------------\n",camera_data->s32_KalmanObjectNum);
    

    /////////////////////////// For Checking Processed Image /////////////////////////////////////////
    b_NoLaneLeft= false;
    b_NoLaneRight = false;
    s32_KalmanStateCnt =0;

    // GUI 출력이 필요 없다면 아래 imshow / waitKey 를 주석 처리해 성능을 확보한다.
    cv::imshow("st_TmpImage",st_TmpImage);
    cv::imshow("st_ResultImage",st_ResultImage);
    cv::waitKey(1);
    //////////////////////////////////////////////////////////////////////////////////////////////////
}

// 칼만 상태를 선형 차선 계수로 변환
void MakeKalmanStateBasedLaneCoef(const LANE_KALMAN_t& st_KalmanObject, LANE_COEFFICIENT_t& st_LaneCoefficient)
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
void DeleteKalmanObject(CAMERA_DATA_t &pst_CameraData, int32_t& s32_KalmanObjectNum, int32_t s32_I)
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
void CheckSameKalmanObject(LANE_KALMAN_t& st_KalmanObject, KALMAN_STATE_t st_KalmanStateLeft)
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
void PredictState(LANE_KALMAN_t& st_KalmanObject)
{
    st_KalmanObject.st_PrevX = st_KalmanObject.st_X;

    st_KalmanObject.st_X = st_KalmanObject.st_A * st_KalmanObject.st_X;
    st_KalmanObject.st_P = st_KalmanObject.st_A * st_KalmanObject.st_P * st_KalmanObject.st_A.transpose() + st_KalmanObject.st_Q;

}

// 칼만 필터 측정 업데이트 단계
void UpdateMeasurement(LANE_KALMAN_t& st_KalmanObject)
{
    st_KalmanObject.st_K = st_KalmanObject.st_P * st_KalmanObject.st_H.transpose() * (st_KalmanObject.st_H * st_KalmanObject.st_P * st_KalmanObject.st_H.transpose() + st_KalmanObject.st_R).inverse();
    st_KalmanObject.st_P = st_KalmanObject.st_P - st_KalmanObject.st_K * st_KalmanObject.st_H * st_KalmanObject.st_P;
    st_KalmanObject.st_X = st_KalmanObject.st_X + st_KalmanObject.st_K * (st_KalmanObject.st_Z - st_KalmanObject.st_H * st_KalmanObject.st_X);

    st_KalmanObject.s32_CntNoMatching = 0;

}

// 관측 기반으로 상태 벡터 초기화
void SetInitialX(LANE_KALMAN_t& st_KalmanObject)
{
    st_KalmanObject.st_X(0) = st_KalmanObject.st_Z(0);
    st_KalmanObject.st_X(1) = st_KalmanObject.st_Z(1);
    st_KalmanObject.st_X(2) = st_KalmanObject.st_Z(2);
    st_KalmanObject.st_X(3) = st_KalmanObject.st_Z(3);
}


// 관측 벡터에 거리/각도 값을 기록
void UpdateObservation(LANE_KALMAN_t& st_KalmanObject, const KALMAN_STATE_t st_KalmanState)
{
    st_KalmanObject.st_Z(0) = st_KalmanState.f64_Distance;
    st_KalmanObject.st_Z(1) = st_KalmanState.f64_DeltaDistance;
    st_KalmanObject.st_Z(2) = st_KalmanState.f64_Angle;
    st_KalmanObject.st_Z(3) = st_KalmanState.f64_DeltaAngle;
}

// 직선 모델을 거리·각도 형태의 칼만 상태로 변환
KALMAN_STATE_t CalculateKalmanState(const LANE_COEFFICIENT_t& st_LaneCoef, float32_t& f64_Distance, float32_t& f64_Angle) 
{

    KALMAN_STATE_t st_KalmanState;
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
void InitializeKalmanObject(LANE_KALMAN_t& st_KalmanObject)
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

// 에지 이미지에서 슬라이딩 윈도로 좌/우 차선 포인트를 추출
void SlidingWindow(const cv::Mat& st_EdgeImage, const cv::Mat& st_NonZeroPosition, CAMERA_LANEINFO_t& st_LaneInfoLeft, 
                        CAMERA_LANEINFO_t& st_LaneInfoRight, int32_t& s32_WindowCentorLeft, int32_t& s32_WindowCentorRight, cv::Mat& st_ResultImage)
{

    /*
    (WindowMinWidth_Left, WindowMinHeight_Left)        
                        |------------------------------|--
                        |   ---- 2*s32_MarginX -----   ||
                        |                              | s32_MarginY
                        |                              ||
                        |------------------------------|--
                                            
    */

    int32_t s32_WindowHeight = 780;
    bool b_ValidWindowLeft = true;
    bool b_ValidWindowRight = true;

    // Parameter로 추후 뺄 것 -> 고정 Value
    // int32_t s32_MarginX = 100, s32_MarginY = 100; // For 동작 확인
    int32_t s32_MarginX = 35, s32_MarginY = 20, s32_I,s32_J, s32_ClosestPnt, s32_CentorX, s32_CentorY;

    // 최종적으로 Valid한 차선임이 판단된 Pixel Point
    vector<Point> st_LeftWindowInds;
    vector<Point> st_LeftLanePoint;
    cv::Mat st_DetectedLane(st_EdgeImage.size(), CV_8UC3, Scalar(0,0,0));

    // Lane Data
    st_LaneInfoLeft.s32_SampleCount = 0;
    st_LaneInfoRight.s32_SampleCount = 0;

    // Sliding Window Search 과정에서 매번 재할당
    vector<vector<cv::Point>> st_Contours;
    bool b_CheckValidWindowLeft = false;
    bool b_CheckValidWindowRight = false;
    int32_t s32_CountAnvalidLeft = 0;
    int32_t s32_CountAnvalidRight = 0;
    int32_t s32_Width;
    cv::Mat st_WindowMask;
    cv::Rect st_LeftWindow;
    cv::Rect st_RightWindow;

    while(s32_WindowHeight>0) 
    {

        int32_t s32_WindowMinWidthLeft = s32_WindowCentorLeft - s32_MarginX >= 0 ? s32_WindowCentorLeft - s32_MarginX : 0;
        int32_t s32_WindowMaxWidthLeft = s32_WindowCentorLeft + s32_MarginX < 600 ? s32_WindowCentorLeft + s32_MarginX : 599;
        int32_t s32_WindowMinWidthRight = s32_WindowCentorRight - s32_MarginX >= 0 ? s32_WindowCentorRight - s32_MarginX : 0;
        int32_t s32_WindowMaxWidthRight = s32_WindowCentorRight + s32_MarginX < 600 ? s32_WindowCentorRight + s32_MarginX : 599;

        int32_t s32_WindowMinHeight = s32_WindowHeight-s32_MarginY;        // Image의 가장 아래부터 Window 생성

        if(!b_NoLaneLeft)
        {
            if(b_ValidWindowLeft)
            {
                s32_Width = s32_WindowMinWidthLeft + 2*s32_MarginX < 600 ? 2*s32_MarginX : 599-s32_WindowMinWidthLeft;
                st_LeftWindow = cv::Rect(s32_WindowMinWidthLeft,s32_WindowMinHeight,s32_Width,s32_MarginY);

                st_WindowMask = st_EdgeImage(st_LeftWindow);

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
                        b_CheckValidWindowLeft = true;
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
                if (s32_CountAnvalidLeft == 7)
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
                        if(s32_ClosestPnt>300-st_Position.x)
                        {
                            st_Position.y = 780 + st_Position.y*(-1);
                            st_LaneInfoLeft.arst_LaneSample[st_LaneInfoLeft.s32_SampleCount] = st_Position;
                        }
                    }
                }
                st_LaneInfoLeft.s32_SampleCount += 1;
            }
            st_Contours.clear();
        }

        

        // ---------------------------- Right Lane Detection ----------------------------------------
        if(!b_NoLaneRight)
        {
            if(b_ValidWindowRight)
            {
                s32_Width = s32_WindowMinWidthRight + 2*s32_MarginX < 600 ? 2*s32_MarginX : 599-s32_WindowMinWidthRight;
                st_RightWindow = cv::Rect(s32_WindowMinWidthRight,s32_WindowMinHeight,s32_Width,s32_MarginY);
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
                            st_Position.y = 780 + st_Position.y*(-1);
                            st_LaneInfoRight.arst_LaneSample[st_LaneInfoRight.s32_SampleCount] = st_Position;
                            s32_ClosestPnt = st_Position.x;
                        }
                    }
                }
                st_LaneInfoRight.s32_SampleCount += 1;
            }
        }


        // // Reset Parameter
        s32_WindowHeight -= s32_MarginY;
        b_CheckValidWindowLeft = false;
        b_CheckValidWindowRight = false;
        st_Contours.clear();
    }

    // printf("-------Left--------\n");
    for(s32_I=0;s32_I<st_LaneInfoLeft.s32_SampleCount-1;s32_I++)
    {
        // printf("(%d, %d)\n",st_LaneInfoLeft.arst_LaneSample[s32_I].x, st_LaneInfoLeft.arst_LaneSample[s32_I].y);
    }
    // printf("--------Right------\n");

    // printf("-------------------\n");
    // for(s32_I=0;s32_I<st_LaneInfoRight.s32_SampleCount-1;s32_I++)
    // {
        // printf("(%d, %d)\n",st_LaneInfoRight.arst_LaneSample[s32_I].x, st_LaneInfoRight.arst_LaneSample[s32_I].y);
    // }
    // printf("-------------------\n");

    if (st_LaneInfoLeft.s32_SampleCount < 5)
        b_NoLaneLeft = true;
    if (st_LaneInfoRight.s32_SampleCount < 5)
        b_NoLaneRight = true;

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


    // Draw Line
    // int32_t x0 = int(-st_LaneInfoLeftMain.st_LaneCoefficient.f64_Intercept/st_LaneInfoLeftMain.st_LaneCoefficient.f64_Slope);
    // int32_t x1 = int((779-st_LaneInfoLeftMain.st_LaneCoefficient.f64_Intercept)/st_LaneInfoLeftMain.st_LaneCoefficient.f64_Slope);
    // cv::line(st_ResultImage, cv::Point(x0, 780), cv::Point(x1, 0), cv::Scalar(255, 0, 0), 2);

    // x0 = int(-st_LaneInfoRightMain.st_LaneCoefficient.f64_Intercept/st_LaneInfoRightMain.st_LaneCoefficient.f64_Slope);
    // x1 = int((779-st_LaneInfoRightMain.st_LaneCoefficient.f64_Intercept)/st_LaneInfoRightMain.st_LaneCoefficient.f64_Slope);
    // cv::line(st_ResultImage, cv::Point(x0, 780), cv::Point(x1, 0), cv::Scalar(0, 0, 255), 2);

    // ------------------------------------

    // imshow("st_DetectedLane",st_DetectedLane);
}

// 추정된 차선 계수로 결과 영상에 선을 그린다
void DrawDrivingLane(cv::Mat& st_ResultImage, const LANE_COEFFICIENT_t st_LaneCoef, cv::Scalar st_Color)
{
    int32_t x0 = int(-st_LaneCoef.f64_Intercept/st_LaneCoef.f64_Slope);
    int32_t x1 = int((779-st_LaneCoef.f64_Intercept)/st_LaneCoef.f64_Slope);
    cv::line(st_ResultImage, cv::Point(x0, 780), cv::Point(x1, 0), st_Color, 2);
}

// YAML 카메라 설정을 로드하고 기본 상태를 초기화
void LoadParam(CAMERA_DATA_t *pst_CameraData)
{
    YAML::Node st_CameraParam = YAML::LoadFile("cameraParam.yaml");

    pst_CameraData->st_CameraParameter.s_IPMParameterX = st_CameraParam["IPMParameterX"].as<std::string>();
    pst_CameraData->st_CameraParameter.s_IPMParameterY = st_CameraParam["IPMParameterY"].as<std::string>();
    pst_CameraData->st_CameraParameter.s32_RemapHeight = st_CameraParam["RemapHeight"].as<int32_t>();
    pst_CameraData->st_CameraParameter.s32_RemapWidth  = st_CameraParam["RemapWidth"].as<int32_t>();

    // Kalman Object InitialLize
    pst_CameraData->s32_KalmanObjectNum = 0;
    pst_CameraData->f32_LastDistanceLeft = 0;
    pst_CameraData->f32_LastAngleLeft = 0;
    pst_CameraData->f32_LastDistanceRight = 0;
    pst_CameraData->f32_LastAngleRight = 0;
}  

// 두 포인트를 이용해 직선 모델을 구성
LANE_COEFFICIENT_t FitModel(const Point& st_Point1, const Point& st_Point2, bool& b_Flag)
{
    LANE_COEFFICIENT_t st_TmpModel;
    
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
void CalculateLaneCoefficient(CAMERA_LANEINFO_t& st_LaneInfo, int32_t s32_Iteration, int64_t s64_Threshold)
{
    // 양쪽 차선 데이터 중 중앙선에 가장 가까운 안촉 차선 기준으로 RANSAC을 활용한 기울기 계산
    srand(time(0)); // 난수 초기화
    int32_t s32_BestInlierCount = 0, s32_I, s32_Idx1, s32_Idx2, s32_InlierCount, s32_J;
    bool b_Flag = true;
    LANE_COEFFICIENT_t st_Temp;

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

// 히스토그램에서 가장 높은 다섯 개의 열 인덱스를 구한다
void FindTop5MaxIndices(const int32_t* ps32_Histogram, int32_t s32_MidPoint, int32_t ars32_TopIndices[5], bool& b_NoLane) 
{

    int32_t s32_I, s32_Cnt=0;
    std::pair<int32_t, int32_t> topValues[5];                 // (value, index)
    std::fill_n(topValues, 5, std::make_pair(0, -1));         // 초기화

    for (s32_I = 0; s32_I < s32_MidPoint; ++s32_I) {
        if (ps32_Histogram[s32_I] > topValues[4].first) {
            topValues[4] = std::make_pair(ps32_Histogram[s32_I], s32_I);
            std::sort(topValues, topValues + 5, std::greater<>());
        }
    }

    for (s32_I = 0; s32_I < 5; ++s32_I) {
        ars32_TopIndices[s32_I] = topValues[s32_I].second;
        if (topValues[s32_I].second == -1)
        {
            s32_Cnt += 1;
        }
        // cout<<"ars32_TopIndices[s32_I] :"<< ars32_TopIndices[s32_I]<<endl;
    }
    
    if(s32_Cnt == 5)
        b_NoLane = true;
}

// 후보 인덱스 중 중앙선과 가장 가까운 값을 반환
int32_t FindClosestToMidPoint(const int32_t points[5], int32_t s32_MidPoint) 
{
    int32_t s32_MinDistance = std::abs(points[0] - s32_MidPoint);
    int32_t s32_ClosestIndex = points[0],s32_I;
    
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

    int32_t s32_col, s32_row, s32_I;

    // Histogram 계산
    // OpenCV의 Mat객체의 Height, Width는 우리가 생각하는 것과 반대
    int32_t* ps32_Histogram = new int32_t[st_Edge.cols](); // 동적 할당, 모두 0으로 초기화

    // 700~780 열에 해당하는 행 데이터들을 각 열별로 다 더한 후 최대가 되는 x좌표(행) 추출
    for (s32_row = 0; s32_row < st_Edge.cols; ++s32_row) {
        for (s32_col = 700; s32_col < st_Edge.rows; ++s32_col) {
            ps32_Histogram[s32_row] += st_Edge.at<uchar>(s32_col, s32_row) > 0 ? 1 : 0;
        }
    }

    int32_t ars32_LeftCandidate[5], ars32_RightCandidate[5];

    // cout<<"Left"<<endl;
    // 왼쪽 및 오른쪽 최대 5개 인덱스 찾기
    FindTop5MaxIndices(ps32_Histogram, st_Edge.cols / 2, ars32_LeftCandidate, b_NoLaneLeft);
    if(!b_NoLaneLeft)
    {
        s32_WindowCentorLeft = FindClosestToMidPoint(ars32_LeftCandidate, st_Edge.cols / 2);
    }

    // cout<<"Right"<<endl;

    FindTop5MaxIndices(ps32_Histogram + st_Edge.cols / 2, st_Edge.cols - st_Edge.cols / 2, ars32_RightCandidate, b_NoLaneRight);
    if(!b_NoLaneRight)
    {
        // 오른쪽 인덱스 보정
        for (s32_I = 0; s32_I < 5; ++s32_I) {
            if (ars32_RightCandidate[s32_I] != -1) {
                ars32_RightCandidate[s32_I] += st_Edge.cols / 2;
            }
        }

        s32_WindowCentorRight = FindClosestToMidPoint(ars32_RightCandidate, st_Edge.cols / 2);
    }

    delete[] ps32_Histogram;
}

// IPM 맵핑 테이블을 파일에서 읽어 cv::Mat으로 구성
void LoadMappingParam(CAMERA_DATA_t *pst_CameraData) 
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

// ROS 런타임을 올리고 CameraProcessing 노드를 실행
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraProcessing>()); // 객체 생성 및 spin(이벤트 루프 홀출)
  rclcpp::shutdown();
  return 0;
}
