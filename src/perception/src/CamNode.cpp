// lane detection v2

#include "perception/CamNode.hpp"
#include "perception/msg/lane_pnt.hpp"

// ransac 난수 초기화 전역설정
struct RansacRandomInit {
    RansacRandomInit() { std::srand(static_cast<unsigned int>(std::time(nullptr))); }
} g_ransacRandomInit;

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
bool track_bar = false;

// histogram param
bool b_IsprevHistogram = false ;
int32_t g_PrevWindowCenterLeft  = -1;
int32_t g_PrevWindowCenterRight = -1;

static CAMERA_DATA static_camera_data;

//################################################## helper function ##################################################//

namespace
{
perception::msg::Lane build_lane_message(const CAMERA_LANEINFO & lane_info)
{
    perception::msg::Lane lane_msg;
    const int32_t max_samples = static_cast<int32_t>(sizeof(lane_info.arst_LaneSample) /
                                                    sizeof(lane_info.arst_LaneSample[0]));
    const int32_t clamped_samples = std::min(lane_info.s32_SampleCount, max_samples);
    lane_msg.lane_points.reserve(clamped_samples);

    for (int32_t i = 0; i < clamped_samples; ++i)
    {
        const cv::Point & sample = lane_info.arst_LaneSample[i];
        perception::msg::LanePnt point_msg;
        point_msg.x = static_cast<float>(sample.x);
        point_msg.y = static_cast<float>(sample.y);
        lane_msg.lane_points.push_back(point_msg);
    }

    return lane_msg;
}
}  // namespace

// ======= 전역 설정값 (트랙바랑 연결할 애들) =======
int g_thresh      = 160;  // 이진화 임계값
int g_canny_low   = 140;  // Canny low
int g_canny_high  = 330;  // Canny high
int g_dilate_ksize = 8;   // 팽창 커널 크기

void on_trackbar(int, void*)
{
    // 트랙바 콜백은 안 써도 됨. 값은 전역 변수에 자동으로 들어감.
}

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

  lane_left_pub_ = create_publisher<perception::msg::Lane>("/lane/left", rclcpp::QoS(10));
  lane_right_pub_ = create_publisher<perception::msg::Lane>("/lane/right", rclcpp::QoS(10));


  RCLCPP_INFO(get_logger(), "Perception node subscribing to %s", image_topic.c_str()); //debug msg

  if (track_bar)
  {
    // 디버그용 윈도우 + 트랙바 컨트롤 창
    cv::namedWindow("IPM");
    cv::namedWindow("Temp_Img");
    cv::namedWindow("st_ResultImage");
    cv::namedWindow("PreprocessControl");

    cv::createTrackbar("Threshold", "PreprocessControl", nullptr, 255, on_trackbar);
    cv::createTrackbar("CannyLow",  "PreprocessControl", nullptr, 500, on_trackbar);
    cv::createTrackbar("CannyHigh", "PreprocessControl", nullptr, 500, on_trackbar);
    cv::createTrackbar("DilateK",   "PreprocessControl", nullptr, 31,  on_trackbar);

  }
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
    publish_lane_messages();
    
  }
  catch (const cv::Exception & e) // cv에러 예외처리 
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000, "OpenCV exception during decode: %s", e.what());
  }
}

void CameraProcessing::publish_lane_messages()
{
  auto publish_single_lane = [](const CAMERA_LANEINFO & lane_info,
                                bool lane_missing,
                                const rclcpp::Publisher<perception::msg::Lane>::SharedPtr & publisher)
  {
    if (!publisher || lane_missing)
    {
      return;
    }

    auto lane_msg = build_lane_message(lane_info);
    if (lane_msg.lane_points.empty())
    {
      return;
    }

    publisher->publish(lane_msg);
  };

  publish_single_lane(st_LaneInfoLeftMain, b_NoLaneLeft, lane_left_pub_);
  publish_single_lane(st_LaneInfoRightMain, b_NoLaneRight, lane_right_pub_);
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

    // =======================  이미지 전처리 튜닝  ===================
    if(track_bar)
    {
        int thresh     = cv::getTrackbarPos("Threshold", "PreprocessControl");
        int canny_low  = cv::getTrackbarPos("CannyLow",  "PreprocessControl");
        int canny_high = cv::getTrackbarPos("CannyHigh", "PreprocessControl");
        int ksize      = cv::getTrackbarPos("DilateK",   "PreprocessControl");

        // 커널 크기 보정
        if (ksize < 1) ksize = 1;
        if (ksize % 2 == 0) ksize += 1;

        if (canny_low > canny_high) std::swap(canny_low, canny_high);

        cv::Mat st_K = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));

        cv::threshold(g_TempImg, g_TempImg, thresh, 255, cv::THRESH_BINARY);
        cv::dilate(g_TempImg, g_TempImg, st_K);
        cv::Canny(g_TempImg, g_TempImg, canny_low, canny_high);
    }
    else
    {
    // =======================  이진화 + 팽창 + Canny  ===================
        cv::Mat st_K = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

        // 이진화
        cv::threshold(g_TempImg, g_TempImg, 170, 255, cv::THRESH_BINARY);   // 170 보다 크면 255 아니면 0

        // 팽창
        cv::dilate(g_TempImg, g_TempImg, st_K);

        // Canny Edge , 140보다 낮으면 무시, 330보다 높으면 엣지 
        cv::Canny(g_TempImg, g_TempImg, 140, 330); // non zero 스캔 때문에 canny가 더 빠름 
    }

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
    // 결과물 Left/Right Lane Info main에 저장
    SlidingWindow(g_TempImg,
                  st_Tmp,
                  st_LaneInfoLeft,
                  st_LaneInfoRight,
                  s32_WindowCentorLeft,
                  s32_WindowCentorRight,
                  g_ResultImage);
    
    // ======================= RANSAC으로 차선 계수 산출 ====================

    // RANSAC으로 차선 계수 산출 , 셈플 5개 미만이면 차선 없는걸로 간주 
    if (st_LaneInfoLeft.s32_SampleCount < 5)
        b_NoLaneLeft = true;
    if (st_LaneInfoRight.s32_SampleCount < 5)
        b_NoLaneRight = true;

    // 차선이 있다고 판단한 경우 ransac 수행 
    if(!b_NoLaneLeft)
    {
        CalculateLaneCoefficient(st_LaneInfoLeft,1000,1); // iteration, threshold
        st_LaneInfoLeftMain = st_LaneInfoLeft; // 관측값 보관용 
    }
    if(!b_NoLaneRight)
    {
        CalculateLaneCoefficient(st_LaneInfoRight,1000,1);
        st_LaneInfoRightMain = st_LaneInfoRight;
    }

    // ======================= Kalman Filter 단계 ========================

    if (!camera_data->b_ThereIsLeft || !camera_data->b_ThereIsRight) // 왼쪽 or 오른쪽 칼만 객체 모두 없는 경우 
    {
        int margin = 50 ; // b_IsLeft 판단용 마진
        // ---- 왼쪽 칼만 객체 새로 생성 ----
        if (!camera_data->b_ThereIsLeft && !b_NoLaneLeft) // 왼쪽 칼만 객체 없고, 왼쪽 차선 감지된 경우
        {
            // Kalman state structure 계산
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

            // x 절편 로직 새로 구성 
            int center_x = camera_data->st_CameraParameter.s32_RemapWidth / 2;
            double x_intercept = -st_KalmanObject.st_LaneCoefficient.f64_Intercept /
                     st_KalmanObject.st_LaneCoefficient.f64_Slope;
            
            st_KalmanObject.b_IsLeft = (x_intercept < center_x - margin);

            // 좌/우 판단 (x절편 계산)
            // if (-(st_KalmanObject.st_LaneCoefficient.f64_Intercept /
            //       st_KalmanObject.st_LaneCoefficient.f64_Slope) < 300)
            //     st_KalmanObject.b_IsLeft = true;
            // else
            //     st_KalmanObject.b_IsLeft = false;

            st_KalmanObject.st_LaneState = st_KalmanStateLeft;
            camera_data->b_ThereIsLeft = true;
            camera_data->arst_KalmanObject[camera_data->s32_KalmanObjectNum] = st_KalmanObject;
            camera_data->s32_KalmanObjectNum += 1;

            DrawDrivingLane(g_ResultImage,
                            st_KalmanObject.st_LaneCoefficient,
                            cv::Scalar(255, 0, 255));
        }

        // ---- 오른쪽 칼만 객체 새로 생성 ----
        if (!camera_data->b_ThereIsRight && !b_NoLaneRight) // 오른쪽 칼만 객체 없고, 오른쪽 차선 감지된 경우
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

            // b_IsLeft 판단 : false --> 오 / true --> 왼
            int center_x = camera_data->st_CameraParameter.s32_RemapWidth / 2;
            double x_intercept = -st_KalmanObject.st_LaneCoefficient.f64_Intercept /
                     st_KalmanObject.st_LaneCoefficient.f64_Slope;
            
            st_KalmanObject.b_IsLeft = (x_intercept < center_x + margin);

            // if (-(st_KalmanObject.st_LaneCoefficient.f64_Intercept /
            //       st_KalmanObject.st_LaneCoefficient.f64_Slope) < 300)
            //     st_KalmanObject.b_IsLeft = true;
            // else
            //     st_KalmanObject.b_IsLeft = false;

            st_KalmanObject.st_LaneState = st_KalmanStateRight;
            camera_data->b_ThereIsRight = true;
            camera_data->arst_KalmanObject[camera_data->s32_KalmanObjectNum] = st_KalmanObject;
            camera_data->s32_KalmanObjectNum += 1;

            DrawDrivingLane(g_ResultImage,
                            st_KalmanObject.st_LaneCoefficient,
                            cv::Scalar(255, 255, 255));
        }

    }

    else // 하나 이상의 칼만 객체가 이미 있는 경우
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

    // =======================  RANSAC 디버그 창 ===========================
    // RANSAC 직선만 IPM 이미지 복사해서 사용
    if (visualize)
    {
        // 1) RANSAC 직선만 보고 싶으면: IPM 이미지 복사해서 사용
        cv::Mat ransac_debug;
        g_IpmImg.copyTo(ransac_debug);   // 또는 g_TempImg를 COLOR_GRAY2BGR로 변환해서 써도 됨

        if (!b_NoLaneLeft) {
            DrawDrivingLane(ransac_debug,
                            st_LaneInfoLeftMain.st_LaneCoefficient,
                            cv::Scalar(255, 0, 0));   // 파란/빨간 아무 색
        }

        if (!b_NoLaneRight) {
            DrawDrivingLane(ransac_debug,
                            st_LaneInfoRightMain.st_LaneCoefficient,
                            cv::Scalar(0, 0, 255));
        }

        cv::imshow("RANSAC Debug", ransac_debug);   // RANSAC 전용 창
    }

    // =======================  (H) Debug GUI ================================
    if (visualize)
    {
        cv::imshow("IPM", g_IpmImg);          // 탑뷰
        cv::imshow("Temp_Img", g_TempImg);    // 현재는 Edge 결과
        cv::imshow("st_ResultImage", g_ResultImage); // 차선 + Kalman 결과
        cv::waitKey(1);
    }
}
//######################################### MakeKalmanStateBasedLaneCoef func  ##################################################//

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
 //######################################### DeleteKalmanObject func  ##################################################//

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
//######################################### CheckSameKalmanObject func  ##################################################//

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
//######################################### PredictState func  ##################################################//

// 칼만 필터 예측 단계
void PredictState(LANE_KALMAN& st_KalmanObject)
{
    st_KalmanObject.st_PrevX = st_KalmanObject.st_X;

    st_KalmanObject.st_X = st_KalmanObject.st_A * st_KalmanObject.st_X;
    st_KalmanObject.st_P = st_KalmanObject.st_A * st_KalmanObject.st_P * st_KalmanObject.st_A.transpose() + st_KalmanObject.st_Q;

}

//######################################### UpdateMeasurement func  ##################################################//

// 칼만 필터 측정 업데이트 단계
void UpdateMeasurement(LANE_KALMAN& st_KalmanObject)
{
    st_KalmanObject.st_K = st_KalmanObject.st_P * st_KalmanObject.st_H.transpose() * (st_KalmanObject.st_H * st_KalmanObject.st_P * st_KalmanObject.st_H.transpose() + st_KalmanObject.st_R).inverse();
    st_KalmanObject.st_P = st_KalmanObject.st_P - st_KalmanObject.st_K * st_KalmanObject.st_H * st_KalmanObject.st_P;
    st_KalmanObject.st_X = st_KalmanObject.st_X + st_KalmanObject.st_K * (st_KalmanObject.st_Z - st_KalmanObject.st_H * st_KalmanObject.st_X);

    st_KalmanObject.s32_CntNoMatching = 0;

}

//######################################### SetInitialX func  ##################################################//

// 관측 기반으로 상태 벡터 초기화
void SetInitialX(LANE_KALMAN& st_KalmanObject)
{
    st_KalmanObject.st_X(0) = st_KalmanObject.st_Z(0);
    st_KalmanObject.st_X(1) = st_KalmanObject.st_Z(1);
    st_KalmanObject.st_X(2) = st_KalmanObject.st_Z(2);
    st_KalmanObject.st_X(3) = st_KalmanObject.st_Z(3);
}

//######################################### UpdateObservation func  ##################################################//

// 관측 벡터에 거리/각도 값을 기록
void UpdateObservation(LANE_KALMAN& st_KalmanObject, const KALMAN_STATE st_KalmanState)
{
    st_KalmanObject.st_Z(0) = st_KalmanState.f64_Distance;
    st_KalmanObject.st_Z(1) = st_KalmanState.f64_DeltaDistance;
    st_KalmanObject.st_Z(2) = st_KalmanState.f64_Angle;
    st_KalmanObject.st_Z(3) = st_KalmanState.f64_DeltaAngle;
}

//######################################### CalculateKalmanState func  ##################################################//

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

//######################################### InitializeKalmanObject func  ##################################################//

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

//######################################### SlidingWindow func ##################################################//

// 에지 이미지에서 슬라이딩 윈도로 좌/우 차선 포인트를 추출
void SlidingWindow(const cv::Mat& st_EdgeImage,
                   const cv::Mat& st_NonZeroPosition,
                   CAMERA_LANEINFO& st_LaneInfoLeft,
                   CAMERA_LANEINFO& st_LaneInfoRight,
                   int32_t& s32_WindowCentorLeft,
                   int32_t& s32_WindowCentorRight,
                   cv::Mat& st_ResultImage)
{
    // 이미지 크기
    const int cols = st_EdgeImage.cols; // x
    const int rows = st_EdgeImage.rows; // y
    const int32_t ImgHeight = rows - 1; // IPM 좌표계 맞추기용

    // 윈도우 파라미터
    int32_t s32_WindowHeight = rows - 1;        // 현재 윈도우의 "아래쪽" y (맨 아래에서 시작)
    const int32_t s32_MarginX = 30;             // 윈도우 가로 반폭
    const int32_t s32_MarginY = 60;             // 윈도우 세로 높이
    int32_t s32_I, s32_CentorX, s32_CentorY;

    // 유효 윈도우 유지 여부
    bool b_ValidWindowLeft  = true;
    bool b_ValidWindowRight = true;

    // 디버그용 이미지 (선택 포인트를 색칠)
    cv::Mat st_DetectedLane(st_EdgeImage.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    // RANSAC용 샘플 개수 초기화
    st_LaneInfoLeft.s32_SampleCount  = 0;
    st_LaneInfoRight.s32_SampleCount = 0;

    // 컨투어 버퍼
    std::vector<std::vector<cv::Point>> st_Contours;

    // 윈도우 유효 여부 플래그
    bool b_CheckValidWindowLeft  = false;
    bool b_CheckValidWindowRight = false;

    // 연속으로 실패한 윈도우 카운트
    int32_t s32_CountAnvalidLeft  = 0;
    int32_t s32_CountAnvalidRight = 0;

    cv::Mat st_WindowMask;
    cv::Rect st_LeftWindow;
    cv::Rect st_RightWindow;

    const int img_center_x = cols / 2; // 이미지 전체 중앙

    // ====================== 슬라이딩 윈도우 루프 ======================
    while (s32_WindowHeight > 0)
    {
        // 윈도우 세로(높이) 방향 경계
        int32_t s32_WindowMinHeight = s32_WindowHeight - s32_MarginY;
        if (s32_WindowMinHeight < 0) s32_WindowMinHeight = 0;

        // ---- 현재 Left/Right 윈도우의 좌우 경계 계산 ----
        int32_t s32_WindowMinWidthLeft  = std::max(s32_WindowCentorLeft  - s32_MarginX, 0);
        int32_t s32_WindowMaxWidthLeft  = std::min(s32_WindowCentorLeft  + s32_MarginX, cols - 1);
        int32_t s32_WindowMinWidthRight = std::max(s32_WindowCentorRight - s32_MarginX, 0);
        int32_t s32_WindowMaxWidthRight = std::min(s32_WindowCentorRight + s32_MarginX, cols - 1);

        int32_t s32_WidthLeft  = s32_WindowMaxWidthLeft  - s32_WindowMinWidthLeft  + 1;
        int32_t s32_WidthRight = s32_WindowMaxWidthRight - s32_WindowMinWidthRight + 1;

        // --- 윈도우 유효 플래그 초기화 (이번 루프에서 다시 세팅) ---
        b_CheckValidWindowLeft  = false;
        b_CheckValidWindowRight = false;

        // ======================= Left Lane =======================
        if (!b_NoLaneLeft)
        {
            if (b_ValidWindowLeft)
            {
                // 현재 윈도우 영역(Rect) 정의
                st_LeftWindow = cv::Rect(
                    s32_WindowMinWidthLeft,
                    s32_WindowMinHeight,
                    s32_WidthLeft,
                    s32_MarginY);

                // 해당 영역만 잘라서 컨투어 검출
                st_WindowMask = st_EdgeImage(st_LeftWindow);
                cv::findContours(st_WindowMask, st_Contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

                // 모멘트 기반 윈도우 내부 중심 계산 → 다음 윈도우 x 중심 갱신
                for (s32_I = 0; s32_I < (int)st_Contours.size(); ++s32_I)
                {
                    cv::Moments M = cv::moments(st_Contours[s32_I]);
                    double m00 = M.m00, m10 = M.m10, m01 = M.m01;
                    if (m00 != 0.0)
                    {
                        s32_CentorX = static_cast<int>(m10 / m00);
                        s32_CentorY = static_cast<int>(m01 / m00);
                        s32_WindowCentorLeft = s32_WindowMinWidthLeft + s32_CentorX; // 로컬→글로벌 x
                        b_CheckValidWindowLeft = true;
                    }
                }

                // 디버그용 윈도우 시각화
                cv::rectangle(
                    st_EdgeImage,
                    cv::Point(s32_WindowMinWidthLeft,  s32_WindowMinHeight),
                    cv::Point(s32_WindowMaxWidthLeft,  s32_WindowHeight),
                    cv::Scalar(255, 255, 255),
                    2);
            }

            // ---- 이번 윈도우에서 유효한 차선 조각을 못 찾은 경우 ----
            if (!b_CheckValidWindowLeft)
            {
                ++s32_CountAnvalidLeft;
                if (s32_CountAnvalidLeft >= 7)
                {
                    b_ValidWindowLeft = false; // 더 이상 윈도우 올리지 않음
                }
            }
            else
            {
                // 유효 윈도우 → 샘플 픽셀 선택
                s32_CountAnvalidLeft = 0;
                double best_score = std::numeric_limits<double>::max();

                // 새로 갱신된 윈도우 중심
                const int window_center_x_left = s32_WindowCentorLeft;

                for (int idx = 0; idx < st_NonZeroPosition.total(); ++idx)
                {
                    cv::Point st_Position = st_NonZeroPosition.at<cv::Point>(idx);

                    // 현재 윈도우 영역 내의 픽셀만 사용
                    if (st_Position.y >= s32_WindowMinHeight && st_Position.y <= s32_WindowHeight &&
                        st_Position.x >= s32_WindowMinWidthLeft && st_Position.x <= s32_WindowMaxWidthLeft)
                    {
                        // 디버깅용 색칠
                        st_DetectedLane.at<cv::Vec3b>(st_Position.y, st_Position.x) = cv::Vec3b(255, 0, 0);

                        // 윈도우 중심 + 이미지 중심 복합 스코어
                        double d_window = std::abs(st_Position.x - window_center_x_left);
                        double d_global = std::abs(st_Position.x - img_center_x);
                        double score    = 0.7 * d_window + 0.3 * d_global;

                        if (score < best_score)
                        {
                            best_score = score;
                            cv::Point ipmPt = st_Position;
                            ipmPt.y = ImgHeight - ipmPt.y; // 좌표계 반전(IPM 기준)
                            st_LaneInfoLeft.arst_LaneSample[st_LaneInfoLeft.s32_SampleCount] = ipmPt;
                        }
                    }
                }

                if (best_score < std::numeric_limits<double>::max())
                {
                    ++st_LaneInfoLeft.s32_SampleCount;
                }
            }

            st_Contours.clear();
        }

        // ======================= Right Lane =======================
        if (!b_NoLaneRight)
        {
            if (b_ValidWindowRight)
            {
                st_RightWindow = cv::Rect(
                    s32_WindowMinWidthRight,
                    s32_WindowMinHeight,
                    s32_WidthRight,
                    s32_MarginY);

                st_WindowMask = st_EdgeImage(st_RightWindow);
                cv::findContours(st_WindowMask, st_Contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

                for (s32_I = 0; s32_I < (int)st_Contours.size(); ++s32_I)
                {
                    cv::Moments M = cv::moments(st_Contours[s32_I]);
                    double m00 = M.m00, m10 = M.m10, m01 = M.m01;
                    if (m00 != 0.0)
                    {
                        s32_CentorX = static_cast<int>(m10 / m00);
                        s32_CentorY = static_cast<int>(m01 / m00);
                        s32_WindowCentorRight = s32_WindowMinWidthRight + s32_CentorX;
                        b_CheckValidWindowRight = true;
                    }
                }

                cv::rectangle(
                    st_EdgeImage,
                    cv::Point(s32_WindowMinWidthRight,  s32_WindowMinHeight),
                    cv::Point(s32_WindowMaxWidthRight,  s32_WindowHeight),
                    cv::Scalar(255, 255, 255),
                    2);
            }

            if (!b_CheckValidWindowRight)
            {
                ++s32_CountAnvalidRight;
                if (s32_CountAnvalidRight >= 7)
                {
                    b_ValidWindowRight = false;
                }
            }
            else
            {
                s32_CountAnvalidRight = 0;
                double best_score = std::numeric_limits<double>::max();

                const int window_center_x_right = s32_WindowCentorRight;

                for (int idx = 0; idx < st_NonZeroPosition.total(); ++idx)
                {
                    cv::Point st_Position = st_NonZeroPosition.at<cv::Point>(idx);

                    if (st_Position.y >= s32_WindowMinHeight && st_Position.y <= s32_WindowHeight &&
                        st_Position.x >= s32_WindowMinWidthRight && st_Position.x <= s32_WindowMaxWidthRight)
                    {
                        st_DetectedLane.at<cv::Vec3b>(st_Position.y, st_Position.x) = cv::Vec3b(0, 0, 255);

                        double d_window = std::abs(st_Position.x - window_center_x_right);
                        double d_global = std::abs(st_Position.x - img_center_x);
                        double score    = 0.7 * d_window + 0.3 * d_global;

                        if (score < best_score)
                        {
                            best_score = score;
                            cv::Point ipmPt = st_Position;
                            ipmPt.y = ImgHeight - ipmPt.y;
                            st_LaneInfoRight.arst_LaneSample[st_LaneInfoRight.s32_SampleCount] = ipmPt;
                        }
                    }
                }

                if (best_score < std::numeric_limits<double>::max())
                {
                    ++st_LaneInfoRight.s32_SampleCount;
                }
            }

            st_Contours.clear();
        }

        // ======================= 다음 윈도우로 이동 =======================
        s32_WindowHeight -= s32_MarginY;
    }

}

//###################################### DrawDrivingLane func ##################################################//

// 추정된 차선 계수로 결과 영상에 선을 그린다(직선)
void DrawDrivingLane(cv::Mat& st_ResultImage, const LANE_COEFFICIENT st_LaneCoef, cv::Scalar st_Color)
{
    if (std::abs(st_LaneCoef.f64_Slope) < 1e-6)
        return; // 혹은 x = const 직선 방식으로 따로 처리

    int rows = st_ResultImage.rows;
    int32_t x0 = int(-st_LaneCoef.f64_Intercept / st_LaneCoef.f64_Slope);
    int32_t x1 = int(((rows - 1) - st_LaneCoef.f64_Intercept) / st_LaneCoef.f64_Slope);

    cv::line(st_ResultImage,
             cv::Point(x0, rows - 1),
             cv::Point(x1, 0),
             st_Color, 2);
}

//###################################### FitModel func ##################################################//

// 두 포인트를 이용해 직선 모델을 구성
LANE_COEFFICIENT FitModel(const Point& st_Point1, const Point& st_Point2, bool& b_Flag)
{
    LANE_COEFFICIENT st_TmpModel;
    
    if((st_Point2.x != st_Point1.x))
    {
        // 기존 --> 정수로 떨어짐
        // st_TmpModel.f64_Slope = float((st_Point2.y - st_Point1.y) / (st_Point2.x - st_Point1.x));
        // st_TmpModel.f64_Intercept = int32_t(st_Point1.y - st_TmpModel.f64_Slope * st_Point1.x);

        // 수정 --> 실수로 계산 
        st_TmpModel.f64_Slope = 
            static_cast<double>(st_Point2.y - st_Point1.y) / static_cast<double>(st_Point2.x - st_Point1.x);

        st_TmpModel.f64_Intercept = 
            static_cast<double>(st_Point1.y) - st_TmpModel.f64_Slope * st_Point1.x;
    }
    else 
        b_Flag = false;

    return st_TmpModel;
}

//###################################### CalculateLaneCoefficient func ##################################################//

// 슬라이딩 윈도우로 수집한 점들에서 RANSAC으로 차선 계수를 산출
void CalculateLaneCoefficient(CAMERA_LANEINFO& st_LaneInfo, int32_t s32_Iteration, int64_t s64_Threshold)
{
    // 양쪽 차선 데이터 중 중앙선에 가장 가까운 안촉 차선 기준으로 RANSAC을 활용한 기울기 계산
    // srand(time(0)); // 난수 초기화
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

//###################################### FindTop5MaxIndices func ##################################################//

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

//###################################### FindClosestToMidPoint func ##################################################//

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

//###################################### FindLaneStartPositions func ##################################################//

// 히스토그램 분석으로 좌·우 슬라이딩 윈도 시작 위치를 계산 
void FindLaneStartPositions(const cv::Mat& st_Edge, int32_t& s32_WindowCentorLeft, int32_t& s32_WindowCentorRight, bool& b_NoLaneLeft, bool& b_NoLaneRight) 
{

    int32_t s32_col, s32_row, s32_I; //col : y , row : x

    // Histogram 계산
    int32_t* ps32_Histogram = new int32_t[st_Edge.cols](); // 동적 할당, cols는 가로방향 픽셀 개수 만큼 배열 생성 , 모두 0으로 초기화 ; cols가 x
    int32_t img_mid = st_Edge.cols / 2;

    // 이미지 하단 30프로  열에 해당하는 행 데이터들을 각 열별로 다 더한 후 최대가 되는 x좌표(행) 추출 --> height가 700이상이여야 작동한다. 
    for (s32_row = 0; s32_row < st_Edge.cols; ++s32_row) {
        for (s32_col = st_Edge.rows*0.7 ; s32_col < st_Edge.rows; ++s32_col) {
            ps32_Histogram[s32_row] += st_Edge.at<uchar>(s32_col, s32_row) > 0 ? 1 : 0; //검정색이 아닌 픽셀을 카운트 
        }
    }

    int32_t ars32_LeftCandidate[5];
    int32_t ars32_RightCandidate[5];

     // 왼쪽 / 오른쪽에서 Top-5 피크 후보 얻기
    FindTop5MaxIndices(ps32_Histogram, img_mid, ars32_LeftCandidate, b_NoLaneLeft);

    // 오른쪽은 히스토그램 포인터를 가운데부터로 옮긴 후 top-5 찾고, 나중에 x 인덱스 보정
    FindTop5MaxIndices(
        ps32_Histogram + img_mid,
        st_Edge.cols - img_mid,
        ars32_RightCandidate,
        b_NoLaneRight);

    // 오른쪽 후보 인덱스 보정 (0 기반 → 전체 이미지 기준으로 옮김)
    for (s32_I = 0; s32_I < 5; ++s32_I) {
        if (ars32_RightCandidate[s32_I] != -1) {
            ars32_RightCandidate[s32_I] += img_mid;
        }
    }

    // ====================== 이전 히스토그램이 없는 첫 프레임 ======================
    if (!b_IsprevHistogram)
    {
        // 왼쪽: 단순히 이미지 중앙 기준으로 가장 가까운 피크 선택
        if (!b_NoLaneLeft) {
            s32_WindowCentorLeft = FindClosestToMidPoint(ars32_LeftCandidate, img_mid);
            g_PrevWindowCenterLeft = s32_WindowCentorLeft;
        }

        // 오른쪽: 마찬가지
        if (!b_NoLaneRight) {
            s32_WindowCentorRight = FindClosestToMidPoint(ars32_RightCandidate, img_mid);
            g_PrevWindowCenterRight = s32_WindowCentorRight;
        }

        b_IsprevHistogram = true;
        delete[] ps32_Histogram;
        return;
    }

    // ====================== 이전 히스토그램이 있는 경우 ======================
    // 프레임 간 허용 이동량, 너무 약한 피크 필터링용 threshold
    const int32_t MAX_SHIFT   = 80;   // 프레임 간 최대 이동 허용(px)
    const int32_t MIN_HISTVAL = 15;   // 이 값보다 작은 피크는 노이즈로 간주

    // ---- 왼쪽 차선 시작점 선택 ----
    if (!b_NoLaneLeft)
    {
        double best_score = std::numeric_limits<double>::max();
        int32_t best_idx  = -1;

        for (int i = 0; i < 5; ++i)
        {
            int idx = ars32_LeftCandidate[i];
            if (idx < 0) continue;

            int hval = ps32_Histogram[idx];
            if (hval < MIN_HISTVAL) continue;  // 너무 약한 피크는 스킵

            if (g_PrevWindowCenterLeft >= 0 &&
                std::abs(idx - g_PrevWindowCenterLeft) > MAX_SHIFT)
            {
                // 이전 프레임 위치에서 너무 멀리 튀면 노이즈로 봄
                continue;
            }

            double d_prev   = (g_PrevWindowCenterLeft >= 0)
                                ? std::abs(idx - g_PrevWindowCenterLeft) : 0.0;
            double d_center = std::abs(idx - img_mid);
            double score    = 0.7 * d_prev + 0.3 * d_center; // 가중합

            if (score < best_score)
            {
                best_score = score;
                best_idx   = idx;
            }
        }

        if (best_idx >= 0) {
            s32_WindowCentorLeft  = best_idx;
            g_PrevWindowCenterLeft = best_idx;
        } else {
            // 이번 프레임에는 믿을 만한 왼쪽 차선 피크가 없다
            b_NoLaneLeft = true;
        }
    }

    // ---- 오른쪽 차선 시작점 선택 ----
    if (!b_NoLaneRight)
    {
        double best_score = std::numeric_limits<double>::max();
        int32_t best_idx  = -1;

        for (int i = 0; i < 5; ++i)
        {
            int idx = ars32_RightCandidate[i];
            if (idx < 0) continue;

            int hval = ps32_Histogram[idx];
            if (hval < MIN_HISTVAL) continue;

            if (g_PrevWindowCenterRight >= 0 &&
                std::abs(idx - g_PrevWindowCenterRight) > MAX_SHIFT)
            {
                continue;
            }

            double d_prev   = (g_PrevWindowCenterRight >= 0)
                                ? std::abs(idx - g_PrevWindowCenterRight) : 0.0;
            double d_center = std::abs(idx - img_mid);
            double score    = 0.7 * d_prev + 0.3 * d_center;

            if (score < best_score)
            {
                best_score = score;
                best_idx   = idx;
            }
        }

        if (best_idx >= 0) {
            s32_WindowCentorRight  = best_idx;
            g_PrevWindowCenterRight = best_idx;
        } else {
            b_NoLaneRight = true;
        }
    }

    delete[] ps32_Histogram;
}

//###################################### Parameter loader ##################################################//

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
