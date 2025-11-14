

// RAW 카메라 버퍼에서 차선 정보까지 계산하는 메인 파이프라인
void ImgProcessing(const cv::Mat& img_frame, CAMERA_DATA* camera_data)
{

    // kalman fileter variables
    CAMERA_LANEINFO st_LaneInfoLeft, st_LaneInfoRight; // sliding window에서 검출된 차선 정보 담는 구조체 
    st_LaneInfoLeft.b_IsLeft = true; // 왼쪽 차선 표시 --> 칼만 객체 생성에서 구분 
    KALMAN_STATE st_KalmanStateLeft, st_KalmanStateRight; // 새로 계산된 좌우 차선 거리 , 각도 저장 
    int32_t s32_I, s32_J, s32_KalmanStateCnt = 0;
    KALMAN_STATE arst_KalmanState[2] = {0};

    //setting mat 

    // 1) IPM 결과 버퍼 준비 (카메라 해상도가 바뀔 수 있으니 create 사용)
    g_IpmImg.create(
        camera_data->st_CameraParameter.s32_RemapHeight,
        camera_data->st_CameraParameter.s32_RemapWidth,
        img_frame.type()          // 보통 CV_8UC3
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

    // process pipe line 

    // (A) 원본 → IPM (탑뷰)
    cv::remap(img_frame, g_IpmImg, st_IPMX, st_IPMY,
             cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	// 그냥 원본을 바로 사용 --> IPM error test OK
    // g_IpmImg = img_frame.clone();
    // (B) IPM → Gray
    cv::cvtColor(g_IpmImg, g_TempImg, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(g_TempImg, g_TempImg, cv::Size(3,3), 0);  // 1x1은 사실상 no-op

    //########################cmh codes for lane detection v2 ########################//

    // 이미지 전처리 진행할 임시 이미지 버퍼 생성 Camera.yaml의 remap size 참고
    cv::Mat Temp_Img(camera_data->st_CameraParameter.s32_RemapHeight, camera_data->st_CameraParameter.s32_RemapWidth, CV_8UC1);
    cv::Mat st_NoneZero,st_Tmp, st_ResultImage(Temp_Img.size(), CV_8UC3, Scalar(0,0,0)); //결과랑 중간계산 mat, 최종시각화 검정색으로 초기화 
    // std::cout << img_frame.size() <<std::endl; // check origin img size (1280*720)

    // // 1) 콜백 img_frame 얕은 복사  
    cv::Mat ori_img = img_frame; // 원본 이미지 자체를 같은 버퍼로 사용 

    // // 2) IPM 맵을 이용해 860*640 영역으로 리맵 (맵 자체는 LoadMappingParam에서 로드) 
    cv::remap(img_frame,g_IpmImg,st_IPMX, st_IPMY, cv::INTER_NEAREST, cv::BORDER_CONSTANT);

    // // 3) 전처리: 그레이 변환 + 블러 (필요 시 커널 크기/유형 변경)
    cv::cvtColor(ori_img, Temp_Img, COLOR_BGR2GRAY);
    cv::GaussianBlur(Temp_Img, Temp_Img, Size(1,1), 0);

    // 4) 이진화 + 팽창 + 캐니 엣지로 차선 후보 강조 (threshold, Canny 범위가 튜닝 포인트)
    cv::Mat st_K = cv::getStructuringElement(MORPH_RECT, Size(5, 5));
    cv::threshold(Temp_Img,Temp_Img,170,255,cv::THRESH_BINARY);   // 170 보다 크면 255 아니면 0
    cv::dilate(Temp_Img,Temp_Img,st_K);
    cv::Canny(Temp_Img, Temp_Img, 100, 360);

    // 5) 슬라이딩 윈도우 준비: 0이 아닌 픽셀 좌표 추출
    cv::findNonZero(Temp_Img,st_Tmp);    // 0이 아닌 Pixel 추출

    // _1. 히스토그램으로 좌/우 차선의 시작 x 위치 계산 (FindLaneStartPositions 내부 로직 수정 가능)
    cv::Mat HalfImage = Temp_Img(cv::Range(700, Temp_Img.rows), cv::Range(0, Temp_Img.cols));
    double totalSum = cv::sum(HalfImage)[0]; // 그레이스케일 이미지의 경우
    int32_t s32_WindowCentorLeft, s32_WindowCentorRight;
    FindLaneStartPositions(Temp_Img, s32_WindowCentorLeft, s32_WindowCentorRight, b_NoLaneLeft, b_NoLaneRight);

    // _2. 슬라이딩 윈도우로 좌/우 차선 포인트 추출
    SlidingWindow(Temp_Img, st_Tmp, st_LaneInfoLeft, st_LaneInfoRight, s32_WindowCentorLeft, s32_WindowCentorRight, st_ResultImage);
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
    if(visualize)
    {
        cv::imshow("IPM", g_IpmImg);
        cv::imshow("Temp_Img", g_TempImg);
        cv::imshow("st_ResultImage", g_ResultImage);
        cv::waitKey(1);
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////
}
