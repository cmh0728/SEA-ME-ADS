#include "global.hpp"

extern float64_t f64_Vx, f64_Vy, f64_Ax, f64_Ay, f64_c_V, f64_c_A;

float64_t c_ORIGIN_LATITUDE_DEG = 0.0;
float64_t c_ORIGIN_LONGITUDE_DEG = 0.0;
float64_t c_ORIGIN_LATITUDE_RAD = 0.0;
float64_t c_ORIGIN_LONGITUDE_RAD = 0.0;
float64_t c_ORIGIN_ALTITUDE = 0.0;
float64_t c_ORIGIN_REFERENCE_X = 0.0;
float64_t c_ORIGIN_REFERENCE_Y = 0.0;
float64_t c_ORIGIN_REFERENCE_Z = 0.0;
int32_t s32_FileNum = 0;


// extern int32_t s32_Obj_Cnt;
// extern vector<vector<float64_t>> st_Lidar_XYZ;
// extern vector<vector<float64_t>> st_Lidar_ENU;
// extern st_Ego_ENU;
float64_t DegtoRad()
{
    return M_PI/180;
}

float64_t RadtoDeg()
{
    return 180/M_PI;
}


uint64_t getMillisecond()
{
    auto st_Now = std::chrono::steady_clock::now();
    auto st_Now_ms = time_point_cast<std::chrono::milliseconds>(st_Now);
    milliseconds st_Millisecond = duration_cast<std::chrono::milliseconds>(st_Now_ms.time_since_epoch());
    return st_Millisecond.count();
}

float32_t deg2rad(float32_t f32_Degree)
{
    return f32_Degree * M_PI / 180.f;
}


float32_t rad2deg(float32_t f32_Radian)
{
    return f32_Radian * 180.f / M_PI;
}


float64_t deg2rad(float64_t f64_Degree)
{
    return f64_Degree * M_PI / 180.f;
}


float64_t rad2deg(float64_t f64_Radian)
{
    return f64_Radian * 180.f / M_PI;
}

float32_t ms2kph(float32_t f32_Speed)
{
    return f32_Speed * 3.6f;
}

float32_t kph2ms(float32_t f32_Speed)
{
    return f32_Speed / 3.6f;
}

float64_t ms2kph(float64_t f64_Speed)
{
    return f64_Speed * 3.6;
}

float64_t kph2ms(float64_t f64_Speed)
{
    return f64_Speed / 3.6;
}

float32_t getDistance3d(float32_t f32_X1, float32_t f32_Y1, float32_t f32_Z1, float32_t f32_X2, float32_t f32_Y2, float32_t f32_Z2)
{
    return sqrtf(powf(f32_X1 - f32_X2, 2.f) + powf(f32_Y1 - f32_Y2, 2.f) + powf(f32_Z1 - f32_Z2, 2.f));
}

float64_t getDistance3d(float64_t f64_X1, float64_t f64_Y1, float64_t f64_Z1, float64_t f64_X2, float64_t f64_Y2, float64_t f64_Z2)
{
    return sqrtf(powf(f64_X1 - f64_X2, 2.f) + powf(f64_Y1 - f64_Y2, 2.f) + powf(f64_Z1 - f64_Z2, 2.f));
}


float32_t getDistance2d(float32_t f32_X1, float32_t f32_Y1, float32_t f32_X2, float32_t f32_Y2)
{
    return sqrtf(powf(f32_X1 - f32_X2, 2.f) + powf(f32_Y1 - f32_Y2, 2.f));
}


float64_t getDistance2d(float64_t f64_X1, float64_t f64_Y1, float64_t f64_X2, float64_t f64_Y2)
{
    return sqrtf(powf(f64_X1 - f64_X2, 2.f) + powf(f64_Y1 - f64_Y2, 2.f));
}

float32_t pi2pi(float32_t f32_Angle)
{
    if (f32_Angle > M_PI)
        f32_Angle -= 2 * M_PI;
    else if(f32_Angle < -M_PI)
        f32_Angle += 2 * M_PI;

    return f32_Angle;
}

float64_t pi2pi(float64_t f64_Angle)
{
    if (f64_Angle > M_PI)
        f64_Angle -= 2 * M_PI;
    else if(f64_Angle < -M_PI)
        f64_Angle += 2 * M_PI;

    return f64_Angle;
}

void GetModData(float32_t min, float32_t max, float32_t &data)
{
	if (data > max)
	{
		data = max;
	}
	else if (data < min)
	{
		data = min;
	}
}

void GetModData(float64_t min, float64_t max, float64_t &data)
{
	if (data > max)
	{
		data = max;
	}
	else if (data < min)
	{
		data = min;
	}
}




void CalcNearestDistIdx(float64_t f64_X, float64_t f64_Y, float64_t* f64_MapX, float64_t* f64_MapY, int32_t s32_MapLength, float64_t& f64_NearDist, int32_t& s32_NearIdx)
{
    float64_t f64_MinDist = 999999999.f;
    int32_t s32_I;
    for(s32_I = 0; s32_I < s32_MapLength; s32_I++)
    {
        float64_t f64_Dist = getDistance2d(f64_X, f64_Y, f64_MapX[s32_I], f64_MapY[s32_I]);
        if(f64_MinDist > f64_Dist)
        {
            f64_MinDist = f64_Dist; 
            f64_NearDist = f64_Dist; 
            s32_NearIdx = s32_I; 
        }
    }
}


void CalcNearestDistIdx(float32_t f32_X, float32_t f32_Y, float32_t* f32_MapX, float32_t* f32_MapY, int32_t s32_MapLength, float32_t& f32_NearDist, int32_t& s32_NearIdx)
{
    float32_t f32_MinDist = 999999999.f;
    int32_t s32_I;
    for(s32_I = 0; s32_I < s32_MapLength; s32_I++)
    {
        float32_t f32_Dist = getDistance2d(f32_X, f32_Y, f32_MapX[s32_I], f32_MapY[s32_I]);
        if(f32_MinDist > f32_Dist)
        {
            f32_MinDist = f32_Dist; 
            f32_NearDist = f32_Dist; 
            s32_NearIdx = s32_I; 
        }
    }
}
void QUINTIC_t::Init(float32_t f32_tXS, float32_t f32_tVXS, float32_t f32_tAXS, float32_t f32_tXE, float32_t f32_tVXE, float32_t f32_tAXE, float32_t f32_T)
{
    f32_XS = f32_tXS;
    f32_VXS = f32_tVXS;
    f32_AXS = f32_tAXS;
    f32_XE = f32_tXE;
    f32_VXE = f32_tVXE;
    f32_AXE = f32_tAXE;

    f32_A0 = f32_XS;
    f32_A1 = f32_VXS;
    f32_A2 = f32_AXS / 2.0f;

    Eigen::MatrixXf st_A(3, 3);
    Eigen::MatrixXf st_B(3, 1);
    Eigen::MatrixXf st_X(3, 1);

    st_A << powf(f32_T, 3.f),           powf(f32_T, 4.f),               powf(f32_T, 5.f),
            3.f * powf(f32_T, 2.f),     4.f * powf(f32_T, 3.f),         5.f * powf(f32_T, 4.f),
            6.f * f32_T,                12.f * powf(f32_T, 2.f),        20 * powf(f32_T, 3.f);

    st_B << f32_XE - f32_A0 - f32_A1 * f32_T - f32_A2 * f32_T * f32_T,
            f32_VXE - f32_A1 - 2.f * f32_A2 * f32_T,
            f32_AXE - 2.f * f32_A2;

    st_X = st_A.inverse() * st_B;

    f32_A3 = st_X(0, 0);
    f32_A4 = st_X(1, 0);
    f32_A5 = st_X(2, 0);
}

float32_t QUINTIC_t::CalcPoint(float32_t f32_T)
{
    float32_t f32_XT = f32_A0 + 
                       f32_A1 * f32_T + 
                       f32_A2 * powf(f32_T, 2.f) + 
                       f32_A3 * powf(f32_T, 3.f) + 
                       f32_A4 * powf(f32_T, 4.f) + 
                       f32_A5 * powf(f32_T, 5.f) ;
    return f32_XT;
}

float32_t QUINTIC_t::CalcFirstDerivative(float32_t f32_T)
{
    float32_t f32_XT = f32_A1 + 
                       2.f * f32_A2 * f32_T + 
                       3.f * f32_A3 * powf(f32_T, 2.f) + 
                       4.f * f32_A4 * powf(f32_T, 3.f) + 
                       5.f * f32_A5 * powf(f32_T, 4.f);
    return f32_XT;
}

float32_t QUINTIC_t::CalcSecondDerivative(float32_t f32_T)
{
    float32_t f32_XT = 2.f * f32_A2 + 
                       6.f * f32_A3 * f32_T + 
                       12.f * f32_A4 * powf(f32_T, 2.f) + 
                       20.f * f32_A5 * powf(f32_T, 3.f);
    return f32_XT;
}

float32_t QUINTIC_t::CalcThirdDerivative(float32_t f32_T)
{
    float32_t f32_XT = 6.f * f32_A3 + 
                       24.f * f32_A4 * f32_T + 
                       60.f * f32_A5 * powf(f32_T, 2.f);
    return f32_XT;
}




void QUARTIC_t::Init(float32_t f32_tXS, float32_t f32_tVXS, float32_t f32_tAXS, float32_t f32_tVXE, float32_t f32_tAXE, float32_t f32_T)
{
    f32_XS = f32_tXS;
    f32_VXS = f32_tVXS;
    f32_AXS = f32_tAXS;
    f32_VXE = f32_tVXE;
    f32_AXE = f32_tAXE;

    f32_A0 = f32_XS;
    f32_A1 = f32_VXS;
    f32_A2 = f32_AXS / 2.0f;

    Eigen::MatrixXf st_A(2, 2);
    Eigen::MatrixXf st_B(2, 1);
    Eigen::MatrixXf st_X(2, 1);

    st_A << 3.f * powf(f32_T, 2.f),         4.f * powf(f32_T, 3.f),
            6.f * f32_T,                    12.f * powf(f32_T, 2.f);

    st_B << f32_VXE - f32_A1 - 2.f * f32_A2 * f32_T,
         f32_AXE - 2.f * f32_A2;

    st_X = st_A.inverse() * st_B;

    f32_A3 = st_X(0, 0);
    f32_A4 = st_X(1, 0);
}

float32_t QUARTIC_t::CalcPoint(float32_t f32_T)
{
    float32_t f32_XT = f32_A0 + 
                       f32_A1 * f32_T + 
                       f32_A2 * powf(f32_T, 2.f) + 
                       f32_A3 * powf(f32_T, 3.f) + 
                       f32_A4 * powf(f32_T, 4.f);
    return f32_XT;
}

float32_t QUARTIC_t::CalcFirstDerivative(float32_t f32_T)
{
    float32_t f32_XT = f32_A1 + 
                       2.f * f32_A2 * f32_T + 
                       3.f * f32_A3 * powf(f32_T, 2.f) + 
                       4.f * f32_A4 * powf(f32_T, 3.f);
    return f32_XT;
}

float32_t QUARTIC_t::CalcSecondDerivative(float32_t f32_T)
{
    float32_t f32_XT = 2.f * f32_A2 + 
                       6.f * f32_A3 * f32_T + 
                       12.f * f32_A4 * powf(f32_T, 2.f);
    return f32_XT;
}

float32_t QUARTIC_t::CalcThirdDerivative(float32_t f32_T)
{
    float32_t f32_XT = 6.f * f32_A3 + 
                       24.f * f32_A4 * f32_T;
    return f32_XT;
}



int32_t SPLINE_PATH_t::GetNearestIndex(float32_t f32_X, float32_t f32_Y)
{

    int32_t s32_I = 0;
    int32_t s32_Result = -1;
    float32_t f32_MinDistance = 9999999.f;
    float32_t f32_Distance = 0.;

    for (s32_I = 0; s32_I < s32_Num; s32_I++)
    {
        f32_Distance = getDistance2d(arf32_X[s32_I], arf32_Y[s32_I], f32_X, f32_Y);
        if (f32_MinDistance > f32_Distance)
        {
            f32_MinDistance = f32_Distance;
            s32_Result = s32_I;
        }
    }

    s32_NearestIndex = s32_Result;
    return s32_Result;
}



void SPLINE_PARAM_t::Init(float32_t *pf32_X, float32_t *pf32_Y, int32_t s32_Num)
{
    int32_t s32_I;

    float32_t arf32_H[c_PLANNING_MAX_PATH_NUM] = { 0 };

    for (s32_I = 0;s32_I < s32_Num;s32_I++)
    {
        arf32_X[s32_I] = pf32_X[s32_I];
        arf32_Y[s32_I] = pf32_Y[s32_I];
        arf32_A[s32_I] = pf32_Y[s32_I];

        if (s32_I > 0)
        {
            arf32_H[s32_I - 1] = arf32_X[s32_I] - arf32_X[s32_I - 1];  
        }
    }

    s32_NX = s32_Num;

    //Ax = b
    // => AC = B
    // => C = A^-1 * B

    Eigen::MatrixXd st_A = CalcA(arf32_H);     
    Eigen::MatrixXd st_B = CalcB(arf32_H);
    Eigen::MatrixXd st_C = st_A.inverse() * st_B;

    for(s32_I = 0; s32_I < st_C.rows(); s32_I++)
    {
        arf32_C[s32_I] = st_C(s32_I,0);
    }   

    for(s32_I = 0; s32_I < s32_Num - 1;s32_I++)
    {
        arf32_D[s32_I] = ((arf32_C[s32_I + 1] - arf32_C[s32_I]) / (3.0 * arf32_H[s32_I]));
        arf32_B[s32_I] = ((arf32_A[s32_I + 1] - arf32_A[s32_I]) / arf32_H[s32_I]) 
                        - (arf32_H[s32_I] * (arf32_C[s32_I + 1])
                        + ((2.0 * arf32_C[s32_I])) / 3.0);
    }   
}

int32_t SPLINE_PARAM_t::SearchNearestIndex(float32_t f32_P)
{
    int32_t s32_I;
    float32_t f32_X;
    int32_t s32_Result = 0;
    float32_t f32_MinDistance = 999999.f;
    float32_t f32_Distance = 0.f;

    for (s32_I = 0; s32_I < s32_NX;s32_I++)
    {
        f32_X = (float32_t)arf32_X[s32_I];
        f32_Distance = fabsf(f32_X - f32_P);

        if (f32_Distance < f32_MinDistance)
        {
            f32_MinDistance = f32_Distance;
            s32_Result = s32_I;
        }
    }

    return (s32_Result + s32_NX - 1) % s32_NX;
}


Eigen::MatrixXd SPLINE_PARAM_t::CalcA(float32_t* pf32_H)
{

    int32_t s32_I;

    Eigen::MatrixXd st_A = Eigen::MatrixXd::Zero(s32_NX, s32_NX);
    st_A(0, 0) = 1.0;

    for (s32_I = 0; s32_I < s32_NX - 1; s32_I++)
    {
        if(s32_I != s32_NX - 2)
        {
            st_A(s32_I + 1, s32_I + 1) = 2.0 * (pf32_H[s32_I] + pf32_H[s32_I + 1]);
        }
        st_A(s32_I + 1, s32_I) = pf32_H[s32_I]; 
        st_A(s32_I, s32_I + 1) = pf32_H[s32_I]; 
    }

    st_A(0, 1) = 0.0;
    st_A(s32_NX - 1, s32_NX - 2) = 0.0;
    st_A(s32_NX - 1, s32_NX - 1) = 1.0;

    return st_A;    
}


Eigen::MatrixXd SPLINE_PARAM_t::CalcB(float32_t* pf32_H)
{
    int32_t s32_I;

    Eigen::MatrixXd st_B = Eigen::MatrixXd::Zero(s32_NX, 1);

    for (s32_I = 0; s32_I < s32_NX - 2 ; s32_I++)
    {
        st_B(s32_I + 1, 0) = (3.0 * (arf32_A[s32_I + 2] - arf32_A[s32_I + 1]) / pf32_H[s32_I + 1]) 
                                - (3.0 * (arf32_A[s32_I + 1] - arf32_A[s32_I]) / pf32_H[s32_I]);
    }

    return st_B;
}

float32_t SPLINE_PARAM_t::Calc(float32_t f32_T)
{

    int32_t s32_NearestIndex;
    float32_t f32_DX = 0;
    float32_t f32_Result = 0;

    if(s32_NX == 0)
    {
        return NONE;
    }
    else if(f32_T < arf32_X[0])
    {
        return NONE;
    }
    else if(f32_T > arf32_X[s32_NX - 1])
    {
        return NONE;
    }

    s32_NearestIndex = SearchNearestIndex(f32_T);
    f32_DX = f32_T - arf32_X[s32_NearestIndex];

    f32_Result = arf32_A[s32_NearestIndex] + 
                arf32_B[s32_NearestIndex] * f32_DX + 
                arf32_C[s32_NearestIndex] * f32_DX * f32_DX + 
                arf32_D[s32_NearestIndex] * f32_DX * f32_DX * f32_DX;

    return f32_Result;   
}

float32_t SPLINE_PARAM_t::CalcD(float32_t f32_T)
{
    int32_t s32_NearestIndex;
    float32_t f32_DX = 0;
    float32_t f32_Result = 0;

    if(f32_T < arf32_X[0])
    {
        return NONE;
    }
    else if(f32_T > arf32_X[s32_NX - 1] )
    {
        return NONE;
    }

    s32_NearestIndex = SearchNearestIndex(f32_T);
    f32_DX = f32_T - arf32_X[s32_NearestIndex];

    f32_Result =  arf32_B[s32_NearestIndex] + 
                 2 * arf32_C[s32_NearestIndex] * f32_DX +
                 3 * arf32_D[s32_NearestIndex] * f32_DX * f32_DX;

    return f32_Result;
}

float32_t SPLINE_PARAM_t::CalcDD(float32_t f32_T)
{
    int32_t s32_NearestIndex;
    float32_t f32_DX = 0;
    float32_t f32_Result = 0;

    if(f32_T < arf32_X[0])
    {
        return NONE;
    }
    else if(f32_T > arf32_X[s32_NX - 1] )
    {
        return NONE;
    }

    s32_NearestIndex = SearchNearestIndex(f32_T);
    f32_DX = f32_T - arf32_X[s32_NearestIndex];

    f32_Result = 2 * arf32_C[s32_NearestIndex] + 
                 6 * arf32_D[s32_NearestIndex] * f32_DX;

    return f32_Result; 
}

void SPLINE2D_PARAM_t::Init(PATH_t *pst_Path)
{
    memcpy(arf32_X, pst_Path->arf32_X, pst_Path->s32_Num * sizeof(float32_t));
    memcpy(arf32_Y, pst_Path->arf32_Y, pst_Path->s32_Num * sizeof(float32_t));
    s32_NX = pst_Path->s32_Num;

    CalcS();

    st_SplineX.Init(arf32_S, arf32_X, s32_NX);
    st_SplineY.Init(arf32_S, arf32_Y, s32_NX);

}

void SPLINE2D_PARAM_t::CalcS()
{
    int32_t s32_I = 0;
    float32_t f32_DX = 0.f;
    float32_t f32_DY = 0.f;
    float32_t f32_Distance = 0.f;

    arf32_S[0] = 0;

    for (s32_I = 1;s32_I < s32_NX;s32_I++)
    {
        f32_DX = arf32_X[s32_I] - arf32_X[s32_I - 1];
        f32_DY = arf32_Y[s32_I] - arf32_Y[s32_I - 1];

        f32_Distance = getDistance2d(0, 0, f32_DX, f32_DY);

        arf32_DS[s32_I - 1] = f32_Distance;
        arf32_S[s32_I] = arf32_S[s32_I - 1] + f32_Distance;
    }
}

void SPLINE2D_PARAM_t::CalcPosition(float32_t f32_T, float32_t &f32_X, float32_t &f32_Y)
{
    f32_X = st_SplineX.Calc(f32_T);
    f32_Y = st_SplineY.Calc(f32_T);
}

float32_t SPLINE2D_PARAM_t::CalcCurvature(float32_t f32_T)
{
    float32_t f32_DX = st_SplineX.CalcD(f32_T);
    float32_t f32_DDX = st_SplineX.CalcDD(f32_T);

    float32_t f32_DY = st_SplineY.CalcD(f32_T);
    float32_t f32_DDY =st_SplineY.CalcDD(f32_T);

    float32_t f32_K = (f32_DDY * f32_DX - f32_DDX * f32_DY) / (f32_DX * f32_DX + f32_DY * f32_DY);

    return f32_K;
}

float32_t SPLINE2D_PARAM_t::CalcYawRad(float32_t f32_T)
{
    float32_t f32_DX = st_SplineX.CalcD(f32_T); 
    float32_t f32_DY = st_SplineY.CalcD(f32_T);

    if(f32_DX == 0)
    {
        return M_PI / 2 * (f32_DY > 0);
    } 

    return atan2(f32_DY, f32_DX);
}

float32_t SPLINE2D_PARAM_t::GetLastS()
{
    return arf32_S[s32_NX - 1];
}









void lla2enu(float64_t f64_Lat_deg, float64_t f64_Lon_deg, float64_t f64_Alt, float32_t &f32_E, float32_t &f32_N, float32_t &f32_U)
{
    float64_t f64_Lat_rad = deg2rad(f64_Lat_deg);
    float64_t f64_Lon_rad = deg2rad(f64_Lon_deg);

    float64_t f64_Chi = sqrt(1 - c_LLA2ENU_N_2 * pow(sin(f64_Lat_rad), 2));
    float64_t f64_Q = (c_LLA2ENU_A / f64_Chi + f64_Alt) * cos(f64_Lat_rad);

    float64_t f64_X = f64_Q * cos(f64_Lon_rad);
    float64_t f64_Y = f64_Q * sin(f64_Lon_rad);
    float64_t f64_Z = ((c_LLA2ENU_A * (1 - c_LLA2ENU_N_2) / f64_Chi) + f64_Alt) * sin(f64_Lat_rad);

    float64_t f64_dX = f64_X - c_ORIGIN_REFERENCE_X;
    float64_t f64_dY = f64_Y - c_ORIGIN_REFERENCE_Y;
    float64_t f64_dZ = f64_Z - c_ORIGIN_REFERENCE_Z;

    f32_E = (float32_t)(-sin(c_ORIGIN_LONGITUDE_RAD) * f64_dX + cos(c_ORIGIN_LONGITUDE_RAD) * f64_dY);
    f32_N = (float32_t)(-sin(c_ORIGIN_LATITUDE_RAD) * cos(c_ORIGIN_LONGITUDE_RAD) * f64_dX - sin(c_ORIGIN_LATITUDE_RAD) * sin(c_ORIGIN_LONGITUDE_RAD) * f64_dY + cos(c_ORIGIN_LATITUDE_RAD) * f64_dZ);
    f32_U = (float32_t)(cos(c_ORIGIN_LATITUDE_RAD) * cos(c_ORIGIN_LONGITUDE_RAD) * f64_dX + cos(c_ORIGIN_LATITUDE_RAD) * sin(c_ORIGIN_LONGITUDE_RAD) * f64_dY + sin(c_ORIGIN_LATITUDE_RAD) * f64_dZ);
}

void CreateBagFileName(char* datetime)
{
    time_t timer;
    struct tm* t;
    timer = time(NULL);
    t = localtime(&timer);
    sprintf(datetime, "%d_%02d%02d_%d.bin", 
            t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,s32_FileNum);

}
int extractNumberFromFilename(const std::string& filename) {
    size_t pos = filename.find_last_of('_'); // '_' 문자를 기준으로 분리
    if (pos != std::string::npos) {
        try {
            return std::stoi(filename.substr(pos + 1)); // 분리된 숫자 부분을 정수로 변환하여 반환
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid number format in filename: " << filename << std::endl;
        }
    }
    return 0; // 숫자를 추출할 수 없을 경우 0을 반환
}
bool CompareFilenames(const std::string& filename1, const std::string& filename2) {
    int number1 = extractNumberFromFilename(filename1);
    int number2 = extractNumberFromFilename(filename2);
    return number1 < number2;
}
void rotationMatrix(float32_t f32_Roll, float32_t f32_Pitch, float32_t f32_Yaw, float32_t f32_matrix[3][3]) {
    float32_t f32_cos_roll = cos(deg2rad(f32_Roll));
    float32_t f32_sin_roll = sin(deg2rad(f32_Roll));
    float32_t f32_cos_pitch = cos(deg2rad(f32_Pitch));
    float32_t f32_sin_pitch = sin(deg2rad(f32_Pitch));
    float32_t f32_cos_yaw = cos(deg2rad(f32_Yaw));
    float32_t f32_sin_yaw = sin(deg2rad(f32_Yaw));

    f32_matrix[0][0] = f32_cos_yaw * f32_cos_pitch;
    f32_matrix[0][1] = f32_cos_yaw * f32_sin_pitch * f32_sin_roll - f32_sin_yaw * f32_cos_roll;
    f32_matrix[0][2] = f32_cos_yaw * f32_sin_pitch * f32_cos_roll + f32_sin_yaw * f32_sin_roll;
    f32_matrix[1][0] = f32_sin_yaw * f32_cos_pitch;
    f32_matrix[1][1] = f32_sin_yaw * f32_sin_pitch * f32_sin_roll + f32_cos_yaw * f32_cos_roll;
    f32_matrix[1][2] = f32_sin_yaw * f32_sin_pitch * f32_cos_roll - f32_cos_yaw * f32_sin_roll;
    f32_matrix[2][0] = -f32_sin_pitch;
    f32_matrix[2][1] = f32_cos_pitch * f32_sin_roll;
    f32_matrix[2][2] = f32_cos_pitch * f32_cos_roll;
}

// 3차원 포인트 회전 함수
void rotatePoint(float32_t f32_X, float32_t f32_Y, float32_t f32_Z, float32_t f32_Roll, float32_t f32_Pitch, float32_t f32_Yaw, float32_t &f32_RX, float32_t &f32_RY, float32_t &f32_RZ) {
}
