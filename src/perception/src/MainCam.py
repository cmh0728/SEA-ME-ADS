#!/usr/bin/env python3
#-*'coding:utf-8-*-

# x:3, Y:0, Z:1
# 1020x720 , FOV:90

import rospy
import cv2
import numpy as np
import os, rospkg
import sys
import math
import time
# from sklearn.cluster import DBSCAN
from sympy import symbols, Eq, solve, sqrt

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridgeError
from Camera.msg import Lane,LanePnt
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from casadi import *

class KalmanObject:
    def __init__(self):

        self.st_Z = [[0 for j in range(4)] for i in range(4)]               # 측정값                1x4
        self.st_X = [[0 for j in range(4)] for i in range(4)]               # 추정값                1x4
        self.st_P = [[0 for j in range(4)] for i in range(4)]               # 오차 공분산            4x4
        self.st_A = [[0 for j in range(4)] for i in range(4)]               # 시스템 모델 행렬        4x4
        self.st_H = [[0 for j in range(4)] for i in range(4)]               # 상태 전이 행렬         4x4
        self.st_Q = [[0 for j in range(4)] for i in range(4)]               # 시스템 노이즈 행렬      4x4  
        self.st_R = [[0 for j in range(4)] for i in range(4)]               # 관측(센서) 노이즈 행렬  4x4  
        self.st_K = [[0 for j in range(4)] for i in range(4)]               # Kalman Gain         4x4
        # self.st_LastP = [[0 for j in range(4)] for i in range(4)]           # P(before)        4x4
        self.st_LastX = [[0 for j in range(4)] for i in range(4)]           # X(before)        4x4
        self.b_InitializeFlag = True
        self.b_FirstProcessFalg = True
        self.s32_DistThreshold = 10  # Pixel Coordinate -> World Dist == 2cm
        self.b_SameObject = False
        self.s32_Count = 0
        self.s64_CoefA = 0
        self.s64_CoefB = 0

    def UpdateObservation(self, st_z):
        self.st_Z = st_z

    def SetInitialX(self, st_z):
        self.st_X = st_z

    def CheckSameOBject(self, st_NewObservation):
        # st_NewObservation: st_X와 동일한 구조 -> dist를 우선적으로 비교
        if abs(self.st_X[0] - st_NewObservation[0]) > self.s32_DistThreshold:
            self.b_SameObject = True
        else:
            self.b_SameObject = False

    def InitializateKalmanParamter(self):
        if self.b_InitializeFlag:
            for i in range(4):
                self.st_P[i][i] = 0.01     # 임의의 초기값
                self.st_Q[i][i] = 2        # 실험적 설정
                self.st_R[i][i] = 2        # 실험적 설정

                self.st_A[i][i] = 1     # 시스템 모델
                self.st_H[i][i] = 1     # 상태전이 행렬 
            
            self.st_A[0][1] = 1
            self.st_A[2][3] = 1

            self.b_InitializeFlag = False

            self.st_Z = np.array(self.st_Z)
            self.st_X = np.array(self.st_X)
            self.st_P = np.array(self.st_P)
            self.st_A = np.array(self.st_A)
            self.st_H = np.array(self.st_H)
            self.st_Q = np.array(self.st_Q)
            self.st_R = np.array(self.st_R)
            self.st_K = np.array(self.st_K)
            # self.st_PredictionP = np.array(self.st_PredictionP)
            self.st_LastX = np.array(self.st_LastX)

    def PredictState(self):
        
        self.st_LastX = self.st_X

        # 추정값과 오차 공분산 예측
        self.st_X = self.st_A @ self.st_X
        self.st_P = self.st_A @ self.st_P @ np.transpose(self.st_A) + self.st_Q
        
        # 예측했으므로 Update
        self.UpdateObservation(self.st_X)

        # 칼만 게인 계산
        B = self.st_H @ self.st_P @ np.transpose(self.st_H) + self.st_R
        B = np.linalg.inv(B)
        B = np.array(B)
        self.st_K = self.st_P @ np.transpose(self.st_H) @ B

        if self.b_SameObject:
            self.EstimationState()


    def EstimationState(self):

        # 추정값 계산
        self.st_X = self.st_X + self.st_K @ (self.st_Z - self.st_H @ self.st_X)

        # 추정했으므로 Update
        self.UpdateObservation(self.st_X)

        # 오차 공분산 계산
        self.st_P = self.st_P - self.st_K @ self.st_H @ self.st_H




class IMGParser:
    def __init__(self):

        self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)
        self.publisher = rospy.Publisher("Camcom" ,Lane, queue_size=10)
        self.intrinsic = np.array(
                       [[510,0,510,0],
                        [0,510,360,0],
                        [0,0,1,0]])

        self.world_x_max = 40
        self.world_x_min = 1
        self.world_y_max = 6
        self.world_y_min = -6
        self.world_x_interval = 0.05
        self.world_y_interval = 0.02
        self.map_x = np.loadtxt('/home/autonav/catkin_ws/src/Camera/Param/ParamX.txt', delimiter='\t')   
        self.map_y = np.loadtxt('/home/autonav/catkin_ws/src/Camera/Param/ParamY.txt', delimiter='\t')
        self.Param_Margin_x = 35
        self.Param_Margin_y = 20
        self.Cnt_L = 0
        self.Cnt_R = 0
        self.Left_Befor = 0
        self.Right_Befor = 0
        self.Coef_L = None
        self.Coef_R = None
        self.Pnt_Last_L = None
        self.Pnt_Last_R = None
        self.Interp_L = []          # Left Y Pnt for Interpolating Right Lane Pnt
        self.Interp_R = []          # Right Y Pnt for Interpolating Left Lane Pnt
        self.Coef_Flag = True
        self.SolvedX = 0
        self.SolvedY = 0
        self.LastSolvedX = 0
        self.LastSolvedY = 0
        self.f64_MinDistance = 0
        self.f64_LastMinDistance = 0
        self.f64_Angle = 0
        self.f64_LastAngle = 0
        self.st_KalmanList = []     # Kalman Object List
        self.b_Kalman = False
        self.s64_StartTime = 0
        self.s64_EndTime = 0

    # def InitializeKalmanParameter(self):

    def make_zero(self,arr):
        arr[abs(arr)<=6.123234e-16] = 0
        return arr

    def translation_matrix(self, vector):
        M = np.identity(4)
        M[:3, 3] = vector[:3]
        return M

    def remap_nearest(self, src, map_x, map_y):
        src_height = src.shape[0]
        src_width = src.shape[1]
        
        dst_height = map_x.shape[0]
        dst_width = map_x.shape[1]
        dst = np.zeros((dst_height, dst_width, 3)).astype(np.uint8)
        for i in range(dst_height):
            for j in range(dst_width):
                src_y = int(np.round(map_y[i][j]))
                src_x = int(np.round(map_x[i][j]))
                if 0 <= src_y and src_y < src_height and 0 <= src_x and src_x < src_width:
                    dst[i][j] = src[src_y, src_x, :]
        return dst    

    def generate_direct_backward_mapping(self,
        world_x_min, world_x_max, world_x_interval, 
        world_y_min, world_y_max, world_y_interval, extrinsic, intrinsic):
        
        world_x_coords = np.arange(world_x_max, world_x_min, -world_x_interval)
        world_y_coords = np.arange(world_y_max, world_y_min, -world_y_interval)
        
        output_height = len(world_x_coords)
        output_width = len(world_y_coords)

        map_x = np.zeros((output_height, output_width)).astype(np.float32)
        map_y = np.zeros((output_height, output_width)).astype(np.float32)
        
        for i, world_x in enumerate(world_x_coords):
            for j, world_y in enumerate(world_y_coords):
                
                world_coord = [world_x, world_y, 0, 1]
                camera_coord = extrinsic[:3, :] @ world_coord
                uv_coord = intrinsic[:3, :3] @ camera_coord
                uv_coord /= uv_coord[2]

                map_x[i][j] = uv_coord[0]
                map_y[i][j] = uv_coord[1]
                
        return map_x, map_y
        
    def Final_Matrix(self,roll, pitch, yaw):
        
        Rotation_x = np.array([[     1    ,      0        ,        0       ],
                                [     0    , math.cos(roll), -math.sin(roll)],
                                [     0    , math.sin(roll),  math.cos(roll)]])
        
        Rotation_y = np.array([[ math.cos(pitch) ,  0  ,  math.sin(pitch)  ],
                                [        0        ,  1  ,        0          ],
                                [-math.sin(pitch) ,  0  ,  math.cos(pitch)  ]])
        
        Rotation_z = np.array([[math.cos(yaw), -math.sin(yaw) ,  0   ],
                                [math.sin(yaw),  math.cos(yaw) ,  0   ],
                                [     0       ,       0        ,  1   ]])
        

        R_P = np.matmul(np.matmul(Rotation_z,Rotation_y),Rotation_x)
        R_P = self.make_zero(R_P)
        R_P = np.array([[R_P[0][0],R_P[0][1],R_P[0][2],0],
                        [R_P[1][0],R_P[1][1],R_P[1][2],0],
                        [R_P[2][0],R_P[2][1],R_P[2][2],0],
                        [0,0,0,1]])

        Trans = self.translation_matrix((0.88,0.,-1.1))

        return R_P@Trans

    def Save_Mapping_To_txt(self, map_x, map_y, filename_x='IPMx_0228.txt', filename_y='IPMy_0228.txt'):
        np.savetxt(filename_x, map_x, fmt='%f', delimiter='\t')
        np.savetxt(filename_y, map_y, fmt='%f', delimiter='\t')

    def ShowHistogram(self, st_Histogram):
        plt.plot(st_Histogram)
        plt.title('Histogram')
        plt.xlabel('Pixel')
        plt.ylabel('Sum')
        plt.show()

    def extractLane(self, x, y, Downsampling_mask, Lane_mask):
        dx = [0, 0, -10, 10]
        dy = [10, -10, 0, 0]

        for i in range(4):
            X = x + dx[i]
            Y = y + dy[i]
            if(0<=X<800 and 0<=Y<580):
                if Downsampling_mask[Y][X] != 0:
                    Downsampling_mask[Y][X] = 0
                    Lane_mask[Y][X] = 255
                    self.extractLane(X,Y,Downsampling_mask,Lane_mask)

    def CalculateMinDistance(self, s64_a, s64_b):

        x, y = symbols('x y')
        f64_Slope = -1 / s64_a

        opti = Opti()

        casx = opti.variable(1)
        casy = opti.variable(1)

        cost = 0.0

        opti._subject_to(s64_a * casx - casy + s64_b == 0)
        opti._subject_to(casy >= 0)
    
        cost += fabs(casx ** 2 + casy ** 2)
        opti.minimize(cost)
        opti.solver("ipopt")

        sol = opti.solve()

        # World Coordinate : 위가 y, 오른쪽 x
        self.SolvedX = int(sol.value(casx))
        self.SolvedY = int(sol.value(casy))

        self.f64_MinDistance = sqrt(self.SolvedX**2 + self.SolvedY**2)
        self.f64_Angle = atan2(self.SolvedY , self.SolvedX)

        # 각도계산 atan2(y,x)

        # print("Original Equation\n y = ",s64_a,"*x + ",s64_b)
        # print("----------------------------------------------------------")
        # print("Perpendicular Equation\n y = ",f64_Slope,"*x + ",f64_Intercept)
        # print("----------------------------------------------------------")
        # print("Intersection Point:  (",self.SolvedX,",",self.SolvedY,")")    # Dictionary 형태
        # print("Min Distance:        ", self.f64_MinDistance)
        # print("f64_Angle:           ", self.f64_Angle)
        # print("----------------------------------------------------------")

    # Img.shape -> (가로,세로)
    def callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            st_Img_BGR = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)

        CamCom = Lane()    

        # Map을 못 불러온다면 에러가 뜨기 때문
        if self.map_x.all() != None and self.map_y.all() != None:
            
            # Shape -> 780x600
            st_Img_Remap = cv2.remap(st_Img_BGR, self.map_x.astype(np.float32),self.map_y.astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            st_Img_Gray = cv2.cvtColor(st_Img_Remap, cv2.COLOR_RGB2GRAY)
            st_Img_Gray = cv2.GaussianBlur(st_Img_Gray, (1,1), 0)

            _, dst = cv2.threshold(st_Img_Gray,170,255, cv2.THRESH_BINARY)            #retval : 사용된 임계값, dst : 출력 영상 // Origin
            # cv2.imshow("dstBefore",dst)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))    
            dst2 = cv2.dilate(dst, k)
            k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))    
            dst3 = cv2.dilate(dst, k2)
            st_Edges = cv2.Canny(dst,100,360)
            # cv2.imshow("dst2",dst2)
            # cv2.imshow("dst3",dst3)

            # make nonzero pixel array
            nonzero = st_Edges.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # histogram
            histogram = np.sum(st_Edges[700:,:],axis = 0)
            # self.ShowHistogram(histogram)

            # IPM Image의 반을 잘라서 왼쪽 차선과 오른쪽 차선을 구분
            s32_MidPoint_Width = np.int32(histogram.shape[0]/2)

            # 최대 5개의 값을 내림차순으로 정렬 후 차량의 Heading과 비교하여 가장 가까운 값을 사용
            indicesl = np.argsort(histogram[:s32_MidPoint_Width])[-5:][::-1]
            indicesr = np.argsort(histogram[s32_MidPoint_Width:])[-5:][::-1] + s32_MidPoint_Width
            diff = np.abs(indicesl - s32_MidPoint_Width)  
            L = indicesl[np.argmin(diff)]  
            diff = np.abs(indicesr - s32_MidPoint_Width)  
            R = indicesr[np.argmin(diff)]  

            s32_Point_LeftWidth = L
            s32_Point_RightWidth = R
            
            s32_ParamY = 780
            ars32_Pnt_Lx = []
            ars32_Pnt_Rx = []

            win_left_lane = []
            win_right_lane = []
            leftxx = []
            leftyy = []
            rightxx = []
            rightyy = []

            st_Msk = st_Edges.copy()

            output = np.dstack((st_Edges, st_Edges, st_Edges)) * 255

            b_Valid_L = True
            b_Valid_R = True

            # Height 전반에 대해서 Window 생성
            while s32_ParamY > 0:
                ## Left Threshold
                tmp = LanePnt()
                b_Win_L = False
                b_Win_R = False

                Win_LeftX_Min = s32_Point_LeftWidth - self.Param_Margin_x
                Win_LeftX_Max = s32_Point_LeftWidth + self.Param_Margin_x
                Win_RightX_Min = s32_Point_RightWidth - self.Param_Margin_x
                Win_RightX_Max = s32_Point_RightWidth + self.Param_Margin_x
                Win_Y_min = s32_ParamY-self.Param_Margin_y

                # Window가 IPM Image 범위에 나갔을 경우 해당 차선에 대한 차선 검출 Logic Stop

                st_Window = st_Msk[Win_Y_min:s32_ParamY, Win_LeftX_Min:Win_LeftX_Max]
                contours, _ = cv2.findContours(st_Window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # 차선에 대한 모멘트 계산
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"]/M["m00"])
                        cy = int(M["m01"]/M["m00"])
                        ars32_Pnt_Lx.append(Win_LeftX_Min + cx)
                        s32_Point_LeftWidth = Win_LeftX_Min + cx
                        s32_Pnt_Left_Y = s32_ParamY - cy
                        b_Win_L = True
                    
                cv2.rectangle(st_Msk, (Win_LeftX_Min,s32_ParamY),(Win_LeftX_Max,Win_Y_min),(255,255,255),2)

                # 원본 Edge Data 중 현재 Window 범위 내의 픽셀을 추출
                left_window_inds = ((nonzeroy >= Win_Y_min) & (nonzeroy <= s32_ParamY) & (nonzerox >= Win_LeftX_Min) & (
                    nonzerox <= Win_LeftX_Max)).nonzero()[0]
                
                # print("left_window_inds")
                # print(left_window_inds)
                # print("---------------------------------------")
                     

                # 차선이 끊겼는지 판단 : Window가 3칸 이상 빈 Window인 경우 차선이 가려졌거나 없는 것이라 판단
                # 해당 부분의 픽셀을 차선 픽셀이라고 최종적으로 판단하지 않아 차가 회전 중일 때 오판이 되는 경우를 방지
                if len(left_window_inds) == 0:
                    if self.Left_Befor == 0:
                        self.Cnt_L += 1
                    else:
                        self.Cnt_L = 0
                self.Left_Befor = len(left_window_inds)

                if self.Cnt_L > 4:
                    b_Valid_L = False

                # 타당한 차선에 대한 원본의 픽셀 좌표를 저장
                # 3칸 이상 빈 윈도우 존재 -> 타당하지 않다고 판단
                if b_Valid_L:
                    win_left_lane.append(left_window_inds)
                    
                    sorted_indices = sorted(list(set(nonzerox[left_window_inds])))[-1:] 

                    # print("sorted_indices")
                    # print(sorted_indices)
                    # print("---------------------------------------")

                    for idx, x in enumerate(nonzerox[left_window_inds]):
                        if x in sorted_indices:
                            leftxx.append(x)
                            leftyy.append(nonzeroy[left_window_inds][idx])

                # -------------------------------------------------------------------------------------------------------------------
                # 우측 차선 판단 로직
                if Win_RightX_Max < st_Edges.shape[1]:
                    
                    st_Window = st_Msk[Win_Y_min:s32_ParamY, Win_RightX_Min:Win_RightX_Max]
                    contours, _ = cv2.findContours(st_Window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"]/M["m00"])
                            cy = int(M["m01"]/M["m00"])
                            ars32_Pnt_Rx.append(Win_RightX_Min + cx)
                            s32_Point_RightWidth = Win_RightX_Min + cx
                            s32_Pnt_Right_Y = s32_ParamY - cy
                            b_Win_R = True

                    cv2.rectangle(st_Msk, (Win_RightX_Min,s32_ParamY),(Win_RightX_Max,Win_Y_min),(255,255,255),2)

                    right_window_inds = ((nonzeroy >= Win_Y_min) & (nonzeroy <= s32_ParamY) & (nonzerox >= Win_RightX_Min) & (
                        nonzerox <= Win_RightX_Max)).nonzero()[0]

                    if len(right_window_inds) == 0:
                        if self.Right_Befor == 0:
                            self.Cnt_R += 1
                        else:
                            self.Cnt_R = 0
                    self.Right_Befor = len(right_window_inds)
                    if self.Cnt_R > 4:
                        b_Valid_R = False

                    if b_Valid_R:
                        win_right_lane.append(right_window_inds)

                        sorted_indices = sorted(list(set(nonzerox[right_window_inds])))[:1]  
                        # print("sorted_indices")
                        # print(sorted_indices)
                        # print("---------------------------------------")


                        for idx, x in enumerate(nonzerox[right_window_inds]):
                            if x in sorted_indices:
                                rightxx.append(x)
                                rightyy.append(nonzeroy[right_window_inds][idx])

                # 현재 주행 차선의 Centor Pnt를 저장
                if b_Win_L and b_Win_R:
                    if b_Valid_L and b_Valid_R:
                        cv2.circle(output, (int((s32_Point_LeftWidth+s32_Point_RightWidth)/2),int((s32_Pnt_Left_Y+s32_Pnt_Right_Y)/2)),2, (255, 255, 255), cv2.FILLED, cv2.LINE_4)
                        tmp.Y = ((s32_Point_LeftWidth+s32_Point_RightWidth)/2-st_Msk.shape[0]//2)*0.025
                        tmp.X = (st_Msk.shape[1]-(s32_Pnt_Left_Y+s32_Pnt_Right_Y)/2)*0.05
                        CamCom.arf32_Lane.append(tmp)
                
                # # 좌,우측 중 한쪽 차선만 감지되는 경우 나머지 차선을 보간하기 위한 데이터 저장
                # elif b_Win_L and b_Win_R == False:
                #     if b_Valid_L and b_Valid_R == False:
                #         self.Interp_L.append([s32_Pnt_Left_Y,s32_Point_LeftWidth])

                # elif b_Win_R and b_Win_L == False:
                #     if b_Valid_R and b_Valid_L == False:
                #         self.Interp_R.append([s32_Pnt_Right_Y,s32_Point_RightWidth])
                    
                s32_ParamY -= self.Param_Margin_y

            if len(win_left_lane) != 0:
                win_left_lane = np.concatenate(win_left_lane)
            if len(win_right_lane) != 0:
                win_right_lane = np.concatenate(win_right_lane)

            # if len(lefty) != 0:
            #     lefty = np.concatenate(lefty)
            # if len(leftx) != 0:
            #     leftx = np.concatenate(leftx)

            # # Extract left and right line pixel positionsd
            leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
            rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

            output[lefty, leftx] = [255, 0, 0]
            output[righty, rightx] = [0, 0, 255]

            # print("---------------------------------")
            # for i,j in zip(rightx,righty):
            #     print(f"({i},{-j+780})")
            # print("---------------------------------")

            # lefty = [-x+780 for x in lefty]
            # righty = [-x+780 for x in righty]

            leftyy = [-x+780 for x in leftyy]
            rightyy = [-x+780 for x in rightyy]

            if len(leftyy) != 0 and len(leftxx):
                left_fit = np.polyfit(leftxx, leftyy, 1)
                left_fit[0] = round(left_fit[0],6)
                left_fit[1] = round(left_fit[1],0)
                # print(f"y=+{left_fit[0]}*x + {left_fit[1]}")

                if np.min(leftyy) < 400:
                    self.Coef_L = left_fit

            if len(rightyy) != 0 and len(rightxx):
                right_fit = np.polyfit(rightxx, rightyy, 1)
                right_fit[0] = round(right_fit[0],6)
                right_fit[1] = round(right_fit[1],0)
                # print(f"y=+{right_fit[0]}*x + {right_fit[1]}")

                if np.min(righty) < 400:
                    self.Coef_R = right_fit
            
            left_fit = self.Coef_L
            right_fit = self.Coef_R

            # Generate x and y values for plotting1
            ploty = np.linspace(0, st_Edges.shape[0] - 1, st_Edges.shape[0])
            # ax + n
            left_plotx = left_fit[0] * ploty + left_fit[1]
            right_plotx = right_fit[0] * ploty + right_fit[1]

            #### Calculate Minimum Distance For Kalman####
            if self.b_Kalman:
                # Save Last Coef
                self.LastSolvedX = self.SolvedX
                self.LastSolvedY = self.SolvedY
                self.f64_LastMinDistance = self.f64_MinDistance
                self.f64_LastAngle = self.f64_Angle

                self.CalculateMinDistance(left_fit[0],left_fit[1])

                f64_DeltaDistance = self.f64_MinDistance - self.f64_LastMinDistance
                f64_DeltaAngle = self.f64_Angle - self.f64_LastAngle

                st_Observation = np.array([self.f64_MinDistance,f64_DeltaDistance, self.f64_Angle, f64_DeltaAngle])

                if len(self.st_KalmanList) == 0:
                    st_KalmanObject = KalmanObject()
                    st_KalmanObject.InitializateKalmanParamter()
                    st_KalmanObject.UpdateObservation(st_Observation)
                    st_KalmanObject.SetInitialX(st_Observation)
                    st_KalmanObject.s64_CoefA = left_fit[0]
                    st_KalmanObject.s64_CoefB = left_fit[1]
                    self.st_KalmanList.append(st_KalmanObject)
                    
                else:
                    # 기존의 Kalman Object와 비교해서 동일한 차선임을 확인
                    # 동일한 차선이면 원래 가지고 있던 객체를 Update
                    for TrackedKalmanLane in self.st_KalmanList:
                        TrackedKalmanLane.CheckSameOBject(st_Observation)
                        if TrackedKalmanLane.b_SameObject:
                            TrackedKalmanLane.s64_CoefA = left_fit[0]
                            TrackedKalmanLane.s64_CoefB = left_fit[1]
                            TrackedKalmanLane.UpdateObservation(st_Observation)
                            TrackedKalmanLane.PredictState()
                        else:
                            TrackedKalmanLane.s32_Count += 1
                            TrackedKalmanLane.PredictState()

                ####################################

                ############Check Kalman Result#############
                # print("self.st_KalmanList: ", len(self.st_KalmanList))
                Kalman_Mask = np.zeros_like(output)
                for TrackedKalmanLane in self.st_KalmanList:
                    left_plotx = TrackedKalmanLane.s64_CoefA * ploty + TrackedKalmanLane.s64_CoefB
                    for i in range(len(ploty)-1):
                            # ax + n
                            cv2.line(Kalman_Mask, (int(left_plotx[i]), int(ploty[i])), (int(left_plotx[i+1]), int(ploty[i+1])), (255, 0, 0), thickness=1)

                ####################################


            # print("Left")
            # print(left_fit[0],",",left_fit[1])
            # print("Right")
            # print(right_fit[0],",",right_fit[1])

            # if len(lefty) == 0 or len(righty) == 0:
            #     for i in range(450, -1, -20):
            #         cv2.circle(output, (int((left_plotx[i]+right_plotx[i])/2),int((ploty[i]+ploty[i])/2)),2, (0, 120, 120), cv2.FILLED, cv2.LINE_4)
            # elif np.min(lefty) > 200 or np.min(righty) > 200:
            #     for i in range(280, -1, -20):
            #         cv2.circle(output, (int((left_plotx[i]+right_plotx[i])/2),int((ploty[i]+ploty[i])/2)),2, (0, 0, 255), cv2.FILLED, cv2.LINE_4)

            # for left in self.Interp_L:
            #     cv2.circle(output, (int((left[1]+right_plotx[left[0]])/2),int((left[0]+left[0])/2)),2, (255, 0, 0), cv2.FILLED, cv2.LINE_4)

            # for left in self.Interp_L:
            #     cv2.circle(output, (int((left[1]+right_plotx[left[0]])/2),int((left[0]+left[0])/2)),2, (255, 0, 0), cv2.FILLED, cv2.LINE_4)

            # mask = np.zeros_like(output)

            # for i in range(len(ploty)-1):
            #     cv2.line(mask, (int(left_plotx[i]), int(ploty[i])), (int(left_plotx[i+1]), int(ploty[i+1])), (255, 0, 0), thickness=1)
            #     cv2.line(mask, (int(right_plotx[i]), int(ploty[i])), (int(right_plotx[i+1]), int(ploty[i+1])), (0, 0, 255), thickness=1)
            
            self.publisher.publish(CamCom)


            # Calculate Hz
            self.s64_EndTime = time.time()
            s64_Duration = self.s64_EndTime-self.s64_StartTime
            self.s64_StartTime = time.time()
            s64_Frequency = 1.0 / s64_Duration
            # print("Hz: ", s64_Frequency)

            # Param & Flag Initialization
            self.Cnt_L = 0
            self.Cnt_R = 0
            self.Left_Befor = 0
            self.Right_Befor = 0
            # self.Interp_L=[]
            # self.Interp_R=[]

            # Result = np.concatenate(st_Msk,output,mask)
            # Output = np.concatenate((output,mask), axis=1)
            # Output = np.concatenate((output,mask,Kalman_Mask), axis=1)
            # cv2.imshow("Output",Output)
            # cv2.imshow("st_Msk",st_Msk)
            # cv2.imshow("output",output)
            # cv2.imshow("mask",mask)
            cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('image_parser', anonymous=True)
    image_parser = IMGParser()
    rospy.spin()
