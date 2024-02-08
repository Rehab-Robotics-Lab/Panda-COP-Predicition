#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright 2017 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

# *******************************************************************************
# ***********************     Bulk Read and Bulk Write Example      ***********************
#  Required Environment to run this example :
#    - Protocol 2.0 supported DYNAMIXEL(X, P, PRO/PRO(A), MX 2.0 series). Note that the XL320 does not support Bulk Read and Bulk Write.
#    - DYNAMIXEL Starter Set (U2D2, U2D2 PHB, 12V SMPS)
#  How to use the example :
#    - Select the DYNAMIXEL in use at the MY_DXL in the example code.
#    - Build and Run from proper architecture subdirectory.
#    - For ARM based SBCs such as Raspberry Pi, use linux_sbc subdirectory to build and run.
#    - https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/
#  Author: Ryu Woon Jung (Leon)
#  Maintainer : Zerom, Will Son
# *******************************************************************************

import os
import numpy as np

if os.name == "nt":
    import msvcrt

    def getch():
        return msvcrt.getch().decode()

else:
    import sys, tty, termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


from dynamixel_sdk import *  # Uses Dynamixel SDK library


# ********* DYNAMIXEL Model definition *********
# ***** (Use only one definition at a time) *****
MY_DXL = "MX_SERIES"  # X330 (5.0 V recommended), X430, X540, 2X430
# MY_DXL = 'MX_SERIES'    # MX series with 2.0 firmware update.
# MY_DXL = 'PRO_SERIES'   # H54, H42, M54, M42, L54, L42
# MY_DXL = 'PRO_A_SERIES' # PRO series with (A) firmware update.
# MY_DXL = 'P_SERIES'     # PH54, PH42, PM54

# Control table address
# if MY_DXL == "X_SERIES" or MY_DXL == "MX_SERIES":
ADDR_TORQUE_ENABLE = 64
ADDR_LED_RED = 65
LEN_LED_RED = 1  # Data Byte Length
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4  # Data Byte Length
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4  # Data Byte Length
DXL_MINIMUM_POSITION_VALUE = 0  # Refer to the Minimum Position Limit of product eManual
DXL_MAXIMUM_POSITION_VALUE = (
    4095  # Refer to the Maximum Position Limit of product eManual
)
ADDR_PROFILE_VELOCITY = 112
LEN_GOAL_VELOCITY = 4
BAUDRATE = 57600
# elif MY_DXL == "PRO_SERIES":
#     ADDR_TORQUE_ENABLE = 562  # Control table address is different in DYNAMIXEL model
#     ADDR_LED_RED = 563  # R.G.B Address: 563 (red), 564 (green), 565 (blue)
#     LEN_LED_RED = 1  # Data Byte Length
#     ADDR_GOAL_POSITION = 596
#     LEN_GOAL_POSITION = 4
#     ADDR_PRESENT_POSITION = 611
#     LEN_PRESENT_POSITION = 4
#     DXL_MINIMUM_POSITION_VALUE = (
#         -150000
#     )  # Refer to the Minimum Position Limit of product eManual
#     DXL_MAXIMUM_POSITION_VALUE = (
#         150000  # Refer to the Maximum Position Limit of product eManual
#     )
#     BAUDRATE = 57600
# elif MY_DXL == "P_SERIES" or MY_DXL == "PRO_A_SERIES":
#     ADDR_TORQUE_ENABLE = 512  # Control table address is different in DYNAMIXEL model
#     ADDR_LED_RED = 513  # R.G.B Address: 513 (red), 544 (green), 515 (blue)
#     LEN_LED_RED = 1  # Data Byte Length
#     ADDR_GOAL_POSITION = 564
#     LEN_GOAL_POSITION = 4  # Data Byte Length
#     ADDR_PRESENT_POSITION = 580
#     LEN_PRESENT_POSITION = 4  # Data Byte Length
#     DXL_MINIMUM_POSITION_VALUE = (
#         -150000
#     )  # Refer to the Minimum Position Limit of product eManual
#     DXL_MAXIMUM_POSITION_VALUE = (
#         150000  # Refer to the Maximum Position Limit of product eManual
#     )
#     BAUDRATE = 57600

# DYNAMIXEL Protocol Version (1.0 / 2.0)
# https://emanual.robotis.com/docs/en/dxl/protocol2/
PROTOCOL_VERSION = 2.0

# Make sure that each DYNAMIXEL ID should have unique ID.
# DXL1_ID                     = 1                 # Dynamixel#1 ID : 1
# DXL2_ID                     = 2                  # Dynamixel#1 ID : 2

motors = [65, 66]

# Use the actual port assigned to the U2D2.
# ex) Windows: "COM*", Linux: "/dev/ttyUSB*", Mac: "/dev/tty.usbserial-*"


def angle_to_PWM(theta):
    PWM = (
        theta
        * (
            DXL_MAXIMUM_POSITION_VALUE * np.ones_like(theta)
            - DXL_MINIMUM_POSITION_VALUE
        )
        / 360
    ).astype(int)
    return PWM


def PWM_to_angle(PWM):
    angle = (
        PWM
        * 360
        / (DXL_MAXIMUM_POSITION_VALUE * np.ones_like(PWM) - DXL_MINIMUM_POSITION_VALUE)
    ).astype(int)
    return angle


def enable_torque(DXL_ID):
    # Enable Dynamixel Torque
    for ID in DXL_ID:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
            portHandler, ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
        )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully connected" % ID)

        # Add parameter storage for Dynamixel#1 present position
        dxl_addparam_result = groupBulkRead.addParam(
            ID, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        if dxl_addparam_result != True:
            print("[ID:%03d] groupBulkRead addparam failed" % ID)
            quit()


def disable_torque(DXL_ID):
    # Disable Dynamixel Torque
    for ID in DXL_ID:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
            portHandler, ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))


def move(DXL_ID, dxl_goal_position):
    # print(dxl_goal_position)
    param_goal_position = [
        [
            DXL_LOBYTE(DXL_LOWORD(dxl_goal_position[0])),
            DXL_HIBYTE(DXL_LOWORD(dxl_goal_position[0])),
            DXL_LOBYTE(DXL_HIWORD(dxl_goal_position[0])),
            DXL_HIBYTE(DXL_HIWORD(dxl_goal_position[0])),
        ]
    ]

    for j in range(len(motors) - 1):
        # print(
        #     [
        #         DXL_LOBYTE(DXL_LOWORD(dxl_goal_position[j + 1])),
        #         DXL_HIBYTE(DXL_LOWORD(dxl_goal_position[j + 1])),
        #         DXL_LOBYTE(DXL_HIWORD(dxl_goal_position[j + 1])),
        #         DXL_HIBYTE(DXL_HIWORD(dxl_goal_position[j + 1])),
        #     ]
        # )
        param_goal_position.append(
            [
                DXL_LOBYTE(DXL_LOWORD(dxl_goal_position[j + 1])),
                DXL_HIBYTE(DXL_LOWORD(dxl_goal_position[j + 1])),
                DXL_LOBYTE(DXL_HIWORD(dxl_goal_position[j + 1])),
                DXL_HIBYTE(DXL_HIWORD(dxl_goal_position[j + 1])),
            ]
        )

    for i in range(len(motors)):
        # Add Dynamixel#1 goal position value to the Bulkwrite parameter storage
        ID = motors[i]
        print(ID, param_goal_position[i])
        dxl_addparam_result = groupBulkWrite.addParam(
            ID, ADDR_GOAL_POSITION, LEN_GOAL_POSITION, param_goal_position[i]
        )
        if dxl_addparam_result != True:
            print("[ID:%03d] groupBulkWrite addparam failed" % ID)
            quit()

    # Bulkwrite goal position and LED value
    dxl_comm_result = groupBulkWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))

    # Clear bulkwrite parameter storage
    groupBulkWrite.clearParam()


def read(DXL_ID, dxl_goal_position):
    break_flag = False
    while break_flag == False:
        # Bulkread present position and LED status
        dxl_comm_result = groupBulkRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))

        for ID in motors:
            # Check if groupbulkread data of Dynamixel#1 is available
            dxl_getdata_result = groupBulkRead.isAvailable(
                ID, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
            )
            if dxl_getdata_result != True:
                print("[ID:%03d] groupBulkRead getdata failed" % ID)
                quit()

        for i in range(len(DXL_ID)):
            dxl_present_position = groupBulkRead.getData(
                motors[i], ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
            )
            # print("[ID:%03d] Present Position : %d \t [ID:%03d] LED Value: %d" % (DXL_ID[i], dxl_present_position))
            # print(dxl_goal_position[i] ,dxl_present_position,abs(dxl_goal_position[i] - dxl_present_position))
            print(i + 1, PWM_to_angle(dxl_present_position))
            if not (
                abs(dxl_goal_position[i] - dxl_present_position)
                > DXL_MOVING_STATUS_THRESHOLD
            ):
                break_flag = True
                break
        # while 1:
    #     # Bulkread present position and LED status
    #     dxl_comm_result = groupBulkRead.txRxPacket()
    #     if dxl_comm_result != COMM_SUCCESS:
    #         print("%s" % packetHandler.getTxRxResult(dxl_comm_result))

    #     for ID in motors:
    #         # Check if groupbulkread data of Dynamixel#1 is available
    #         dxl_getdata_result = groupBulkRead.isAvailable(ID, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    #         if dxl_getdata_result != True:
    #             print("[ID:%03d] groupBulkRead getdata failed" % ID)
    #             quit()

    #     # Get present position value
    #     dxl1_present_position = groupBulkRead.getData(motors[0], ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    #     dxl2_present_position = groupBulkRead.getData(motors[1], ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)

    #     print("[ID:%03d] Present Position : %d \t [ID:%03d] LED Value: %d" % (motors[0], dxl1_present_position, motors[1], dxl2_present_position))

    #     if not (abs(dxl_goal_position1[index] - dxl1_present_position) > DXL_MOVING_STATUS_THRESHOLD):
    #         break
    #     if not (abs(dxl_goal_position2[index] - dxl2_present_position) > DXL_MOVING_STATUS_THRESHOLD):
    #         break


def home(DXL_ID):
    home_positions = np.zeros([len(DXL_ID)]).astype(int)
    move(DXL_ID, home_positions)


def home_differential(DXL_ID):
    home_positions = angle_to_PWM(np.ones([len(DXL_ID)]).astype(int) * 180)
    move(DXL_ID, home_positions)


def side_bending(DXL_ID, goal_angle, previous_angle, previous_motor_angle):
    gear_ratio = 30 / 20

    previous_angle = previous_angle * gear_ratio
    goal_angle = goal_angle * gear_ratio

    dxl1_present_position = previous_motor_angle[0]
    dxl2_present_position = previous_motor_angle[1]

    angle1 = dxl1_present_position + angle_to_PWM(abs(previous_angle - goal_angle))
    angle2 = dxl2_present_position - angle_to_PWM(abs(previous_angle - goal_angle))

    goal = [angle1, angle2]
    print("293: ", goal)
    move(DXL_ID, goal)


def rotation(DXL_ID, goal_angle, previous_angle, previous_motor_angle):
    gear_ratio = 30 / 20

    previous_angle = previous_angle * gear_ratio
    goal_angle = goal_angle * gear_ratio

    dxl1_present_position = previous_motor_angle[0]
    dxl2_present_position = previous_motor_angle[1]

    angle1 = dxl1_present_position + angle_to_PWM(abs(previous_angle - goal_angle))
    angle2 = dxl2_present_position + angle_to_PWM(abs(previous_angle - goal_angle))

    goal = [angle1, angle2]
    move(DXL_ID, goal)


DEVICENAME = "com3"

TORQUE_ENABLE = 1  # Value for enabling the torque
TORQUE_DISABLE = 0  # Value for disabling the torque
DXL_MOVING_STATUS_THRESHOLD = 20  # Dynamixel moving status threshold

index = 0
dxl_goal_position1 = angle_to_PWM([0, 180, 90, 0])  # Goal position 1
# dxl_goal_position2 = [DXL_MINIMUM_POSITION_VALUE, DXL_MAXIMUM_POSITION_VALUE]
dxl_goal_position2 = angle_to_PWM([180, 0, 90, 0])  # Goal position 2

dxl_goal_positions = np.array([dxl_goal_position1, dxl_goal_position2])


dxl_led_value = [0x00, 0x01]  # Dynamixel LED value for write

# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Set the protocol version
# Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Initialize GroupBulkWrite instance
groupBulkWrite = GroupBulkWrite(portHandler, packetHandler)

# Initialize GroupBulkRead instace for Present Position
groupBulkRead = GroupBulkRead(portHandler, packetHandler)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()


# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()


enable_torque(motors)
home_differential(motors)

while 1:
    print("Press any key to continue! (or press ESC to quit!)")
    if getch() == chr(0x1B):
        break

    goal = dxl_goal_positions[:, index]
    # move(motors,goal)

    prev_motor = angle_to_PWM(np.ones([len(motors)]).astype(int) * 180)
    print("PP ", prev_motor)
    side_bending(motors, 180, 0, prev_motor)
    read(motors, goal)

    # Change goal position
    if index == 3:
        index = 0
    else:
        index = index + 1

# Clear bulkread parameter storage
groupBulkRead.clearParam()

disable_torque(motors)

# Close port
portHandler.closePort()
