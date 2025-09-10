#### Based on code from https://github.com/Interbotix/dynamixelsdk_xsarm_examples/blob/main/python/interbotix_arm.py

import time
import numpy as np
import pandas as pd
from dynamixel_sdk import *

ADDR_DRIVE_MODE = 10
LEN_DRIVE_MODE = 1

ADDR_OPERATING_MODE = 11
LEN_OPERATING_MODE = 1

ADDR_TORQUE_ENABLE = 64
LEN_TORQUE_ENABLE = 1

ADDR_PROFILE_ACCELERATION = 108
LEN_PROFILE_ACCELERATION = 4

ADDR_PROFILE_VELOCITY = 112
LEN_PROFILE_VELOCITY = 4

ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4

ADDR_PRESENT_LOAD = 126
LEN_PRESENT_LOAD = 2

ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4

ADDR_PRESENT_VELOCITY = 128
LEN_PRESENT_VELOCITY = 4

ADDR_PRESENT_TEMP = 146
LEN_PRESENT_TEMP = 1

MOVING = 122
LEN_MOVING = 1

PI = 3.14159265

DXL_MINIMUM_POSITION_VALUE = 0
DXL_MAXIMUM_POSITION_VALUE = 4095


### @brief Helper function to check error messages and print them if need be
### @param dxl_comm_result - if nonzero, there is an error in the communication
### @param dxl_error - if nonzero, there is an error in something related to the data
### @return <bool> - true if no error occurred; false otherwise
def checkError(dxl_comm_result, dxl_error):
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        return False
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
        return False
    return True


### @brief Writes data to the specified register for a given motor
### @param id - Dynamixel ID to write data to
### @param address - register number
### @param data - data to write
### @param length - size of register in bytes
### @return <bool> - true if data was successfully written; false otherwise
def itemWrite(id, address, data, length):
    if length == 1:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
            portHandler, id, address, data
        )
    elif length == 2:
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
            portHandler, id, address, data
        )
    elif length == 4:
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
            portHandler, id, address, data
        )
    else:
        print("Invalid data length...")
        return False
    return checkError(dxl_comm_result, dxl_error)


### @brief Sequentially writes the same data to the specified register for a group of motors
### @param ids - vector of IDs to write data to
### @param address - register number
### @param data - data to write (could be a list [index matches index in 'ids' list] or a single value [for all ids])
### @param length - size of register in bytes
### @return <bool> - true if data was successfully written; false otherwise
def itemWriteMultiple(ids, address, data, length):
    if type(data) != list:
        for id in ids:
            success = itemWrite(id, address, data, length)
            if success != True:
                return False
    else:
        for id, dat in zip(ids, data):
            success = itemWrite(id, address, dat, length)
            if success != True:
                return False
    return True


### @brief Reads data from the specified register for a given motor
### @param id - Dynamixel ID to read data from
### @param address - register number
### @param length - size of register in bytes
### @return state - variable to store the requested data
### @return <bool> - true if data was successfully retrieved; false otherwise
### @details - DynamixelSDK uses 2's complement so we need to check to see if 'state' should be negative (hex numbers)
def itemRead(id, address, length):
    if length == 1:
        state, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(
            portHandler, id, address
        )
    elif length == 2:
        state, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(
            portHandler, id, address
        )
        if state > 0x7FFF:
            state = state - 65536
    elif length == 4:
        state, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
            portHandler, id, address
        )
        if state > 0x7FFFFFFF:
            state = state - 4294967296
    else:
        print("Invalid data length...")
        return 0, False
    return state, checkError(dxl_comm_result, dxl_error)


### @brief Sequentially reads data from the specified register for a group of motors
### @param id - vector of Dynamixel IDs to read data from
### @param address - register number
### @param length - size of register in bytes
### @return states - list to store the requested data
### @return <bool> - true if data was successfully retrieved; false otherwise
def itemReadMultiple(ids, address, length):
    states = []
    for id in ids:
        state, success = itemRead(id, address, length)
        if success != True:
            return [], False
        states.append(state)
    return states, True


def bulkWrite(ids, commands):
    # print(dxl_goal_position)
    param_goal_position = [
        [
            DXL_LOBYTE(DXL_LOWORD(commands[0])),
            DXL_HIBYTE(DXL_LOWORD(commands[0])),
            DXL_LOBYTE(DXL_HIWORD(commands[0])),
            DXL_HIBYTE(DXL_HIWORD(commands[0])),
        ]
    ]

    for j in range(len(ids) - 1):
        param_goal_position.append(
            [
                DXL_LOBYTE(DXL_LOWORD(commands[j + 1])),
                DXL_HIBYTE(DXL_LOWORD(commands[j + 1])),
                DXL_LOBYTE(DXL_HIWORD(commands[j + 1])),
                DXL_HIBYTE(DXL_HIWORD(commands[j + 1])),
            ]
        )

    for i in range(len(ids)):
        # Add Dynamixel#1 goal position value to the Bulkwrite parameter storage
        ID = ids[i]
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


def bulkRead(ids, address, length):
    groupBulkRead.clearParam()
    for id in ids:
        dxl_addparam_result = groupBulkRead.addParam(id, address, length)
        if dxl_addparam_result != True:
            print("ID:%03d groupBulkRead addparam failed", id)
            return [], False

    dxl_comm_result = groupBulkRead.txRxPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        return [], False

    states = []
    for id in ids:
        state = groupBulkRead.getData(id, address, length)
        if length == 2 and state > 0x7FFF:
            state = state - 65536
        elif length == 4 and state > 0x7FFFFFFF:
            state = state - 4294967296
        states.append(state)
    return states, True


### @brief Initializes the port that the U2D2 is connected to
### @param port_name - name of the port
### @baudrate - desired baudrate in bps (should be the same as the motors)
### @return <bool> - true if the port was initialized; false otherwise
def initPort(port_name, baudrate):
    global portHandler, packetHandler
    portHandler = PortHandler(port_name)
    packetHandler = PacketHandler(2.0)

    if portHandler.openPort():
        print("Successfully opened the port at %s!" % port_name)
    else:
        print("Failed to open the port at %s!", port_name)
        return False

    if portHandler.setBaudRate(baudrate):
        print("Succeeded to change the baudrate to %d bps!" % baudrate)
    else:
        print("Failed to change the baudrate to %d bps!" % baudrate)
        return False

    return True


### @brief Ping the desired motors to verify their existence
### @param ids - vector of Dynamixel IDs to ping
### @return <bool> - true if all IDs were pinged successfully
def ping(ids):
    for id in ids:
        model_num, dxl_comm_result, dxl_error = packetHandler.ping(portHandler, id)
        success = checkError(dxl_comm_result, dxl_error)
        if success:
            print("Pinged ID: %03d successfully! Model Number: %d" % (id, model_num))
        else:
            return False
    return True


def angle2PWM(theta):
    conversion = (DXL_MAXIMUM_POSITION_VALUE - DXL_MINIMUM_POSITION_VALUE) / 360
    PWM = (np.asarray(theta) * conversion).astype(int)
    return PWM.tolist()


def PWM2angle(PWM):
    conversion = 360 / (DXL_MAXIMUM_POSITION_VALUE - DXL_MINIMUM_POSITION_VALUE)
    angle = (np.asarray(PWM) * conversion).astype(int)
    return angle.tolist()


def home_limbs(ids):
    zero_positions, success = position_status(ids)
    # home_angles = PWM2angle(home_positions)
    # print("Homing Limbs")
    air_positions = [0, 0, 0, 0]
    # for pos in home_positions:
    #     pos = pos + angle2PWM(40)
    print("Homing Limbs")
    goal = angle2PWM([20, 20, 20, 20])
    for i in range(4):
        air_positions[i] = zero_positions[i] + goal[i]

    # bulkWrite(ids, air_positions)

    return zero_positions


def home_trunk(ids):
    home_positions = angle2PWM([180] * len(ids))
    print("Homing Trunk")
    bulkWrite(ids, home_positions)


def move_limbs(home, ids, goal_angle):
    goal = [0, 0, 0, 0]
    for i in range(len(ids)):
        goal[i] = angle2PWM(goal_angle[i]) + home[i]
    bulkWrite(ids, goal)


def side_bending(ids, goal_angle):
    gear_ratio = 30 / 20

    goal_angle = goal_angle * gear_ratio

    angle1 = angle2PWM(180) + angle2PWM(goal_angle)
    angle2 = angle2PWM(180) - angle2PWM(goal_angle)

    goal = [angle1, angle2]

    bulkWrite(ids, goal)

    # return goal_angle


def rotation(ids, goal_angle):
    gear_ratio = 30 / 20

    goal_angle = goal_angle * gear_ratio

    present_position = angle2PWM(np.ones([len(ids)]).astype(int) * 180)

    angle1 = angle2PWM(180) + angle2PWM(goal_angle)
    angle2 = angle2PWM(180) + angle2PWM(goal_angle)

    goal = [angle1, angle2]

    bulkWrite(ids, goal)

    # return goal_angle


def moving_status(ids):
    status, success = bulkRead(ids, MOVING, LEN_MOVING)

    return any(status), success


def position_status(ids):
    ## Read current arm joint positions
    positions, success = bulkRead(ids, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    # velocity, success = bulkRead(ids, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)

    return positions, success


def main():
    ## Motor IDs
    r_leg = 4
    l_leg = 3
    r_arm = 2
    l_arm = 1

    # IDs for differential motors and drive modes
    diff_id = [65, 66]
    diff_modes = [4, 4]

    # IDs for limb motors and drive modes
    # limb_ids = [l_arm, r_arm, r_leg, l_leg]
    # limb_modes = [5, 4, 4, 5]
    limb_ids = [l_arm, r_arm, l_leg, r_leg]
    limb_modes = [5, 4, 5, 4]
    limb_op_modes = [4, 4, 4, 4]

    # Drive modes
    all_ids = limb_ids + diff_id
    drive_modes = limb_modes + diff_modes

    # Initial value for differential
    previous_angle = 0

    ## Initialize the port, ping the motors, and create syncWrite and syncRead objects
    ## It's faster and better design to read/write motors with the 'sync' objects than to command each motor sequentially
    ## Commanding each motor sequentially should only be done for 'non-realtime sensitive' registers - like torquing on/off, setting EEPROM registers, etc...

    DEVICENAME = "com3"
    BAUDRATE = 57600

    if not initPort(DEVICENAME, BAUDRATE):
        return
    if not ping(all_ids):
        portHandler.closePort()
        return

    global groupBulkWrite
    global groupBulkRead

    groupBulkWrite = GroupBulkWrite(portHandler, packetHandler)
    groupBulkRead = GroupBulkRead(portHandler, packetHandler)

    ## setting profile velocity and acceleration
    # calauclate (t1+t3) for each movement and profile accelartion/velocity is calculated
    t = 2
    pV = int(t * 0.5 * 1000)
    pA = int(pV * 0.5)

    ## Initialize register values
    itemWriteMultiple(all_ids, ADDR_TORQUE_ENABLE, 0, LEN_TORQUE_ENABLE)
    print("zero the limbs")
    time.sleep(5)
    itemWriteMultiple(limb_ids, ADDR_OPERATING_MODE, limb_op_modes, LEN_OPERATING_MODE)
    itemWriteMultiple(all_ids, ADDR_DRIVE_MODE, drive_modes, LEN_DRIVE_MODE)
    itemWriteMultiple(all_ids, ADDR_PROFILE_VELOCITY, pV, LEN_PROFILE_VELOCITY)
    itemWriteMultiple(all_ids, ADDR_PROFILE_ACCELERATION, pA, LEN_PROFILE_ACCELERATION)
    itemWriteMultiple(all_ids, ADDR_TORQUE_ENABLE, 1, LEN_TORQUE_ENABLE)

    ## Home all motors at the start

    home_trunk(diff_id)
    # itemWriteMultiple(diff_id, ADDR_TORQUE_ENABLE, 0, LEN_TORQUE_ENABLE)
    home_pos = home_limbs(limb_ids)

    time.sleep(5)
    print("Hit record")
    time.sleep(5)

    output = []
    start_time = time.time()

    # itemWriteMultiple([limb_ids], ADDR_TORQUE_ENABLE, 0, LEN_TORQUE_ENABLE)
    print("Starting")

    ## limbs + trunk

    # 0-5s
    print("0-5s")
    move_limbs(home_pos, limb_ids, [180, 180, 0, 0])
    time.sleep(0.7)
    move_limbs(home_pos, limb_ids, [0, 0, 120, 120])
    time.sleep(0.7)
    move_limbs(home_pos, limb_ids, [0, 0, 120, 0])
    time.sleep(0.7)
    move_limbs(home_pos, limb_ids, [0, 0, 120, 120])
    time.sleep(0.7)
    move_limbs(home_pos, limb_ids, [0, 0, 120, 0])
    time.sleep(0.7)
    move_limbs(home_pos, limb_ids, [0, 0, 120, 120])
    time.sleep(0.7)
    move_limbs(home_pos, limb_ids, [0, 0, 120, 0])
    time.sleep(0.7)
    move_limbs(home_pos, limb_ids, [0, 0, 120, 120])
    time.sleep(0.7)

    # 6-10s
    print("6-10s")
    move_limbs(home_pos, limb_ids, [90, 0, 0, 0])
    time.sleep(4)
    move_limbs(home_pos, limb_ids, [0, 90, 0, 0])
    time.sleep(0.7)

    # 11-15s
    print("11-15s")
    move_limbs(home_pos, limb_ids, [0, 0, 60, 60])
    time.sleep(4)

    # 15-20s
    print("16-20s")
    move_limbs(home_pos, limb_ids, [0, 0, 0, 60])
    time.sleep(2)
    move_limbs(home_pos, limb_ids, [0, 0, 120, 120])
    time.sleep(2)
    move_limbs(home_pos, limb_ids, [0, 0, 0, 0])
    time.sleep(1)

    # 21-25s
    print("21-25s")
    # side_bending(diff_id, 35)
    time.sleep(0.7)
    # rotation(diff_id, 45)
    move_limbs(home_pos, limb_ids, [180, 180, 10, 10])
    time.sleep(2)
    move_limbs(home_pos, limb_ids, [180, 180, 120, 120])
    time.sleep(2)

    # 25-30s
    print("25-30s")
    move_limbs(home_pos, limb_ids, [90, 90, 120, 120])
    time.sleep(2)
    # side_bending(diff_id, 0)
    time.sleep(0.7)
    # rotation(diff_id, 0)
    time.sleep(2)

    # 30-35s
    print("31-35s")
    move_limbs(home_pos, limb_ids, [0, 0, 60, 60])
    # side_bending(diff_id, -35)
    time.sleep(2)
    move_limbs(home_pos, limb_ids, [90, 90, 0, 0])
    # side_bending(diff_id, 0)
    time.sleep(2)

    # 35-40s
    print("36-40s")
    move_limbs(home_pos, limb_ids, [0, 0, 60, 60])
    time.sleep(3)
    move_limbs(home_pos, limb_ids, [0, 0, 120, 120])
    time.sleep(2)

    # 41-45s
    print("41-45s")
    move_limbs(home_pos, limb_ids, [90, 90, 0, 0])
    time.sleep(2)
    move_limbs(home_pos, limb_ids, [90, 90, 120, 120])
    # side_bending(diff_id, 35)
    time.sleep(0.7)
    # rotation(diff_id, 45)
    time.sleep(1)

    # 46-50s
    print("46-50s")
    # side_bending(diff_id, 0)
    time.sleep(0.7)
    # rotation(diff_id, 0)
    move_limbs(home_pos, limb_ids, [180, 180, 0, 0])
    time.sleep(0.7)
    move_limbs(home_pos, limb_ids, [90, 180, 120, 120])
    time.sleep(0.7)
    move_limbs(home_pos, limb_ids, [90, 90, 60, 60])
    time.sleep(0.7)
    move_limbs(home_pos, limb_ids, [90, 90, 60, 120])
    time.sleep(0.7)

    # 50-55s
    print("51-55s")
    move_limbs(home_pos, limb_ids, [90, 0, 60, 0])
    time.sleep(1)
    move_limbs(home_pos, limb_ids, [90, 0, 60, 120])
    # side_bending(diff_id, -35)
    time.sleep(0.7)
    # rotation(diff_id, -45)
    time.sleep(1)
    # side_bending(diff_id, 0)
    time.sleep(0.7)
    # rotation(diff_id, 0)
    move_limbs(home_pos, limb_ids, [0, 90, 60, 60])
    time.sleep(1)

    # 56-60s
    print("56-60s")
    move_limbs(home_pos, limb_ids, [0, 90, 0, 0])
    # side_bending(diff_id, 35)
    time.sleep(2)
    # side_bending(diff_id, 0)
    time.sleep(2)

    # 61-66s
    print("61-66s")
    move_limbs(home_pos, limb_ids, [90, 90, 120, 120])
    time.sleep(1)
    move_limbs(home_pos, limb_ids, [90, 180, 0, 60])
    time.sleep(2)
    move_limbs(home_pos, limb_ids, [90, 90, 120, 120])
    time.sleep(2)

    print("done")

    # for i in range(10):

    #     ##limbs
    #     rotation(diff_id, -45)
    #     time.sleep(0.5)
    #     rotation(diff_id, 0)
    #     time.sleep(0.5)
    #     rotation(diff_id, 45)
    #     time.sleep(0.5)
    #     rotation(diff_id, 0)
    #     time.sleep(0.5)

    ##trunk

    # move_limbs(home_pos, limb_ids, [0, 0, 0, 0])
    # time.sleep(0.7)
    # move_limbs(home_pos, limb_ids, [0, 0, 120, 120])

    ## limbs + trunk
    # side_bending(diff_id, -35)
    # move_limbs(home_pos, limb_ids, [120, 120, 120, 0])
    # time.sleep(1)
    # side_bending(diff_id, 0)
    # move_limbs(home_pos, limb_ids, [50, 50, 50, 0])
    # time.sleep(1)
    # side_bending(diff_id, 35)
    # time.sleep(1)
    # side_bending(diff_id, 0)
    # time.sleep(0.5)

    # print(i + 1)

    # for j in range(20):
    #     limb_pos, limb_pos_success = position_status(limb_ids)
    #     # limb_pos = home_pos
    #     # limb_pos_success = True
    #     # print(limb_pos)
    #     trunk_pos, trunk_pos_success = position_status(diff_id)
    #     # limb_status,limb_status_success = moving_status(limb_ids)
    #     # trunk_status,tunk_status_success = moving_status(diff_id)
    #     if all([limb_pos_success, trunk_pos_success]):
    #         current = [
    #             # 120,
    #             time.time() - start_time,
    #             limb_pos[0] - home_pos[0],
    #             limb_pos[1] - home_pos[1],
    #             limb_pos[2] - home_pos[2],
    #             limb_pos[3] - home_pos[3],
    #             trunk_pos[0],
    #             trunk_pos[1],
    #             # limb_vel[0],
    #             # limb_vel[1],
    #             # limb_vel[2],
    #             # limb_vel[3],
    #             # trunk_vel[0],
    #             # trunk_vel[1],
    #             # int(limb_status),
    #             # int(trunk_status),
    #         ]

    #         # print(len(now))

    #         output.append(current)

    # time.sleep(2)

    # output = np.asarray(output)

    # Head = [
    #     # "angle",
    #     "time",
    #     "limb 1 psoition",
    #     "limb 2 psoition",
    #     "limb 3 psoition",
    #     "limb 4 psoition",
    #     "trunk 1 psoition",
    #     "trunk 2 psoition",
    #     # "limb 1 velcoity",
    #     # "limb 2 velcoity",
    #     # "limb 3 velcoity",
    #     # "limb 4 velcoity",
    #     # "trunk 1 velcoity",
    #     # "trunk 2 velcoity",
    #     # "limb moving",
    #     # "trunk moving",
    # ]

    # DF = pd.DataFrame(output, columns=Head)
    # # print(DF)
    # # path=r'C:\Users\franc\Documents\Infant_Sim_data\'
    # print(DF)
    # DF.to_csv(r"C:\Users\franc\Documents\Infant_Sim_data\motor\rotation_45_x10_2s.csv")

    itemWriteMultiple(all_ids, ADDR_TORQUE_ENABLE, 0, LEN_TORQUE_ENABLE)
    portHandler.closePort()


if __name__ == "__main__":
    main()
