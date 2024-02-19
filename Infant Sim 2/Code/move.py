#### Based on code from https://github.com/Interbotix/dynamixelsdk_xsarm_examples/blob/main/python/interbotix_arm.py

import time
import numpy as np
from dynamixel_sdk import *

ADDR_DRIVE_MODE = 10
LEN_DRIVE_MODE = 1

# ADDR_OPERATING_MODE = 11
# LEN_OPERATING_MODE = 1

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


### @brief Writes data to a group of motors synchronously
### @param groupSyncWrite - groupSyncWrite object
### @param ids - list of Dynamixel IDs to write to
### @param commands - list of commands to write to each respective motor
### @param length - size of register in bytes
### @return <bool> - true if data was successfully written; false otherwise
def syncWrite(ids, commands, length):
    groupSyncWrite.clearParam
    for id, cmd in zip(ids, commands):
        param = []
        # print("command ", cmd)
        if length == 4:
            param.append(DXL_LOBYTE(DXL_LOWORD(cmd)))
            param.append(DXL_HIBYTE(DXL_LOWORD(cmd)))
            param.append(DXL_LOBYTE(DXL_HIWORD(cmd)))
            param.append(DXL_HIBYTE(DXL_HIWORD(cmd)))
        elif length == 2:
            param.append(DXL_LOBYTE(cmd))
            param.append(DXL_HIBYTE(cmd))
        else:
            param.append(cmd)

        print(id, param)
        dxl_addparam_result = groupSyncWrite.addParam(id, param)

        if dxl_addparam_result != True:

            print("ID:%03d groupSyncWrite addparam failed" % id)
            return False

    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        return False
    return True


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


### @brief Reads data from a group of motors synchronously
### @param groupSyncRead - groupSyncRead object
### @param ids - list of Dynamixel IDs to read from
### @param address - register number
### @param length - size of register in bytes
### @return states - list to store the requested data
### @return <bool> - true if data was successfully retrieved; false otherwise
### @details - DynamixelSDK uses 2's complement so we need to check to see if 'state' should be negative (hex numbers)
def syncRead(ids, address, length=4):
    groupSyncRead.clearParam()
    for id in ids:
        dxl_addparam_result = groupSyncRead.addParam(id)
        if dxl_addparam_result != True:
            print("ID:%03d groupSyncRead addparam failed", id)
            return [], False

    dxl_comm_result = groupSyncRead.txRxPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        return [], False

    states = []
    for id in ids:
        state = groupSyncRead.getData(id, address, length)
        if length == 2 and state > 0x7FFF:
            state = state - 65536
        elif length == 4 and state > 0x7FFFFFFF:
            state = state - 4294967296
        states.append(state)
    return states, True


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
    home_positions = angle2PWM([0] * len(ids))
    print("Homing Limbs")
    bulkWrite(ids, home_positions)


def home_trunk(ids):
    home_positions = angle2PWM([180] * len(ids))
    print("Homing Trunk")
    bulkWrite(ids, home_positions)


def move_limbs(ids, goal_angle):
    goal = angle2PWM(goal_angle)
    bulkWrite(ids, goal)


def side_bending(ids, goal_angle):
    gear_ratio = 30 / 20

    goal_angle = goal_angle * gear_ratio

    angle1 = angle2PWM(180) - angle2PWM(goal_angle)
    angle2 = angle2PWM(180) + angle2PWM(goal_angle)

    goal = [angle1, angle2]

    bulkWrite(ids, goal)

    # return goal_angle


def rotation(ids, goal_angle):
    gear_ratio = 30 / 20

    goal_angle = goal_angle * gear_ratio

    present_position = angle2PWM(np.ones([len(ids)]).astype(int) * 180)

    angle1 = angle2PWM(180) - angle2PWM(goal_angle)
    angle2 = angle2PWM(180) - angle2PWM(goal_angle)

    goal = [angle1, angle2]

    bulkWrite(ids, goal)

    # return goal_angle


def side_rot(ids, side_angle, rot_angle):
    gear_ratio = 30 / 20

    side_angle = side_angle * gear_ratio
    rot_angle = rot_angle * gear_ratio

    # present_position, success = bulkRead(
    #     ids, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
    # )
    present_position = angle2PWM(np.ones([len(ids)]).astype(int) * 180)

    angle1 = angle2PWM(180 - side_angle - rot_angle)
    angle2 = angle2PWM(180 + side_angle - rot_angle)

    goal = [angle1, angle2]
    print("goal ", goal)

    bulkWrite(ids, goal)


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
    # limb_ids = [l_arm, r_arm, l_leg, r_leg]
    # limb_modes=[5,4,5,4]
    limb_ids = [l_arm, r_arm]
    limb_modes = [5, 4]

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

    # Initializing synch read/write
    global groupSyncWrite
    global groupSyncRead

    groupSyncWrite = GroupSyncWrite(
        portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
    )
    groupSyncRead = GroupSyncRead(
        portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
    )

    # Initializing bulk read/write
    global groupBulkWrite
    global groupBulkRead

    groupBulkWrite = GroupBulkWrite(portHandler, packetHandler)
    groupBulkRead = GroupBulkRead(portHandler, packetHandler)

    ## setting profile velocity and acceleration
    # calauclate (t1+t3) for each movement and profile accelartion/velocity is calculated
    t = 3
    pV = int(t * 0.5 * 1000)
    pA = int(pV * 0.5)

    ## Initialize register values
    itemWriteMultiple(all_ids, ADDR_TORQUE_ENABLE, 0, LEN_TORQUE_ENABLE)
    itemWriteMultiple(all_ids, ADDR_DRIVE_MODE, drive_modes, LEN_DRIVE_MODE)
    itemWriteMultiple(all_ids, ADDR_PROFILE_VELOCITY, pV, LEN_PROFILE_VELOCITY)
    itemWriteMultiple(all_ids, ADDR_PROFILE_ACCELERATION, pA, LEN_PROFILE_ACCELERATION)
    itemWriteMultiple(all_ids, ADDR_TORQUE_ENABLE, 1, LEN_TORQUE_ENABLE)

    ## Read current arm joint positions
    positions, success = syncRead(all_ids, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    speeds, success = syncRead(all_ids, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)

    ## Home all motors at the start
    home_limbs([l_arm, r_arm])
    home_trunk(diff_id)

    time.sleep(5)

    # side_bending(diff_id, 45, previous_angle)

    # Test angles for differential
    # bulkRead(all_ids)
    angles = [-45, 40, -45, 45]
    for theta in angles:
        move_limbs(limb_ids, [theta, theta])
        side_rot(diff_id, theta, theta)
        # side_bending(diff_id, theta)
        # bulkWrite([l_arm, r_arm], [angle2PWM(theta), angle2PWM(theta)])
        print(theta)
        time.sleep(2)

    ## Command the gripper to open for 2 seconds, then close for 2 seconds
    # itemWrite(gripper_id, ADDR_GOAL_PWM, 350, LEN_GOAL_PWM)
    # time.sleep(2)
    # itemWrite(gripper_id, ADDR_GOAL_PWM, -350, LEN_GOAL_PWM)
    # time.sleep(2)
    # itemWrite(gripper_id, ADDR_GOAL_PWM, 0, LEN_GOAL_PWM)

    # ## Read the present temperature of all the motors
    # temps, success = itemReadMultiple(all_ids, ADDR_PRESENT_TEMP, LEN_PRESENT_TEMP)
    # for id, temp in zip(all_ids, temps):
    #     print("ID: %03d Temperature is: %d degrees Celsius." % (id, temp))

    # ## Read the present loads/currents of all motors
    # loads, success = itemReadMultiple(all_ids, ADDR_PRESENT_LOAD, LEN_PRESENT_LOAD)
    # print(loads)

    portHandler.closePort()


if __name__ == "__main__":
    main()
