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
def syncWrite(groupSyncWrite, ids, commands, length=4):
    groupSyncWrite.clearParam()
    print(commands)
    for id, cmd in zip(ids, commands):
        param = []
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

        dxl_addparam_result = groupSyncWrite.addParam(id, param)

        if dxl_addparam_result != True:
            print("ID:%03d groupSyncWrite addparam failed" % id)
            return False

    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        return False
    return True


### @brief Reads data from a group of motors synchronously
### @param groupSyncRead - groupSyncRead object
### @param ids - list of Dynamixel IDs to read from
### @param address - register number
### @param length - size of register in bytes
### @return states - list to store the requested data
### @return <bool> - true if data was successfully retrieved; false otherwise
### @details - DynamixelSDK uses 2's complement so we need to check to see if 'state' should be negative (hex numbers)
def syncRead(groupSyncRead, ids, address, length=4):
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
    PWM = theta * int((DXL_MAXIMUM_POSITION_VALUE - DXL_MINIMUM_POSITION_VALUE) / 360)
    return PWM


def PWM2angle(PWM):
    angle = int(PWM * 360 / (DXL_MAXIMUM_POSITION_VALUE - DXL_MINIMUM_POSITION_VALUE))
    return angle


def home_limbs(groupSyncWrite, ids):
    home_positions = ([0] * len(ids)).astype(int)
    syncWrite(groupSyncWrite, ids, home_positions)


def home_trunk(groupSyncWrite, ids):
    home_positions = angle2PWM([180] * len(ids))
    syncWrite(groupSyncWrite, ids, home_positions)


def side_bending(groupSyncRead, groupSyncWrite, ids, goal_angle, previous_angle):
    gear_ratio = 30 / 20

    previous_angle = np.asarray(previous_angle) * gear_ratio
    goal_angle = np.asarray(goal_angle) * gear_ratio

    present_position, success = syncRead(
        groupSyncRead, ids, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
    )

    angle1 = present_position[0] + angle2PWM(abs(previous_angle[0] - goal_angle))
    angle2 = present_position[1] - angle2PWM(abs(previous_angle[1] - goal_angle))

    goal = [angle1, angle2]
    syncWrite(groupSyncWrite, ids, goal)

    return goal_angle


def rotation(groupSyncRead, groupSyncWrite, ids, goal_angle, previous_angle):
    gear_ratio = 30 / 20

    previous_angle = np.asarray(previous_angle) * gear_ratio
    goal_angle = np.asarray(goal_angle) * gear_ratio

    present_position, success = syncRead(
        groupSyncRead, ids, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
    )

    angle1 = present_position[0] + angle2PWM(abs(previous_angle - goal_angle))
    angle2 = present_position[1] + angle2PWM(abs(previous_angle - goal_angle))

    goal = [angle1, angle2]
    syncWrite(groupSyncWrite, ids, goal)

    return goal_angle


def main():
    ## Motor IDs
    limb_ids = [1, 2, 4, 5]
    diff_id = [65, 66]

    ## WX200 ARM EEPROM CONFIGS
    # all_ids = [1, 2, 3, 4, 65, 66]
    all_ids = diff_id
    drive_modes = [4] * len(all_ids)

    # Initial value for differential
    previous_angle = [180, 180]

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
    groupSyncWrite = GroupSyncWrite(
        portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
    )
    groupSyncRead = GroupSyncRead(
        portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
    )

    ## Initialize register values
    itemWriteMultiple(all_ids, ADDR_TORQUE_ENABLE, 0, LEN_TORQUE_ENABLE)
    itemWriteMultiple(all_ids, ADDR_DRIVE_MODE, drive_modes, LEN_DRIVE_MODE)
    itemWriteMultiple(all_ids, ADDR_PROFILE_VELOCITY, 1500, LEN_PROFILE_VELOCITY)
    itemWriteMultiple(all_ids, ADDR_PROFILE_ACCELERATION, 750, LEN_PROFILE_ACCELERATION)
    itemWriteMultiple(all_ids, ADDR_TORQUE_ENABLE, 1, LEN_TORQUE_ENABLE)

    ## Read current arm joint positions
    positions, success = syncRead(
        groupSyncRead, all_ids, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
    )
    speeds, success = syncRead(
        groupSyncRead, all_ids, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
    )
    print(positions, " ", speeds)

    ## Home all motors at the start
    # home_limbs(groupSyncWrite, limb_ids)
    home_trunk(groupSyncWrite, diff_id)
    time.sleep(2)

    # Test angles for differential
    angles = [90, -90, 45, -45]

    for theta in angles:
        print("previous angle= ", previous_angle)
        side_bending(groupSyncRead, groupSyncWrite, diff_id, theta, previous_angle)

    home_trunk(groupSyncWrite, diff_id)

    for theta in angles:
        previous_angle = rotation(
            groupSyncRead, groupSyncWrite, diff_id, theta, previous_angle
        )

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
