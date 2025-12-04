import numpy as np
import cv2 as cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import pickle


###Class for performing camera calibrations for PANDA Gym system
class calibrate:
    def __init__(self, video, lensID=None, FOV=None, vid_intrinsic=None):

        # Constant parameters used in Aruco methods
        ARUCO_PARAMETERS = aruco.DetectorParameters()
        self.ARUCO_PARAMETERS = ARUCO_PARAMETERS
        # ARUCO_PARAMETERS.markerBorderBits = 2
        # ARUCO_PARAMETERS.adaptiveThreshWinSizeStep = 5
        ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        self.ARUCO_DICT = ARUCO_DICT

        # Create grid board object we're using in our stream
        self.aruco_board = aruco.GridBoard(
            (5, 7),
            markerSeparation=0.008,
            markerLength=0.03,
            dictionary=ARUCO_DICT,
        )
        # with open(folder + "\extrinsic_params.json") as f:
        #     self.extrinsics = json.load(f)

        # Getting camera intrinsics
        if vid_intrinsic == None:
            with open(
                r"C:\Users\franc\Documents\GitHub\PANDA-Gym-Data-Processing\Calibration\intrinsic_params_all.pkl",
                "rb",
            ) as f:
                intrinsic_dict = pickle.load(f)

            if FOV == "None":
                FOV = "regular"

            # print(FOV)

            if FOV == "Super View":
                FOV = "super"

            intrinsics_0 = intrinsic_dict[lensID, FOV.lower(), 0]
            intrinsics_90 = intrinsic_dict[lensID, FOV.lower(), 90]
            intrinsics_180 = intrinsic_dict[lensID, FOV.lower(), 180]
            intrinsics_270 = intrinsic_dict[lensID, FOV.lower(), 270]

            self.intrinsics = [intrinsics_0, intrinsics_90, intrinsics_180, intrinsics_270]
        else:
            self.intrinsics_final = vid_intrinsic

        self.video = video

    def pull_frames_static(self, video):
        # reading in video file
        cap = cv2.VideoCapture(video)

        # getting number of video frames to find middle frame number
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # cam_angle.append(rot)
        frame = int(length / 2)

        # pulling middel frame from vidoe
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, image = cap.read()

        return image

    def intrinsics_orientation(self, view=0):
        rot = [0, 90, 180, 270]

        nums = np.zeros(4)
        error = np.zeros(4)
        extrinsics = []

        image = self.pull_frames_static(self.video)
        image1 = self.pull_frames_static(self.video)
        image2 = self.pull_frames_static(self.video)
        image3 = self.pull_frames_static(self.video)

        for i in range(4):
            # image = self.image
            if i == 1:
                image = image1
            elif i == 2:
                image = image2
            elif i == 3:
                image = image3
            else:
                image = image
            # nums[i] = self.check_ids(i, image, intrinsic)
            ex, error[i] = self.calibrate_extrinsics(image, self.intrinsics[i], view)
            # print("EXTRINSICS: ", rot[i], "  ", ex)
            extrinsics.append(ex)

        # print(nums)
        # loc = np.argmax(nums)
        loc = np.argmin(error)
        # loc = 0

        # print(error)
        # print(extrinsics[loc])
        self.extrinsics_final = extrinsics[loc]
        self.intrinsics_final = self.intrinsics[loc]

        return rot[loc], extrinsics[loc], self.intrinsics[loc]

    def calibrate_extrinsics_chess(self, image, intrinsics, view=0):

        cam_int = intrinsics

        cv2.namedWindow("ProjectImage", cv2.WINDOW_NORMAL)

        # print("INTRINSICS ", cam)
        cameraMatrix = np.asmatrix(cam_int["Mat"])
        distCoeffs = np.asarray(cam_int["Dist"])

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        img = image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            ProjectImage = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)

        # cv2.undistort(img, mtx, dist)

        ret, mtx, dist, Rvec, Tvec = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs
        )

        Rvec = Rvec[0]
        Tvec = Tvec[0]

        cam_ex = {"Rvec": Rvec, "Tvec": Tvec}
        rerror = None

        # print(rvecs, tvecs)

        # ProjectImage = cv2.drawFrameAxes(ProjectImage, cameraMatrix, distCoeffs, rvec=rvecs, tvec=tvecs, length=0.2)

        if view == 1:
            cv2.imshow("ProjectImage", ProjectImage)
            # print(cam_ex)

            # Exit at the end of the video on the 'q' keypress
            if cv2.waitKey(0) & 0xFF == ord("q"):
                print("exiting")

        return cam_ex, rerror

    def calibrate_extrinsics(self, image, intrinsics, view=0):
        ARUCO_PARAMETERS = self.ARUCO_PARAMETERS
        ARUCO_DICT = self.ARUCO_DICT

        cv2.namedWindow("ProjectImage", cv2.WINDOW_NORMAL)

        # Create grid board object we're using in our stream
        aruco_board = self.aruco_board

        # i=2
        cam_int = intrinsics
        # print("INTRINSICS ", cam)
        cameraMatrix = np.asmatrix(cam_int["Mat"])
        distCoeffs = np.asarray(cam_int["Dist"])

        ProjectImage = image

        # camfile=picfolder+'\\cam '+str(i+1)+'.jpg'

        # ProjectImage=cv2.imread(camfile)

        # grayscale image
        gray = cv2.cvtColor(ProjectImage, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

        # Refine detected markers
        # Eliminates markers not part of our board, adds missing markers to the board

        corners, ids, rejectedImgPoints, recoveredIds = detector.refineDetectedMarkers(
            image=gray,
            board=aruco_board,
            detectedCorners=corners,
            detectedIds=ids,
            rejectedCorners=rejectedImgPoints,
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeffs,
        )

        ## Checking for case of duplicate IDs found(means other tags are visible)
        u, c = np.unique(ids, return_counts=True)
        dup = u[c > 1]
        if (c > 1).any:
            print("Duplicate IDs found: ", dup)
            mean_pos = np.mean(np.mean(corners, axis=0), axis=1)[0]

            for r in dup:
                # locations of specific id number in ID and thus corners matrices
                idx = np.where(ids == r)[0]

                # findind center of all tag sqaures
                corner_mean = np.mean(corners, axis=2)[idx, 0, :]
                # distance from tag centers to mean corner location
                dists = np.linalg.norm(corner_mean - mean_pos, axis=1)
                # index of duplicate id that's furthest from mean position
                furthest = np.where(dists == max(dists))[0]
                idx_far = idx[furthest]
                # deleteing row with furthest mean position
                corners = np.delete(corners, idx_far, axis=0)
                ids = np.delete(ids, idx_far, axis=0)

        # Outline all of the markers detected in our image
        # Uncomment below to show ids as well
        ProjectImage = aruco.drawDetectedMarkers(ProjectImage, corners, borderColor=(0, 0, 255))
        # print(corners)

        cam_ex_error = {"Rvec": np.zeros((3, 1)), "Tvec": np.zeros((3, 1))}

        if ids is None:
            num_ids = 0
            # print(num_ids)
            return cam_ex_error, 100000
        elif len(ids) < 4:
            num_ids = len(ids)
            # print(num_ids)
            return cam_ex_error, 100000

        num_ids = len(ids)

        OBJpts = aruco_board.getObjPoints()

        OBJpts_sq, IMGpts_sq = self.biggest_rectangle(ids, OBJpts, corners)
        # print(rect_corners(ids))

        # id_bool = biggest_square(ids, OBJpts,IMGpts)

        # success,Rvec,Tvec,rerror=cv2.solvePnPGeneric(OBJpts, IMGpts, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE)
        success, Rvec, Tvec, rerror = cv2.solvePnPGeneric(OBJpts_sq, IMGpts_sq, cameraMatrix, distCoeffs)
        Rvec = Rvec[0]
        Tvec = Tvec[0]
        rerror = rerror[0][0]

        OBJpts, IMGpts = self.matchIDandpts(ids, OBJpts, corners)

        # print(ids)

        # success,Rvec,Tvec=cv2.solvePnP(OBJpts, IMGpts, cameraMatrix, distCoeffs, Rvec,Tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE   )

        blank_img = np.zeros((500, 500, 3), np.uint8)
        ArucoPts = self.showobjpts(blank_img, OBJpts)
        ArucoPts = self.showobjpts(ArucoPts, OBJpts_sq, 255)

        n = int(len(IMGpts) / 4)
        imgpts_RPJ = np.zeros((4, n, 2))

        OBJpts = np.reshape(OBJpts, (4, n, 3))
        for z in range(n):
            # print(OBJpts[z].shape)
            imgpts_RPJ[:, [z], :], _ = cv2.projectPoints(OBJpts[:, [z], :], Rvec, Tvec, cameraMatrix, distCoeffs)

        # RPJ=np.reshape(imgpts_RPJ,(4, len(ids)))

        ProjectImage = cv2.drawFrameAxes(ProjectImage, cameraMatrix, distCoeffs, Rvec, Tvec, 0.2)
        ProjectImage = self.ShowReproj(ProjectImage, imgpts_RPJ, n)
        # ProjectImage=ShowRect(ProjectImage,IMGpts_sq)

        imgpts_RPJ = np.reshape(imgpts_RPJ, (4 * n, 2))

        cam_ex = {"Rvec": Rvec, "Tvec": Tvec}

        if view == 1:
            cv2.namedWindow("ProjectImage", cv2.WINDOW_NORMAL)
            cv2.namedWindow("ArucoPts", cv2.WINDOW_NORMAL)

            cv2.imshow("ProjectImage", ProjectImage)
            cv2.imshow("ArucoPts", ArucoPts)
            # print(cam_ex)

            # Exit at the end of the video on the 'q' keypress
            if cv2.waitKey(0) & 0xFF == ord("q"):
                print("exiting")
        # cv2.destroyAllWindows()

        return cam_ex, rerror

    # Calibrating intrinscs based on folder of checkerboard images
    def calibrate_intrinsics(self, camfolder):
        print("Calibratinf Intrinsic Paramters")
        boardsize = (8, 6)
        cnt = 0
        tot = 0

        # cv.namedWindow("ProjectImage", cv.WINDOW_NORMAL)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((6 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        for camfile in os.listdir(camfolder):
            if camfile.endswith(".JPG"):
                tot = tot + 1

                img = cv2.imread(camfolder + "\\" + camfile)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                retval, corners = cv2.findChessboardCorners(gray, (boardsize[0], boardsize[1]))
                # print(camfile, retval)

                corners = np.squeeze(corners)

                if retval == True:
                    cnt = cnt + 1

                    objpoints.append(objp)

                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)

        print(cnt, "/", tot, " valid pictures")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("Matrix: ", mtx)
        print("Distance: ", dist)

        return mtx, dist

    def biggest_rectangle(self, ids, objpts, corners):
        # arrange first row of number changing by 24
        row1 = np.arange(6, 104, 24, dtype=int)
        # instatitating IDs matrix
        id_mat = np.zeros((7, 5))

        # Filling in IDs matrix with correct numbers
        for i in range(7):
            id_mat[i, :] = row1 - i
        id_mat_bool = np.zeros_like(id_mat)

        # Bulidng boolen matrix of found IDs
        for a in range(7):
            for b in range(5):
                if any(id_mat[a, b] == ids):
                    id_mat_bool[a, b] = 1

        # Finding biggest recatngle corners
        # If all 35 IDs found, biggets rectangle corners are 4 corners of board
        if len(ids) == 35:
            found = [6, 102, 96, 0]

        # If less than 35 IDs, use biggets rectangle algorithm
        else:
            found = self.rect_corners(id_mat, id_mat_bool)

        objpts2 = []
        objpts3 = []
        corners2 = []
        corners3 = []
        # print(corners)

        # Flattening id matrices
        IDs = id_mat.flatten()
        ids = ids.flatten()

        # finding onject points and image points of 4 corner IDs
        for j in range(4):
            # find where in full id matrix corner ID is
            loc = np.where(IDs == found[j])[0]
            # find object points for corner ID
            objpts3.append(objpts[loc[0]][j])

            # It's rotated ?
            jdx = j - 1
            if j == 0:
                jdx = 3

            # find where in input/found ids matrix, corner ID is
            loc2 = np.where(ids == found[j])[0]
            # find camera point of corner ID
            # print(loc2)
            cor = corners[loc2[0]]
            corners3.append(cor[-1, jdx, :])

        return (np.array(objpts3)), np.array(corners3)

    def rect_corners(self, id_mat, id_mat_bool):
        rows = 7
        cols = 5

        pairs = np.matrix([0, 0])
        p_rows = [100]

        # initalizing variable for max area and location of max area corners
        maxarea = 0
        tl = 0
        tr = 0
        bl = 0
        br = 0

        # iterating by row
        for i in range(rows):
            # print("Row: ",i+1)
            # iterating by line
            for j in range(cols - 1):
                # if 1 found in cell, check other columns for 1s
                if id_mat_bool[i][j] == 1:
                    # checking every column after a 1 for more ones
                    for k in range(j + 1, cols):
                        # if found two 1's in a row
                        if id_mat_bool[i][k] == 1:
                            idx = np.array([j, k])
                            # if row is already one of those stored in pairs we have a rectangle
                            if ((pairs == idx).all(1)).any():
                                # find length of rectangle with left and right indeces
                                length = k - j + 1

                                # find index of stored row
                                idx_top = np.where((pairs == idx).all(1))[0]
                                # use index found above to pull row number of stored row
                                top = p_rows[idx_top[0]]
                                # calulate width with top and bottom indeces
                                width = i - top + 1

                                area = length * width
                                # check if area is bigger than max area
                                if area > maxarea:
                                    # update max area and tag numbers at those corners
                                    maxarea = area
                                    tl = int(id_mat[top, j])
                                    tr = int(id_mat[top, k])
                                    bl = int(id_mat[i, j])
                                    br = int(id_mat[i, k])
                            else:

                                pairs = np.vstack((pairs, idx))
                                p_rows.append(i)

        # print("Area: ",maxarea)
        # needs to be rreturned in this specific order (clockwise from top right)
        found = [tl, tr, br, bl]
        return found

    def matchIDandpts(self, ids, objpts, corners):
        # objpts is the board object points in row by row order for all 35 markers
        # want output objpts2 to be oject points for identified markers in corner/id order
        row1 = np.arange(6, 104, 24, dtype=int)
        IDs = np.zeros((7, 5))
        for i in range(7):
            IDs[i, :] = row1 - i
        # IDs=IDs.T

        IDs = IDs.flatten()
        ids = ids.flatten()

        objpts2 = []

        n = len(ids)

        for j in range(n):
            # print(ids[j])
            loc = np.where(IDs == ids[j])[0]
            objpts2.append(objpts[loc[0]])

        # print(len(objpts2),len(objpts))
        objpts2 = np.array(objpts2)

        objpts2 = np.reshape(objpts2, (4, n, 3))
        objpts2 = np.reshape(objpts2, (4 * n, 3))

        imgpts = np.reshape(corners[0:n], (4, n, 2))
        imgpts = np.reshape(imgpts, (4 * n, 2))

        return objpts2, imgpts

    def showobjpts(self, img, objpts, clr=0):
        objpts = (objpts * 1000) + 150

        n, a = objpts.shape

        for i in range(n):
            point = objpts[i, 0:2].astype(int)
            # print(point[0:2])
            img = cv2.circle(img, tuple(point), 2, 255, -1)

        for j in range(0, n, 4):
            strt = tuple(objpts[j, 0:2].astype(int))
            edpt = tuple(objpts[j + 2, 0:2].astype(int))

            img = cv2.rectangle(img, strt, edpt, (0, clr, 255), 1)

        return img

    def ShowReproj(self, ProjectImage, imgpts_RPJ, n):
        height, width = ProjectImage.shape[:2]
        # print("height and width", height, width)
        for i in range(n):
            for j in range(4):
                point = imgpts_RPJ[j, i, :].astype(int)

                # Conditional for when found image points are outside image bounds and is a positive nu,ber
                if (width >= point[0]) and (height >= point[1]) and (0 <= point[0]) and (0 <= point[1]):
                    # print("point", point)
                    ProjectImage = cv2.circle(ProjectImage, tuple(point), 2, 255, -1)
                else:
                    print("point outisde bounds of image")

        return ProjectImage
