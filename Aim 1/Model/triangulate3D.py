# import sba
import json
import itertools
import pandas as pd
import numpy as np
import cv2 as cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display
from ProcessPose import processpose
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

# NEEDS A LOT OF REWRITING TO BE ROBUTS TO MISSING KEYPOINTS AND CAMERAS


class tringulatepose:
    def __init__(self):
        pfloder = r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Trials\Pose Estimates\3D Test\unsmoothed"
        folder = r"C:\Users\franc\Documents\GitHub\PANDA-Gym-Data-Proceeing\Calibration"

        # rot = np.array([0, 180, 0, 0, 90])
        # rot = np.array([0, 180, 0, 0, 90, 0, 90])

        # cams = np.array([1, 2, 3, 4, 5])
        cams = np.array([1, 2, 3, 4, 5, 6, 7])

        start = [82, 81, 42, 11, 12, 9, 10]

        self.cams = cams

        n_cams = len(cams)
        self.n_cams = n_cams

        cam_objects = []

        n = 100000000

        for i in range(n_cams):
            print("Processing camera ", i + 1)
            # print(cams[i])
            cam = processpose(pfloder + "\\25_01_31_833180_108_cam" + str(cams[i]) + "_vid4.csv", start[i])
            cam_objects.append(cam)

            if cam.fps == 60:
                X = cam.Xfilt[:, 0:-1:2]
            elif cam.fps == 30:
                X = cam.Xfilt

            if X.shape[1] < n:
                n = int(X.shape[1])

        self.cam_objects = cam_objects

        XA = np.zeros((n_cams, 18, n))
        YA = np.zeros((n_cams, 18, n))
        IA = np.zeros((n_cams, 18, n))
        CA = np.zeros((n_cams, 18, n))
        conf = np.zeros((n_cams, 18, n))

        # for cam in cam_objects:
        for j in range(n_cams):
            cam = cam_objects[j]
            if cam.fps == 60:
                X, Y, I, C = cam.Xfilt[:, 0:-1:2], cam.Yfilt[:, 0:-1:2], cam.idx[:, 0:-1:2], cam.conf[:, 0:-1:2]
            elif cam.fps == 30:
                X, Y, I, C = cam.Xfilt, cam.Yfilt, cam.idx, cam.conf

            d = 30

            # if cams[j] == 1 | cams[j] == 2 | cams[j] == 3:
            #     XA[j, :, :], YA[j, :, :], IA[j, :, :], conf[j, :, :] = (
            #         X[:, d : n + d],
            #         Y[:, d : n + d],
            #         I[:, d : n + d],
            #         C[:, d : n + d],
            #     )
            # else:

            XA[j, :, :], YA[j, :, :], IA[j, :, :], conf[j, :, :] = X[:, 0:n], Y[:, 0:n], I[:, 0:n], C[:, 0:n]

            CA[j, :, :] = np.ones((18, n)) * (cams[j] - 1)

        self.XA = XA
        self.YA = YA
        self.conf = conf

        self.pt2D = np.vstack((self.XA.ravel(), self.YA.ravel())).T  # (2,(7*13*3626))
        self.ptidx = IA.ravel()
        self.camidx = CA.ravel()

        with open(folder + "\\intrinsic_params.json") as f:
            self.intrinsics = json.load(f)

        with open(folder + "\\extrinsic_params.json") as f:
            self.extrinsics = json.load(f)

        # n = 100
        self.n = n
        self.t = np.linspace(0, self.n / 30, num=self.n)

        # XYZ corridnates for current points (changes during bundle adjustmnent)
        self.X = np.zeros((n, 18))
        self.Y = np.zeros((n, 18))
        self.Z = np.zeros((n, 18))
        # XYZ coordinated for original points
        self.X0 = np.zeros((n, 18))
        self.Y0 = np.zeros((n, 18))
        self.Z0 = np.zeros((n, 18))
        # print(n)

        intrinsics = self.intrinsics
        extrinsics = self.extrinsics

        # dim1= cam#, dim2= Tvec or Rvec, dim3= X,Y,Z
        ext_params = np.zeros((n_cams, 2, 3))
        # int_params

        for k in range(n_cams):
            num = cams[k]

            # cam_int = intrinsics["cam" + str(num)]
            cam_ext = extrinsics["cam" + str(num)]

            # cam_mat = np.asmatrix(cam_int["Mat"])
            # cam_dist = np.asarray(cam_int["Dist"])

            cam_rvec = np.asmatrix(cam_ext["Rvec"])
            cam_tvec = np.asmatrix(cam_ext["Tvec"])

            ext_params[k, :, :] = np.vstack((cam_rvec.T, cam_tvec.T))  # (7,2,3)

        self.ext_params = ext_params

        self.combos = list(itertools.combinations(cams, 2))

        # print("XA", self.XA[1, 0, 0:26])
        # print("YA", self.YA[0, 0:5, 0:5])
        # print("-------------------------")
        # print(pt[:, n * 13 : n * 13 + 26])
        # print("-------------------------")
        # print(self.XA[1, :, 0:5])

        # all X for kp1, cam1, then kp2, cam1 ..... then kp1, cam 2

    def trinagulate_each(self, pt1, pt2, num1, num2):
        n = self.n

        intrinsics = self.intrinsics
        extrinsics = self.extrinsics

        cam1_int = intrinsics["cam" + str(num1)]
        cam2_int = intrinsics["cam" + str(num2)]

        cam1_mat = np.asmatrix(cam1_int["Mat"])
        cam1_dist = np.asarray(cam1_int["Dist"])
        cam2_mat = np.asmatrix(cam2_int["Mat"])
        cam2_dist = np.asarray(cam2_int["Dist"])

        cam1_ext = extrinsics["cam" + str(num1)]
        cam2_ext = extrinsics["cam" + str(num2)]

        cam1_rvec = np.asmatrix(cam1_ext["Rvec"])
        cam1_tvec = np.asmatrix(cam1_ext["Tvec"])
        cam2_rvec = np.asmatrix(cam2_ext["Rvec"])
        cam2_tvec = np.asmatrix(cam2_ext["Tvec"])

        cam1_rot, _ = cv2.Rodrigues(cam1_rvec)
        cam2_rot, _ = cv2.Rodrigues(cam2_rvec)

        cam1_proj = np.hstack((cam1_rot, cam1_tvec.reshape(-1, 1)))
        cam2_proj = np.hstack((cam2_rot, cam2_tvec.reshape(-1, 1)))

        # print(np.shape(pt1))
        # print(np.shape(pt2))

        pt1 = cv2.undistortPoints(pt1, cam1_mat, cam1_dist)
        pt2 = cv2.undistortPoints(pt2, cam2_mat, cam2_dist)

        homogeneous_points = cv2.triangulatePoints(cam1_proj, cam2_proj, pt1, pt2)
        points_3d = cv2.convertPointsFromHomogeneous(homogeneous_points.T)

        # t = np.linspace(0, 7360, 7361) / 60

        X = points_3d[:, :, 0]
        Y = points_3d[:, :, 1]
        Z = points_3d[:, :, 2]

        return X, Y, Z

    def trinagulat_all(self, num1, num2):
        self.num1 = num1
        self.num2 = num2

        idx1 = np.where(self.cams == num1)[0][0]
        idx2 = np.where(self.cams == num2)[0][0]

        X1, Y1 = self.XA[idx1, :, :], self.YA[idx1, :, :]
        X2, Y2 = self.XA[idx2, :, :], self.YA[idx2, :, :]

        for i in range(18):

            pt_c1 = np.vstack((X1[i, :], Y1[i, :]))
            pt_c2 = np.vstack((X2[i, :], Y2[i, :]))

            x, y, z = self.trinagulate_each(pt_c1, pt_c2, num1, num2)

            self.X[:, i] = x.ravel()
            self.Y[:, i] = y.ravel()
            self.Z[:, i] = z.ravel()

            # print(i)

        # Updating dynmic X,Y,Z with first guess from trinagulation
        # self.X = self.X0
        # self.Y = self.Y0
        # self.Z = self.Z0

        # print(np.shape(self.X))  # N,13
        # pt = np.vstack((self.X.T.ravel(), self.Y.T.ravel(), self.Z.T.ravel()))  # (3,(13*3626))
        # 13, N

        # all X for kp1, then kp2,... kp13
        # print("XA", self.X.T[1, 0:26])
        # print("YA", self.Y.T[0, 0:26])
        # print("ZA", self.Z.T[0, 0:26])
        # print("-------------------------")
        # print(pt[0, self.n : self.n + 26])

    def trinagulat_i(self, num1, num2, j):
        num1 = self.num1
        num2 = self.num2

        X1, Y1 = self.XA[num1 - 1, :, :], self.YA[num1 - 1, :, :]
        X2, Y2 = self.XA[num2 - 1, :, :], self.YA[num2 - 1, :, :]

        X = np.zeros((18, 1))
        Y = np.zeros((18, 1))
        Z = np.zeros((18, 1))

        for i in range(18):

            pt_c1 = np.vstack((X1[i, j], Y1[i, j]))
            pt_c2 = np.vstack((X2[i, j], Y2[i, j]))

            x, y, z = self.trinagulate_each(pt_c1, pt_c2, num1, num2)

            X[i] = x
            Y[i] = y
            Z[i] = z

        return X, Y, Z

    def trinagulat_conf(self):
        # X1, Y1 = self.XA[num1 - 1, :, :], self.YA[num1 - 1, :, :]
        # X2, Y2 = self.XA[num2 - 1, :, :], self.YA[num2 - 1, :, :]
        conf = self.conf
        # print(np.shape(conf))
        avg_c = np.mean(conf, axis=2)

        # c = avg_c[:, i]
        # # print(c)
        # # print(cc)
        # c_org = np.sort(c)

        # num1 = np.argwhere(c == c_org[-3])[0][0] + 1
        # num2 = np.argwhere(c == c_org[-2])[0][0] + 1
        n = self.n
        perm_list = np.zeros((2, n))

        for i in range(n):
            Err = np.zeros(21)

            for j in range(21):
                # j = 5
                # i = 2000
                p = self.perm[j]
                num1, num2 = p[0], p[1]

                # print(j, [num1, num2])

                X, Y, Z = self.trinagulat_i(num1, num2, i)
                E = self.reproj_i(X, Y, Z, i)

                Err[j] = E

            M = min(Err)
            loc = np.argwhere((Err == M))[0][0]

            P = self.perm[loc]
            N1, N2 = P[0], P[1]

            x, y, z = self.trinagulat_i(N1, N2, i)
            self.X[i, :], self.Y[i, :], self.Z[i, :] = x.ravel(), y.ravel(), z.ravel()

        # print(i)
        print("done")
        # Updating dynmic X,Y,Z with first guess from trinagulation

    def reproj_i(self, X, Y, Z, i):
        n = self.n

        intrinsics = self.intrinsics

        pts = np.vstack((X.T, Y.T, Z.T))

        tot_error = np.zeros((7))

        for j in range(7):
            num = 1 + j
            cam_int = intrinsics["cam" + str(num)]

            cam_mat = np.asmatrix(cam_int["Mat"])
            cam_dist = np.asarray(cam_int["Dist"])

            cam_ext = self.ext_params[num - 1, :]  # (7,2,3)

            cam_rvec = cam_ext[0, :]
            cam_tvec = cam_ext[1, :]

            # print(pts)
            R, _ = cv2.projectPoints(pts, cam_rvec, cam_tvec, cam_mat, cam_dist)
            R = R.reshape(-1, 2).T

            Xreproj = R[0, :]
            Yreproj = R[1, :]

            rpt = np.vstack((Xreproj, Yreproj))
            pt = np.vstack((self.XA[num - 1][:, i], self.YA[num - 1][:, i]))

            error = np.linalg.norm(rpt - pt, axis=0)

            tot_error[j] = np.mean(error)

        return np.mean(tot_error)

    def plotskel(self, j):

        colors = [
            [255, 0, 0],
            [255, 170, 0],
            [255, 255, 0],
            [255, 85, 0],
            [170, 255, 0],
            [85, 255, 0],
            [0, 255, 0],
            [0, 255, 85],
            [0, 255, 170],
            [0, 255, 255],
            [0, 170, 255],
            [0, 85, 255],
            [0, 0, 255],
            [170, 0, 255],
            [255, 0, 255],
            [85, 0, 255],
            [85, 85, 255],
        ]

        # limbSeq = np.matrix(
        #     [[1, 0], [2, 1], [3, 2], [4, 0], [5, 4], [6, 5], [7, 0], [8, 7], [9, 8], [10, 0], [11, 10], [12, 11]]
        # )
        limbSeq = np.matrix(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [1, 5],
                [5, 6],
                [6, 7],
                [1, 8],
                [8, 9],
                [9, 10],
                [1, 11],
                [11, 12],
                [12, 13],
                [0, 14],
                [14, 16],
                [0, 15],
                [15, 17],
            ]
        )

        x = self.X[j, :]
        y = -self.Y[j, :]
        z = -self.Z[j, :]

        plt.cla()

        for p in range(0, 17):

            plt.plot(
                [x[limbSeq[p, 0]], x[limbSeq[p, 1]]],
                [y[limbSeq[p, 0]], y[limbSeq[p, 1]]],
                [z[limbSeq[p, 0]], z[limbSeq[p, 1]]],
                color=np.array(colors[p]) / 255,
            )
            plt.plot(
                [x[limbSeq[p, 0]], x[limbSeq[p, 1]]],
                [y[limbSeq[p, 0]], y[limbSeq[p, 1]]],
                [z[limbSeq[p, 0]], z[limbSeq[p, 1]]],
                "o",
            )

        s = "t= " + str(int(j / 30))

        self.ax.text(0, 0, 0, "%s" % (s), size=20, zorder=1, color="k")

        # plt.text(0, 0, s)
        self.ax.set_zlim3d(-0.1, 0.8)
        self.ax.set_ylim3d(-0.5, 0.1)
        self.ax.set_xlim3d(-0.3, 0.3)
        plt.grid()

    def plotskel_loop(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        ani = animation.FuncAnimation(self.fig, self.plotskel, frames=self.n, interval=2)
        # ani.save(filename=r"C:\Users\franc\Documents\Infant_Sim_data\load tests\side_bend_35.gif", writer="pillow")
        plt.show()

    def plotskel2D(self, j, camnum, frame):
        colors = [
            [255, 0, 0],
            [255, 170, 0],
            [255, 255, 0],
            [255, 85, 0],
            [170, 255, 0],
            [85, 255, 0],
            [0, 255, 0],
            [0, 255, 85],
            [0, 255, 170],
            [0, 255, 255],
            [0, 170, 255],
            [0, 85, 255],
            [0, 0, 255],
            [170, 0, 255],
            [255, 0, 255],
            [85, 0, 255],
            [85, 85, 255],
        ]
        # colors = [
        #     [255, 0, 0],
        #     [255, 170, 0],
        #     [255, 255, 0],
        #     [255, 85, 0],
        #     [170, 255, 0],
        #     [85, 255, 0],
        #     [0, 255, 0],
        #     [0, 255, 85],
        #     [0, 255, 170],
        #     [0, 255, 255],
        #     [0, 170, 255],
        #     [0, 85, 255],
        # ]

        # limbSeq = np.matrix(
        #     [[1, 0], [2, 1], [3, 2], [4, 0], [5, 4], [6, 5], [7, 0], [8, 7], [9, 8], [10, 0], [11, 10], [12, 11]]
        # )
        limbSeq = np.matrix(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [1, 5],
                [5, 6],
                [6, 7],
                [1, 8],
                [8, 9],
                [9, 10],
                [1, 11],
                [11, 12],
                [12, 13],
                [0, 14],
                [14, 16],
                [0, 15],
                [15, 17],
            ]
        )
        x, y = self.XA[camnum - 1, :, j], self.YA[camnum - 1, :, j]

        # print(np.shape(x))

        x2 = np.asarray(x).reshape(-1)
        y2 = np.asarray(y).reshape(-1)

        # plt.cla()

        for p in range(17):

            # Plotting skeleton for original pose
            frame = cv2.line(
                frame,
                (int(x2[limbSeq[p, 0]]), int(y2[limbSeq[p, 0]])),
                (int(x2[limbSeq[p, 1]]), int(y2[limbSeq[p, 1]])),
                color=colors[p],
                thickness=7,
            )

            frame = cv2.circle(
                frame,
                (int(x2[limbSeq[p, 0]]), int(y2[limbSeq[p, 0]])),
                radius=9,
                color=(0, 0, 0),
                thickness=7,
            )

        return frame

    def plotskel2D_reproj(self, j, camnum, frame):

        # limbSeq = np.matrix(
        #     [[1, 0], [2, 1], [3, 2], [4, 0], [5, 4], [6, 5], [7, 0], [8, 7], [9, 8], [10, 0], [11, 10], [12, 11]]
        # )

        limbSeq = np.matrix(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [1, 5],
                [5, 6],
                [6, 7],
                [1, 8],
                [8, 9],
                [9, 10],
                [1, 11],
                [11, 12],
                [12, 13],
                [0, 14],
                [14, 16],
                [0, 15],
                [15, 17],
            ]
        )

        X, Y, E = self.reproj_each(camnum)

        x = X[:, j].T
        y = Y[:, j].T

        # print(np.shape(x))

        x1 = np.asarray(x).reshape(-1)
        y1 = np.asarray(y).reshape(-1)

        # grayscale image
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for p in range(17):
            # Plotting skeleton for reprojection
            frame = cv2.line(
                frame,
                (int(x1[limbSeq[p, 0]]), int(y1[limbSeq[p, 0]])),
                (int(x1[limbSeq[p, 1]]), int(y1[limbSeq[p, 1]])),
                color=(0, 0, 0),
                thickness=7,
            )

            frame = cv2.circle(
                frame,
                (int(x1[limbSeq[p, 0]]), int(y1[limbSeq[p, 0]])),
                radius=9,
                color=(0, 0, 255),
                thickness=7,
            )

        return frame

    def overlay_pose(self, camnum, compare):
        # storing video rotation
        rot = np.array([0, 180, 0, 0, -90, 0, -90])

        filename = (
            r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Trials\Aim I\833180_108\01-31-2025\Cameras\Camera "
            + str(camnum)
            + "\\2025_01_31_833180_108_cam"
            + str(camnum)
            + "_vid4.mp4"
        )
        # read in video
        cap = cv2.VideoCapture(filename)

        # OPENCV_FFMPEG_READ_ATTEMPTS (current value is 4096
        width = cap.get(3)  # float `width`
        height = cap.get(4)  # float `height`

        j = 0

        # print(self.cam_objects[camnum - 1].start)
        while cap.isOpened():
            # variable to skip frames every other frame, remains true if video at 30fps
            skip = True
            frameId = int(cap.get(1))

            # skip = self.cam_objects[camnum - 1].start < frameId

            # if 60fps video skip every other frame to diaplay video at 30fps
            # if self.cam_objects[camnum - 1].start < frameId:
            wait = self.cam_objects[camnum - 1].start < frameId

            if self.cam_objects[camnum - 1].fps == 60:
                skip = frameId % 2

            # Capturing each frame of our video stream
            ret, frame = cap.read()
            # frame = cv2.resize(frame, (700, 500))
            if skip and wait:
                if ret == True:
                    j = j + 1

                    # rotating video
                    if rot[camnum - 1] == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rot[camnum - 1] == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif rot[camnum - 1] == -90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    # plotting original camera pose
                    frame = self.plotskel2D(j, camnum, frame)

                    # displays reprojected pose if compare is set to 1
                    if compare == 1:
                        # plotting reprojected camera pose
                        frame = self.plotskel2D_reproj(j, camnum, frame)

                    # reszing display window to 1/3 original video size
                    frame = cv2.resize(frame, (int(width / 3), int(height / 3)))

                    cv2.imshow("ProjectImage", frame)

            # Exit at the end of the video on the 'q' keypress
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def threshold(self, camnum):
        # storing video rotation
        rot = np.array([0, 180, 0, 0, -90, 0, 90])

        filename = (
            r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Trials\Aim I\833180_108\01-31-2025\Cameras\Camera "
            + str(camnum)
            + "\\2025_01_31_833180_108_cam"
            + str(camnum)
            + "_vid4.mp4"
        )
        # read in video
        cap = cv2.VideoCapture(filename)

        # OPENCV_FFMPEG_READ_ATTEMPTS (current value is 4096
        width = cap.get(3)  # float `width`
        height = cap.get(4)  # float `height`

        j = 0

        means = np.zeros((5))
        M = 0
        while cap.isOpened():
            # variable to skip frames every other frame, remains true if video at 30fps

            # Capturing each frame of our video stream
            ret, frame = cap.read()
            # frame = cv2.resize(frame, (700, 500))
            if ret == True:
                j = j + 1
                # rotating video
                if rot[camnum - 1] == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rot[camnum - 1] == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rot[camnum - 1] == -90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.GaussianBlur(frame, (11, 11), 0)
                frame = cv2.threshold(frame, 250, 255, cv2.THRESH_BINARY)[1]

                # # contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # # print("Number of Contours Found = " + str(len(contours)))

                if j < 5:
                    means[j] = np.mean(frame)
                    M = np.mean(means)
                    print(M)
                elif np.mean(frame) > M * 1.5:
                    # print(np.mean(frame))
                    print(j)

                if np.mean(frame) > 0.005:
                    print(j)
                # print(np.mean(frame))
                # frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 25)

                # reszing display window to 1/3 original video size
                frame = cv2.resize(frame, (int(width / 3), int(height / 3)))

                cv2.imshow("ProjectImage", frame)

            # Exit at the end of the video on the 'q' keypress
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def reproj_each(self, num):
        intrinsics = self.intrinsics

        cam_int = intrinsics["cam" + str(num)]

        cam_mat = np.asmatrix(cam_int["Mat"])
        cam_dist = np.asarray(cam_int["Dist"])

        idx = np.where(self.cams == num)[0][0]

        cam_ext = self.ext_params[idx, :, :]  # (7,2,3)

        cam_rvec = cam_ext[0, :]
        cam_tvec = cam_ext[1, :]

        reproj = np.zeros((2, self.n, 18))

        for i in range(18):
            pts = np.vstack((self.X[:, i], self.Y[:, i], self.Z[:, i]))

            # reproj[:, :, [i]] = cv2.projectPoints(pts, cam_rvec, cam_tvec, cam_mat, cam_dist)
            R, _ = cv2.projectPoints(pts, cam_rvec, cam_tvec, cam_mat, cam_dist)
            R = R.reshape(-1, 2).T

            reproj[:, :, i] = R

        Xreproj = reproj[0, :, :]
        Yreproj = reproj[1, :, :]

        # errorX = np.zeros((13))
        # errorY = np.zeros((13))
        error = np.zeros((18))

        for j in range(18):
            rpt = np.vstack((Xreproj[:, j], Yreproj[:, j]))
            pt = np.vstack((self.XA[idx][j, :], self.YA[idx][j, :]))

            error[j] = np.linalg.norm(rpt - pt) / self.n
            # errorX[j] = np.linalg.norm(Xreproj[:, j] - self.XA[num - 1][j, :])
            # errorY[j] = np.linalg.norm(Xreproj[:, j] - self.XA[num - 1][j, :])

        mean_error = sum(error) / 18

        return Xreproj.T, Yreproj.T, mean_error

    def reproj_all(self):
        n = self.n
        ncams = self.n_cams

        XAr = np.zeros((ncams, 18, n))
        YAr = np.zeros((ncams, 18, n))
        E = np.zeros((18))

        for i in range(ncams):
            X, Y, e = self.reproj_each(self.cams[i])

            XAr[i, :, :] = X
            YAr[i, :, :] = Y
            E[i] = e

        return XAr, YAr, E

    def update3D(self, pts):
        n = self.n

        k = 18 * n

        x = np.zeros((k))
        y = np.zeros((k))
        z = np.zeros((k))

        for j in range(int(k)):
            x[j] = pts[j * 3]
            y[j] = pts[j * 3 + 1]
            z[j] = pts[j * 3 + 2]

        # x = pts[0:k]
        # y = pts[k : 2 * k]
        # z = pts[2 * k : 3 * k]

        # # print(np.shape(pts))

        for i in range(18):
            # print(np.shape(x[i * n : i * n + n]))
            # print(np.shape(self.X[:, i]))

            self.X[:, i] = x[i * n : i * n + n]
            self.Y[:, i] = y[i * n : i * n + n]
            self.Z[:, i] = z[i * n : i * n + n]

    def error_percent(self):
        percent = np.zeros((self.n_cams))
        thresh = 15

        for j in range(self.n_cams):
            X, Y, E = self.reproj_each(self.cams[j])
            rpj = np.vstack((X.ravel(), Y.ravel())).T

            xg, yg = self.XA[j, :, :], self.YA[j, :, :]

            pt = np.vstack((xg.ravel(), yg.ravel())).T

            rez = rpj - pt

            rez_e = np.linalg.norm(rpj - pt, axis=1)

            # l1, l2 = np.where(rez[:, 0] == np.max(rez[:, 0]))[0][0], np.where(rez[:, 1] == np.max(rez[:, 1]))[0][0]

            # s1, s2 = np.where(rez[:, 0] > thresh)[0], np.where(rez[:, 1] > thresh)[0]
            S = np.where(rez_e > thresh)[0]

            # print(rez[40000:40005, :])
            # print(j, np.max(rez[:, 0]), np.max(rez[:, 1]))
            # print(S)
            percent[j] = int(len(S) / len(rez) * 100)

        return percent

    def residual(self, x):
        n = self.n
        ncams = self.n_cams

        params = x[0 : 6 * ncams]
        pts3D = x[6 * ncams : len(x)]

        self.update3D(pts3D)

        # res=np.zeros(ncams*(n+6))
        res = np.zeros(ncams * n * 2 * 18)

        # Reforming and updating 3D parameters
        ext_params = np.zeros((ncams, 2, 3))
        # int_params

        for i in range(ncams):
            # pulling 2x3 extrinsics matrix
            cam_ext = np.reshape(params[i * 6 : i * 6 + 6], (2, 3))
            # putiing 2x3 matrices in big matrix
            ext_params[i, :, :] = cam_ext  # (7,2,3)

        self.ext_params = ext_params

        # updating extrinscs for cameras (golbally)

        # XAr, YAr, E = self.reproj_all()

        # pts_rpj = np.vstack((XAr.ravel(), YAr.ravel())).T

        # # res2 = np.hstack(((self.XA - XAr).ravel(), (self.YA - YAr).ravel()))
        # res = (pts_rpj - self.pt2D).ravel()
        m = 18 * n * 2

        for j in range(ncams):
            X, Y, E = self.reproj_each(self.cams[j])
            rpj = np.vstack((X.ravel(), Y.ravel())).T

            xg, yg = self.XA[j, :, :], self.YA[j, :, :]

            pt = np.vstack((xg.ravel(), yg.ravel())).T
            # print(len(pt))

            res[j * m : j * m + m] = (rpj - pt).ravel()

        # print(np.max(res))
        return res

    def show_reproj(self, name=None):
        # error = pd.DataFrame()  # an empty DataFrame
        ncams = self.n_cams
        error = np.zeros((ncams))

        folder = r"C:\Users\franc\Documents\GitHub\PANDA-Gym-Data-Proceeing\Calibration\compare"

        for i in range(ncams):
            camnum = self.cams[i]

            # fig = plt.figure()

            Xreproj, Yreproj, mean_error = self.reproj_each(camnum)
            # Create new rows as separate DataFrames

            error[i] = mean_error

            # t = self.t
            # fig2 = plt.figure
            # print(np.shape(Xreproj[:, 0]), np.shape((self.XA[camnum - 1][0, :])))
            j = camnum * 2 - 1

            t = np.linspace(0, Xreproj.size, num=Xreproj.size)

            plt.subplot(ncams, 2, j)
            plt.plot(t, Xreproj.ravel() - self.XA[i].ravel(), linewidth=0.75)
            # plt.plot(t, self.XA[i].ravel(), linewidth=0.75)
            # plt.fill_between(t, Xreproj.ravel(), self.XA[i].ravel(), color="grey", alpha=0.3)
            # plt.legend(["Reproj", "Orig"])
            plt.title("Camera " + str(camnum) + " Reproj X")

            plt.subplot(ncams, 2, j + 1)
            plt.plot(t, Yreproj.ravel() - self.YA[i].ravel(), linewidth=0.75)
            # plt.plot(t, self.YA[i].ravel(), linewidth=0.75)
            # plt.fill_between(t, Yreproj.ravel(), self.YA[i].ravel(), color="grey", alpha=0.3)
            # plt.legend(["Reproj", "Orig"])
            plt.title("Camera " + str(camnum) + " Reproj Y")

        if name and len(name) > 0:
            # print(name)
            plt.savefig(folder + "\\" + name + ".jpg", dpi=500)
            # print(folder + "\\" + name + ".jpg")
        else:
            plt.show()

        return error

    def jacob(self):
        ncams = self.n_cams

        #     num_observations = camera_inds.size
        # m = num_observations * 2  # total number of residuals (2 * num_observations)
        # n = num_cameras * 9 + num_points * 3  # total number of parameters (all camera parameters and locations of 3D points)
        # A = lil_matrix((m, n), dtype=int)

        # # Indicate non-zero relationships between each camera and its parameters (rotation, translation, focal length, distortion parameters) and 3D points
        # i = np.arange(num_observations)
        # for s in range(9):
        #     A[2 * i, camera_inds * 9 + s] = 1
        #     A[2 * i + 1, camera_inds * 9 + s] = 1

        # for s in range(3):
        #     A[2 * i, num_cameras * 9 + point_inds * 3 + s] = 1
        #     A[2 * i + 1, num_cameras * 9 + point_inds * 3 + s] = 1

        cameraIndices = self.camidx
        pointIndices = self.ptidx
        numCameras = ncams
        numPoints = 18 * self.n

        m = cameraIndices.size * 2
        n = (numCameras * 6) + (3 * numPoints)

        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        # print(i)

        # print(np.shape(A[2 * i, :]))
        # print(np.shape(A[2 * i, cameraIndices * 6]))
        # print(np.shape(cameraIndices))
        # print(np.shape(A[2 * i + 1, numCameras * 6 + pointIndices * 3]))
        # print(np.shape(A))

        for s in range(6):
            A[2 * i, cameraIndices * 6 + s] = 1
            A[2 * i + 1, cameraIndices * 6 + s] = 1

        for s in range(3):
            A[2 * i, numCameras * 6 + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * 6 + pointIndices * 3 + s] = 1

        # print("J")

        return A

    def SBA(self):
        # points_3d = np.vstack((self.X.T.ravel(), self.Y.T.ravel(), self.Z.T.ravel()))
        points_3d = np.vstack((self.X.T.ravel(), self.Y.T.ravel(), self.Z.T.ravel())).T

        x0 = np.hstack((self.ext_params.ravel(), points_3d.ravel()))

        f0 = self.residual(x0)

        # pts3D = x0[6 * 7 : len(x0)]
        # pts3D = points_3d.ravel()

        # k = 13 * self.n

        # x = pts3D[0:k]
        # y = pts3D[k : 2 * k]
        # z = pts3D[2 * k : 3 * k]

        # pts3Dt = points_3dt.ravel()

        # # xt = pts3Dt[0:k] * 0

        # X1 = self.Y
        # self.update3D(pts3D)
        # X2 = self.Y

        A = self.jacob()
        # print(np.any(np.isnan(f0)), np.any(np.isinf(f0)))

        print("Optimizing")

        # res = least_squares(self.residual, x0, loss="linear", jac_sparsity=A, verbose=2, ftol=1e-3)
        res = least_squares(
            self.residual,
            x0,
            jac_sparsity=A,
            verbose=2,
            x_scale="jac",
            ftol=1e-4,
            method="trf",
        )

        # print(res)

        # params = res.x[0 : 6 * 7]
        ff = res.fun

        # print(max(f0))
        # print(max(ff))

        plt.plot(f0)
        plt.plot(ff)
        plt.show()

        # print(self.ext_params)

    def check_combos(self):
        results = np.zeros((len(self.combos), 3))

        i = 0
        print("Checking camera combinations")
        for c in self.combos:

            self.trinagulat_all(c[0], c[1])
            # self.SBA()
            points_3d = np.vstack((self.X.T.ravel(), self.Y.T.ravel(), self.Z.T.ravel())).T

            x0 = np.hstack((self.ext_params.ravel(), points_3d.ravel()))

            f0 = self.residual(x0)
            results[i, :] = np.array([(c[0]), (c[1]), np.linalg.norm(f0) / 2])
            i = i + 1
        # print(np.argpartition(results[:, 2], 4))

        srt = np.argsort(results[:, 2])
        sorted = results[srt, :]

        num1 = int(sorted[0, 0])
        num2 = int(sorted[0, 1])

        print("1st Best reults from cameras: ", num1, "+", num2)
        print("2nd Best reults from cameras: ", int(sorted[1, 0]), "+", int(sorted[1, 1]))
        # print("3rd Best reults from cameras: ", int(sorted[2, 0]), "+", int(sorted[2, 1]))

        # for i in range(5):
        #     self.trinagulat_all(int(sorted[i, 0]), int(sorted[i, 1]))
        #     print(sorted[i, 0], sorted[i, 1])
        #     self.SBA()

        self.trinagulat_all(num1, num2)

    def save_3D(self):
        num1 = self.num1
        num2 = self.num2

        folder = r"C:\Users\franc\Documents\GitHub\PANDA-Gym-Data-Proceeing\Calibration"

        print("Saving As:" + folder + "\\3D_vid_" + str(num1) + "_" + str(num2) + ".csv")

        # initialize data of lists.
        part_idx = np.ones((self.n, 18))
        frame_idx = np.ones((self.n, 18))

        for i in range(18):
            part_idx[:, i] = part_idx[:, i] * i
            frame_idx[:, i] = np.arange(0, self.n, dtype=int).T

        # print(frame_idx)
        # print(part_idx)

        data = {
            "frame": frame_idx.ravel(),
            "x": self.X.ravel(),
            "y": -self.Y.ravel(),
            "z": -self.Z.ravel(),
            "part_idx": part_idx.ravel(),
        }

        # Create DataFrame
        df = pd.DataFrame(data)
        df.to_csv(folder + "\\3D_vid_" + str(num1) + "_" + str(num2) + ".csv", index=False)


tt = tringulatepose()
tt.check_combos()

# num1 = 3
# num2 = 5

# tt.trinagulat_all(num1, num2)
# # tt.trinagulat_conf()


# print(tt.error_percent())

# tt.SBA()
tt.overlay_pose(3, compare=1)
# tt.plotskel_loop()

# print(tt.error_percent())

# tt.save_3D()

# tt.threshold(5)
# tt.show_reproj()
