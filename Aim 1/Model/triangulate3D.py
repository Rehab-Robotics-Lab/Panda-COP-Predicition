# import sba
import json
import time
import os.path
import itertools
import pandas as pd
import numpy as np
import cv2 as cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display
from ProcessPose import processPose

# from anipose_cameras import camer
from scipy.sparse import lil_matrix, dok_matrix
from scipy.optimize import least_squares

# NEEDS A LOT OF REWRITING TO BE ROBUTS TO MISSING KEYPOINTS AND CAMERAS


class tringulatepose:
    def __init__(self, names, cams, start, intrinsics, extrinsics):
        # pfloder = r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Trials\Pose Estimates\3D Test\unsmoothed"
        # folder = r"C:\Users\franc\Documents\GitHub\PANDA-Gym-Data-Proceeing\Calibration"

        # rot = np.array([0, 180, 0, 0, 90])
        # rot = np.array([0, 180, 0, 0, 90, 0, 90])

        self.cams = cams

        n_cams = len(cams)
        self.n_cams = n_cams

        cam_objects = []

        n = 100000000

        for i in range(n_cams):
            print("Processing camera ", cams[i])
            # print(cams[i])
            cam = processPose(names[i], start[i])
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

            XA[j, :, :], YA[j, :, :], IA[j, :, :], conf[j, :, :] = X[:, 0:n], Y[:, 0:n], I[:, 0:n], C[:, 0:n]

            CA[j, :, :] = np.ones((18, n)) * (cams[j] - 1)

        self.XA = XA
        self.YA = YA
        self.conf = conf

        self.pt2D = np.vstack((self.XA.ravel(), self.YA.ravel())).T  # (2,(7*13*3626))
        self.ptidx = IA.ravel()
        self.camidx = CA.ravel()

        self.intrinsics = intrinsics

        self.extrinsics = extrinsics

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

        # intrinsics = self.intrinsics
        # extrinsics = self.extrinsics

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

    def triangulate(self, pt1, pt2, num1, num2):
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
        # print(pt1, cam1_mat, cam1_dist)

        pt1 = cv2.undistortPoints(pt1, cam1_mat, cam1_dist)
        pt2 = cv2.undistortPoints(pt2, cam2_mat, cam2_dist)

        homogeneous_points = cv2.triangulatePoints(cam1_proj, cam2_proj, pt1, pt2)
        points_3d = cv2.convertPointsFromHomogeneous(homogeneous_points.T)

        # t = np.linspace(0, 7360, 7361) / 60

        X = points_3d[:, :, 0]
        Y = points_3d[:, :, 1]
        Z = points_3d[:, :, 2]

        return X, Y, Z

    def trinagulate_all(self, num1, num2):
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

    # def trinagulat_i(self, num1, num2, j):
    #     num1 = self.num1
    #     num2 = self.num2

    #     X1, Y1 = self.XA[num1 - 1, :, :], self.YA[num1 - 1, :, :]
    #     X2, Y2 = self.XA[num2 - 1, :, :], self.YA[num2 - 1, :, :]

    #     X = np.zeros((18, 1))
    #     Y = np.zeros((18, 1))
    #     Z = np.zeros((18, 1))

    #     for i in range(18):

    #         pt_c1 = np.vstack((X1[i, j], Y1[i, j]))
    #         pt_c2 = np.vstack((X2[i, j], Y2[i, j]))

    #         x, y, z = self.trinagulate_each(pt_c1, pt_c2, num1, num2)

    #         X[i] = x
    #         Y[i] = y
    #         Z[i] = z

    #     return X, Y, Z

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

    def triangulate_idx(self, num1, num2, idx):
        self.num1 = num1
        self.num2 = num2

        idx1 = np.where(self.cams == num1)[0][0]
        idx2 = np.where(self.cams == num2)[0][0]

        X1, Y1 = self.XA[idx1, :, :], self.YA[idx1, :, :]
        X2, Y2 = self.XA[idx2, :, :], self.YA[idx2, :, :]

        pt_c1 = np.vstack((X1[idx, :], Y1[idx, :]))
        pt_c2 = np.vstack((X2[idx, :], Y2[idx, :]))

        x, y, z = self.triangulate(pt_c1, pt_c2, num1, num2)

        return x, y, z

    def triangulate_idx_i(self, num1, num2, idx, i):
        x, y, z = self.triangulate_idx(num1, num2, idx)

        return x[i, 0], y[i, 0], z[i, 0]

    def reproj_each_idx(self, X, Y, Z, idx):
        intrinsics = self.intrinsics

        error = np.zeros((self.n, self.n_cams))

        for i in range(self.n_cams):
            num = self.cams[i]
            cam_int = intrinsics["cam" + str(num)]

            cam_mat = np.asmatrix(cam_int["Mat"])
            cam_dist = np.asarray(cam_int["Dist"])

            id = np.where(self.cams == num)[0][0]

            cam_ext = self.ext_params[id, :, :]  # (7,2,3)

            cam_rvec = cam_ext[0, :]
            cam_tvec = cam_ext[1, :]

            pts = np.hstack((X, Y, Z)).T

            # reproj[:, :, [i]] = cv2.projectPoints(pts, cam_rvec, cam_tvec, cam_mat, cam_dist)
            R, _ = cv2.projectPoints(pts, cam_rvec, cam_tvec, cam_mat, cam_dist)
            Rpt = R.reshape(-1, 2).T

            pt = np.vstack((self.XA[i, idx, :], self.YA[i, idx, :]))

            error[:, i] = np.linalg.norm(Rpt - pt, axis=0)

        return error

    def reproj_each_cam(self, num):
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
            X, Y, e = self.reproj_each_cam(self.cams[i])

            XAr[i, :, :] = X
            YAr[i, :, :] = Y
            E[i] = e

        return XAr, YAr, E

    def recover_cam3(self, name=None):
        intrinsics = pd.read_json(
            r"C:\Users\franc\Documents\GitHub\PANDA-Data-Processing\Calibration\intrinsic_params.json"
        ).to_dict()
        extrinsics = pd.read_json(
            r"C:\Users\franc\Documents\GitHub\PANDA-Data-Processing\Calibration\extrinsic_params.json"
        ).to_dict()

        cam_int = intrinsics["cam3"]

        cam_mat = np.asmatrix(cam_int["Mat"])
        cam_dist = np.asarray(cam_int["Dist"])

        cam_ext = extrinsics["cam3"]  # (7,2,3)

        cam_rvec = np.asmatrix(cam_ext["Rvec"])
        cam_tvec = np.asmatrix(cam_ext["Tvec"])

        # print(cam_rvec)

        reproj = np.zeros((2, self.n, 18))

        for i in range(18):
            pts = np.vstack((self.X[:, i], self.Y[:, i], self.Z[:, i]))

            # reproj[:, :, [i]] = cv2.projectPoints(pts, cam_rvec, cam_tvec, cam_mat, cam_dist)
            R, _ = cv2.projectPoints(pts, cam_rvec, cam_tvec, cam_mat, cam_dist)
            R = R.reshape(-1, 2).T

            reproj[:, :, i] = R

        Xreproj = reproj[0, :, :]
        Yreproj = reproj[1, :, :]

        part_idx = np.ones((self.n, 18))
        frame_idx = np.ones((self.n, 18))
        fps = np.ones((self.n, 18)) * 30
        conf = np.ones((self.n, 18)) * 2

        for i in range(18):
            part_idx[:, i] = part_idx[:, i] * i
            frame_idx[:, i] = np.arange(0, self.n, dtype=int).T

        # print(frame_idx)
        # print(part_idx)

        data = {
            "frame": frame_idx.ravel(),
            "x": Xreproj.ravel(),
            "y": Yreproj.ravel(),
            "part_idx": part_idx.ravel(),
            "fps": fps.ravel(),
            "c": conf.ravel(),
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        if name != None:
            df.to_csv(name)

        return Xreproj, Yreproj

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

        idx = np.where(self.cams == camnum)[0][0]
        x, y = self.XA[idx, :, j - 1], self.YA[idx, :, j - 1]

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

        x = X[:, j - 1].T
        y = Y[:, j - 1].T

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

    def overlay_pose(self, camnum, vidname, compare):
        # storing video rotation
        # rot = np.array([0, 180, 0, 0, -90, 0, -90])

        filename = vidname
        # read in video
        cap = cv2.VideoCapture(filename)

        # OPENCV_FFMPEG_READ_ATTEMPTS (current value is 4096
        width = cap.get(3)  # float `width`
        height = cap.get(4)  # float `height`

        j = 0
        idx = np.where(self.cams == camnum)[0][0]

        # print(self.cam_objects[camnum - 1].start)
        while cap.isOpened():
            # variable to skip frames every other frame, remains true if video at 30fps
            skip = True
            frameId = int(cap.get(1))

            # skip = self.cam_objects[camnum - 1].start < frameId

            # if 60fps video skip every other frame to diaplay video at 30fps
            # if self.cam_objects[camnum - 1].start < frameId:

            wait = self.cam_objects[idx].start < frameId

            if self.cam_objects[idx].fps == 60:
                skip = frameId % 2

            # Capturing each frame of our video stream
            ret, frame = cap.read()
            # frame = cv2.resize(frame, (700, 500))
            if skip and wait:
                if ret == True:
                    j = j + 1

                    # rotating video
                    # if rot[idx] == 90:
                    #     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    # elif rot[idx] == 180:
                    #     frame = cv2.rotate(frame, cv2.ROTATE_180)
                    # elif rot[idx] == -90:
                    #     frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

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
            X, Y, E = self.reproj_each_cam(self.cams[j])
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
            X, Y, E = self.reproj_each_cam(self.cams[j])
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

            Xreproj, Yreproj, mean_error = self.reproj_each_cam(camnum)
            # Create new rows as separate DataFrames

            error[i] = mean_error

            # t = self.t
            # fig2 = plt.figure
            # print(np.shape(Xreproj[:, 0]), np.shape((self.XA[camnum - 1][0, :])))
            j = camnum * 2 - 1

            t = np.linspace(0, Xreproj.size, num=Xreproj.size)

            print(j)

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

        cameraIndices = self.camidx
        pointIndices = self.ptidx
        numCameras = ncams
        numPoints = 18 * self.n

        m = cameraIndices.size * 2
        n = (numCameras * 6) + (3 * numPoints)

        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)

        for s in range(6):
            A[2 * i, cameraIndices * 6 + s] = 1
            A[2 * i + 1, cameraIndices * 6 + s] = 1

        for s in range(3):
            A[2 * i, numCameras * 6 + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * 6 + pointIndices * 3 + s] = 1

        # print("J")

        return A

    def SBA(self, disp=1, verbose=2):
        # points_3d = np.vstack((self.X.T.ravel(), self.Y.T.ravel(), self.Z.T.ravel()))
        points_3d = np.vstack((self.X.T.ravel(), self.Y.T.ravel(), self.Z.T.ravel())).T

        x0 = np.hstack((self.ext_params.ravel(), points_3d.ravel()))

        f0 = self.residual(x0)
        A = self.jacob()
        # print(np.any(np.isnan(f0)), np.any(np.isinf(f0)))

        print("Optimizing")

        # res = least_squares(self.residual, x0, loss="linear", jac_sparsity=A, verbose=2, ftol=1e-3)
        res = least_squares(
            self.residual,
            x0,
            jac_sparsity=A,
            verbose=verbose,
            x_scale="jac",
            ftol=1e-4,
            method="trf",
        )

        # print(res)

        # params = res.x[0 : 6 * 7]
        ff = res.fun

        if disp == 1:

            plt.plot(f0)
            plt.plot(ff)
            plt.ylim(-1000, 1000)
            plt.show()

        # print(self.ext_params)

    def _initialize_params_triangulation(self, p3ds, constraints=[], constraints_weak=[]):
        joint_lengths = np.empty(len(constraints), dtype="float64")
        joint_lengths_weak = np.empty(len(constraints_weak), dtype="float64")

        for cix, (a, b) in enumerate(constraints):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            joint_lengths[cix] = np.median(lengths)

        for cix, (a, b) in enumerate(constraints_weak):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            joint_lengths_weak[cix] = np.median(lengths)

        all_lengths = np.hstack([joint_lengths, joint_lengths_weak])
        med = np.median(all_lengths)
        if med == 0:
            med = 1e-3

        mad = np.median(np.abs(all_lengths - med))

        joint_lengths[joint_lengths == 0] = med
        joint_lengths_weak[joint_lengths_weak == 0] = med
        joint_lengths[joint_lengths > med + mad * 5] = med
        joint_lengths_weak[joint_lengths_weak > med + mad * 5] = med

        return np.hstack([p3ds.ravel(), joint_lengths, joint_lengths_weak])

    def _jac_sparsity_triangulation(self, p2ds, constraints=[], constraints_weak=[], n_deriv_smooth=1):

        n_cams, n_frames, n_joints, _ = p2ds.shape
        n_constraints = len(constraints)
        n_constraints_weak = len(constraints_weak)

        p2ds_flat = p2ds.reshape((n_cams, -1, 2))

        point_indices = np.zeros(p2ds_flat.shape, dtype="int32")
        for i in range(p2ds_flat.shape[1]):
            point_indices[:, i] = i

        point_indices_3d = np.arange(n_frames * n_joints).reshape((n_frames, n_joints))

        good = ~np.isnan(p2ds_flat)
        n_errors_reproj = np.sum(good)
        n_errors_smooth = (n_frames - n_deriv_smooth) * n_joints * 3
        n_errors_lengths = n_constraints * n_frames
        n_errors_lengths_weak = n_constraints_weak * n_frames

        n_errors = n_errors_reproj + n_errors_smooth + n_errors_lengths + n_errors_lengths_weak

        n_3d = n_frames * n_joints * 3
        n_params = n_3d + n_constraints + n_constraints_weak

        point_indices_good = point_indices[good]

        A_sparse = dok_matrix((n_errors, n_params), dtype="int16")

        # constraints for reprojection errors
        ix_reproj = np.arange(n_errors_reproj)
        for k in range(3):
            A_sparse[ix_reproj, point_indices_good * 3 + k] = 1

        # sparse constraints for smoothness in time
        frames = np.arange(n_frames - n_deriv_smooth)
        for j in range(n_joints):
            for n in range(n_deriv_smooth + 1):
                pa = point_indices_3d[frames, j]
                pb = point_indices_3d[frames + n, j]
                for k in range(3):
                    A_sparse[n_errors_reproj + pa * 3 + k, pb * 3 + k] = 1

        ## -- strong constraints --
        # joint lengths should change with joint lengths errors
        start = n_errors_reproj + n_errors_smooth
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints):
            A_sparse[start + cix * n_frames + frames, n_3d + cix] = 1

        # points should change accordingly to match joint lengths too
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints):
            pa = point_indices_3d[frames, a]
            pb = point_indices_3d[frames, b]
            for k in range(3):
                A_sparse[start + cix * n_frames + frames, pa * 3 + k] = 1
                A_sparse[start + cix * n_frames + frames, pb * 3 + k] = 1

        ## -- weak constraints --
        # joint lengths should change with joint lengths errors
        start = n_errors_reproj + n_errors_smooth + n_errors_lengths
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints_weak):
            A_sparse[start + cix * n_frames + frames, n_3d + n_constraints + cix] = 1

        # points should change accordingly to match joint lengths too
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints_weak):
            pa = point_indices_3d[frames, a]
            pb = point_indices_3d[frames, b]
            for k in range(3):
                A_sparse[start + cix * n_frames + frames, pa * 3 + k] = 1
                A_sparse[start + cix * n_frames + frames, pb * 3 + k] = 1

        return A_sparse

    def _error_fun_triangulation(
        self,
        params,
        p2ds,
        constraints=[],
        constraints_weak=[],
        scores=None,
        scale_smooth=10000,
        scale_length=1,
        scale_length_weak=0.2,
        reproj_error_threshold=100,
        reproj_loss="soft_l1",
        n_deriv_smooth=1,
        p3ds_fixed=None,
    ):

        n_cams, n_frames, n_joints, _ = p2ds.shape

        n_3d = n_frames * n_joints * 3
        n_constraints = len(constraints)
        n_constraints_weak = len(constraints_weak)

        # load params
        p3ds = params[:n_3d].reshape((n_frames, n_joints, 3))
        joint_lengths = np.array(params[n_3d : n_3d + n_constraints])
        joint_lengths_weak = np.array(params[n_3d + n_constraints :])

        ## if fixed points, first n_fixed parameter points are ignored
        ## and replacement points are put in
        ## this way we can keep rest of code the same, especially _jac_sparsity_triangulation
        if p3ds_fixed is not None:
            n_fixed = p3ds_fixed.shape[0]
            p3ds = np.vstack([p3ds_fixed, p3ds[n_fixed:]])

        # reprojection errors
        p3ds_flat = p3ds.reshape(-1, 3)
        p2ds_flat = p2ds.reshape((n_cams, -1, 2))
        errors = self.reprojection_error(p3ds_flat, p2ds_flat)
        if scores is not None:
            scores_flat = scores.reshape((n_cams, -1))
            errors = errors * scores_flat[:, :, None]
        errors_reproj = errors[~np.isnan(p2ds_flat)]

        rp = reproj_error_threshold
        errors_reproj = np.abs(errors_reproj)
        if reproj_loss == "huber":
            bad = errors_reproj > rp
            errors_reproj[bad] = rp * (2 * np.sqrt(errors_reproj[bad] / rp) - 1)
        elif reproj_loss == "linear":
            pass
        elif reproj_loss == "soft_l1":
            errors_reproj = rp * 2 * (np.sqrt(1 + errors_reproj / rp) - 1)

        # temporal constraint
        errors_smooth = np.diff(p3ds, n=n_deriv_smooth, axis=0).ravel() * scale_smooth

        # joint length constraint
        errors_lengths = np.empty((n_constraints, n_frames), dtype="float64")
        for cix, (a, b) in enumerate(constraints):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            expected = joint_lengths[cix]
            errors_lengths[cix] = 100 * (lengths - expected) / expected
        errors_lengths = errors_lengths.ravel() * scale_length

        errors_lengths_weak = np.empty((n_constraints_weak, n_frames), dtype="float64")
        for cix, (a, b) in enumerate(constraints_weak):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            expected = joint_lengths_weak[cix]
            errors_lengths_weak[cix] = 100 * (lengths - expected) / expected
        errors_lengths_weak = errors_lengths_weak.ravel() * scale_length_weak

        return np.hstack([errors_reproj, errors_smooth, errors_lengths, errors_lengths_weak])

    def reprojection_error(self, pts3d, pts2d):
        n_cams = self.n_cams
        Xr, Yr, Er = self.reproj_all()

        pts2d_rpj = np.stack((np.transpose(Xr, axes=(0, 2, 1)), np.transpose(Yr, axes=(0, 2, 1))), axis=3).reshape(
            (n_cams, -1, 2)
        )

        # print(np.shape(pts2d_rpj.reshape((n_cams, -1, 2))))
        # print(np.shape(pts2d))

        return pts2d - pts2d_rpj

    def optim_points(
        self,
        points,
        p3ds,
        scale_smooth=0.5,
        scale_length=2,
        scale_length_weak=0.5,
        reproj_error_threshold=15,
        reproj_loss="soft_l1",
        n_deriv_smooth=4,
        scores=None,
        verbose=False,
        n_fixed=0,
    ):
        """
        Take in an array of 2D points of shape CxNxJx2,
        an array of 3D points of shape NxJx3,
        and an array of constraints of shape Kx2, where
        C: number of camera
        N: number of frames
        J: number of joints
        K: number of constraints

        This function creates an optimized array of 3D points of shape NxJx3.

        Example constraints:
        constraints = [[0, 1], [1, 2], [2, 3]]
        (meaning that lengths of segments 0->1, 1->2, 2->3 are all constant)

        """
        # assert points.shape[0] == len(
        #     self.cameras
        # ), "Invalid points shape, first dim should be equal to" " number of cameras ({}), but shape is {}".format(
        #     len(self.cameras), points.shape
        # )

        constraints = [[2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [8, 11]]
        constraints_weak = [[1, 2], [1, 5], [2, 5]]

        n_cams, n_frames, n_joints, _ = points.shape
        constraints = np.array(constraints)
        constraints_weak = np.array(constraints_weak)

        # p3ds_intp = np.apply_along_axis(interpolate_data, 0, p3ds)

        # p3ds_med = np.apply_along_axis(medfilt_data, 0, p3ds_intp, size=7)

        default_smooth = 1.0 / np.mean(np.abs(np.diff(p3ds, axis=0)))
        scale_smooth_full = scale_smooth * default_smooth

        t1 = time.time()

        x0 = self._initialize_params_triangulation(p3ds, constraints, constraints_weak)

        x0[~np.isfinite(x0)] = 0

        if n_fixed > 0:
            p3ds_fixed = p3ds[:n_fixed]
        else:
            p3ds_fixed = None

        jac = self._jac_sparsity_triangulation(points, constraints, constraints_weak, n_deriv_smooth)

        opt2 = least_squares(
            self._error_fun_triangulation,
            x0=x0,
            jac_sparsity=jac,
            loss="linear",
            ftol=1e-3,
            verbose=2 * verbose,
            args=(
                points,
                constraints,
                constraints_weak,
                scores,
                scale_smooth_full,
                scale_length,
                scale_length_weak,
                reproj_error_threshold,
                reproj_loss,
                n_deriv_smooth,
                p3ds_fixed,
            ),
        )

        p3ds_new2 = opt2.x[: p3ds.size].reshape(p3ds.shape)

        if n_fixed > 0:
            p3ds_new2 = np.vstack([p3ds_fixed, p3ds_new2[n_fixed:]])

        t2 = time.time()

        if verbose:
            print("optimization took {:.2f} seconds".format(t2 - t1))

        return p3ds_new2

    def SBA_anipose(self, verbose=True):
        pts3d = np.stack((self.X, self.Y, self.Z), axis=2)
        pts2d = np.stack((np.transpose(self.XA, axes=(0, 2, 1)), np.transpose(self.YA, axes=(0, 2, 1))), axis=3)

        pts3D = self.optim_points(pts2d, pts3d, verbose=True)
        self.X = pts3D[:, :, 0]
        self.Y = pts3D[:, :, 1]
        self.Z = pts3D[:, :, 2]
        # print(np.shape(pts3D))
        # print(np.shape(self.X), np.shape(self.Y), np.shape(self.Z))

    def reproj_check(self):
        n = self.n
        ncams = self.n_cams

        diff = np.zeros((ncams, 18, n))

        cams_names = np.empty(ncams, dtype="S4")

        for j in range(ncams):
            X, Y, E = self.reproj_each_cam(self.cams[j])

            xg, yg = self.XA[j, :, :], self.YA[j, :, :]

            # print(np.shape(np.linalg.norm(np.stack((X, Y)) - np.stack((xg, yg)), axis=0)))

            diff[j, :, :] = np.linalg.norm(np.stack((X, Y)) - np.stack((xg, yg)), axis=0)

            cams_names[j] = "Cam" + str(self.cams[j])

        # plt.plot(Xd)
        diff_mean = np.mean(diff, axis=2)
        # print(cams_names)

        bodyparts = [
            "nose",
            "neck",
            "rshldr",
            "relbw",
            "rwrst",
            "lshldr",
            "lelbw",
            "lwrst",
            "rhip",
            "rknee",
            "rfoot",
            "lhip",
            "lknee",
            "lfoot",
            "reye",
            "leye",
            "rear",
            "lear",
        ]
        # print(np.shape(diff_mean))
        for i in range(18):
            plt.scatter(cams_names, diff_mean[:, i], label=bodyparts[i])

        plt.plot(np.mean(diff_mean, axis=1), label="AVERAGE", marker="D", markersize=12, color="gold")

        plt.plot(diff_mean, color="black", alpha=0.5, linewidth=1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid()
        plt.tight_layout(pad=1)

        plt.show()

    def check_combos(self):
        results = np.zeros((len(self.combos), 3))

        i = 0
        print("Checking camera combinations")
        for c in self.combos:
            # print("Combo: ", c[0], c[1])

            self.triangulate_all(c[0], c[1])
            # self.SBA(disp=0, verbose=0)
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

        print("1st Best reults from cameras: ", num1, "+", num2, " Res= ", int(sorted[0, 2]))
        print("2nd Best reults from cameras: ", int(sorted[1, 0]), "+", int(sorted[1, 1]), " Res= ", int(sorted[1, 2]))
        print("3rd Best reults from cameras: ", int(sorted[2, 0]), "+", int(sorted[2, 1]), " Res= ", int(sorted[2, 2]))
        # print("4th Best reults from cameras: ", int(sorted[3, 0]), "+", int(sorted[3, 1]), " Res= ", int(sorted[3, 2]))
        # print("5th Best reults from cameras: ", int(sorted[4, 0]), "+", int(sorted[4, 1]), " Res= ", int(sorted[4, 2]))

        self.triangulate_all(num1, num2)

    def check_combos_idx(self):

        print("Checking camera combinations idx")

        pairs = np.zeros((2, 18))
        error_cams = np.zeros((len(self.combos), self.n_cams))

        for i in range(18):
            results = np.zeros((len(self.combos), 3))
            # i = 12
            j = 0

            col_names = ["num1", "num2", "mean_wght"]
            # print(col_names)
            num_names = []
            for cam in self.cams:
                num_names.append(str(cam))
                col_names.append(str(cam))

            # weights = pd.DataFrame(index=num_names, columns=["weight1", "weight2"])

            df = pd.DataFrame(columns=col_names.append("mean"))

            # print(col_names)

            # pair_weight=np.zeros((len(self.combos),self.n_cams))

            for c in self.combos:
                # print("COMBO--------------------------------------- ", c[0], " ", c[1])
                X, Y, Z = self.triangulate_idx(c[0], c[1], idx=i)
                error = np.mean(self.reproj_each_idx(X, Y, Z, idx=i), axis=0)
                # self.SBA(disp=0, verbose=0)
                # print(error, int(np.mean(error)))

                error_cams[j, :] = error

                # df = pd.concat([df, pd.DataFrame({"num1": [c[0]], "num2": [c[1]], "mean": [np.mean(error)]})])
                df = pd.concat([df, pd.DataFrame({"num1": [c[0]], "num2": [c[1]]})])

                df.loc[(df["num1"] == c[0]) & (df["num2"] == c[1]), num_names] = error

                df.loc[(df["num1"] == c[0]) & (df["num2"] == c[1]), "mean"] = np.mean(
                    df.loc[(df["num1"] == c[0]) & (df["num2"] == c[1]), num_names]
                )

                # df[(df.loc[:, num_names] < 50) & (df.loc[:, num_names] >= 25)] = (
                #     df[(df.loc[:, num_names] < 50) & (df.loc[:, num_names] >= 25)] * -1
                # )
                # df[df.loc[:, num_names] < 25] = df[df.loc[:, num_names] < 25] * -2

                df.loc[(df["num1"] == c[0]) & (df["num2"] == c[1]), str(c[0])] = (
                    df.loc[(df["num1"] == c[0]) & (df["num2"] == c[1]), str(c[0])] * 2
                )
                df.loc[(df["num1"] == c[0]) & (df["num2"] == c[1]), str(c[1])] = (
                    df.loc[(df["num1"] == c[0]) & (df["num2"] == c[1]), str(c[1])] * 2
                )

                mean_weighted = np.mean(df.loc[(df["num1"] == c[0]) & (df["num2"] == c[1]), num_names])
                df.loc[(df["num1"] == c[0]) & (df["num2"] == c[1]), "mean_wght"] = mean_weighted

                # results[j, :] = np.array([(c[0]), (c[1]), np.mean(error)])

                # print(mean_weighted)
                # quit()

                results[j, :] = np.array([(c[0]), (c[1]), mean_weighted])
                j = j + 1

                srt = np.argsort(results[:, 2])
                sorted = results[srt, :]

                pairs[0, i] = int(sorted[0, 0])
                pairs[1, i] = int(sorted[0, 1])

            # print(df)
            # quit()

            # tb = df.loc[:, num_names]
            # tb[df.loc[:, num_names] >= 100] = 3
            # tb[(df.loc[:, num_names] < 100) & (df.loc[:, num_names] >= 50)] = 1
            # tb[(df.loc[:, num_names] < 50) & (df.loc[:, num_names] >= 25)] = -1
            # tb[df.loc[:, num_names] < 25] = -3

            # # print(int(num_names))
            # for num in self.cams:
            #     tb.loc[(df["num1"] == num) | (df["num2"] == num), str(num)] = (
            #         tb.loc[(df["num1"] == num) | (df["num2"] == num), str(num)] * 2
            #     )

            #     weights.loc[str(num), "weight2"] = np.mean(df.loc[(df["num1"] == num) | (df["num2"] == num), "mean"])

            # weights["weight1"] = tb.sum(axis=0)

            # weights.plot(kind="bar")
            # plt.show()

            # tb.loc[(tb["num1"] == c[0]) & (df["num2"] == c[1]), num_names] = error

            # print(tb)

            x, y, z = self.triangulate_idx(int(sorted[0, 0]), int(sorted[0, 1]), i)

            self.X[:, i], self.Y[:, i], self.Z[:, i] = x[:, 0], y[:, 0], z[:, 0]

    def save_3D(self, folder, name=None, suffix=None):
        # num1 = self.num1
        # num2 = self.num2

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
        if name == None:
            full_name = folder + "\\3D_vid_" + suffix + ".csv"
        else:
            full_name = folder + "\\" + name + suffix + ".csv"
        df = pd.DataFrame(data)
        df.to_csv(full_name, index=False)

        print("Saved As:" + full_name)


# tt = tringulatepose()
# tt.check_combos()

# num1 = 3
# num2 = 5

# tt.trinagulate_all(num1, num2)
# # tt.trinagulat_conf()


# print(tt.error_percent())

# tt.SBA()
# tt.overlay_pose(3, compare=1)
# tt.plotskel_loop()

# print(tt.error_percent())

# tt.save_3D()

# tt.threshold(5)
# tt.show_reproj()
