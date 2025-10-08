import pandas as pd
import cv2 as cv2
import cv2.aruco as aruco
import pickle
import numpy as np
import datetime as dt
from Cmanage import c_manage
from Calibrate import calibrate
import scipy
from CompareCOP import compareCOP
from ProcessCOP import processCOP
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class COP_Tag:
    def __init__(self, cam_dir, name, calibnum=1, load=0, intrinsics_name=None):
        self.trial_name = name
        self.cam_dir = cam_dir

        sim = c_manage(cam_direct=cam_dir, vid_name=name)

        calib_status, calib_names = sim.check_vids(calibnum)

        # sim.intrinsics_orientation(calib_names, calib_status)
        # print(sim.intrinsics_final)

        if load == 0:
            sim.intrinsics_orientation(calib_names, calib_status)
            self.intrinsics = sim.intrinsics_final
        elif load == 1:
            self.intrinsics = self.load_intrinsics(name=intrinsics_name).to_dict()

        self.calib_names, self.calib_status = calib_names, calib_status
        self.sim = sim

    def showtags(self, j):
        plt.cla()

        tv_0 = self.tv0_glob[j]
        rv_0 = self.rv0_glob[j]
        tv_25 = self.tv25_glob[j]
        rv_25 = self.rv25_glob[j]

        self.ax.plot3D([tv_0[0], tv_25[0]], [tv_0[1], tv_25[1]], [tv_0[2], tv_25[2]], "o")

        # print(self.tvec_glob.flatten()[2] - tv_0[2], self.tvec_glob.flatten()[2] - tv_25[2])

        self.ax = self.show_Ti(self.ax, self.tvec_glob.flatten(), self.rvec_glob, scale=0.2, ID="G")
        self.ax = self.show_Ti(self.ax, self.tvec_cam.flatten(), self.rvec_cam, scale=0.2, ID="C")
        self.ax = self.show_Ti(self.ax, tv_0, rv_0, ID=0)
        # com = tv_0 + rv_0.apply(np.array([0.01, 0.01, -0.07]))
        # self.ax = self.show_Ti(self.ax, com, rv_0, ID="m")
        self.ax = self.show_Ti(self.ax, tv_25, rv_25, ID=25)

        s = "t= " + str(round(j / 60, 2))

        self.ax.text(0.3, -0.4, 0, "%s" % (s), size=20, zorder=1, color="k")

        self.ax.set_zlim3d(0, 0.6)
        self.ax.set_ylim3d(-0.6, 0.6)
        self.ax.set_xlim3d(-0.6, 0.6)
        # print(4.5)
        plt.grid()
        # print(4)

    def showtags_loop(self, camnum, vidnum, name=None):
        if name == None:
            (
                self.rv0_glob,
                self.tv0_glob,
                self.rv25_glob,
                self.tv25_glob,
                self.rvec_glob,
                self.tvec_glob,
                self.rvec_cam,
                self.tvec_cam,
            ) = self.glob_pose(camnum, vidnum)
        else:
            (
                self.rv0_glob,
                self.tv0_glob,
                self.rv25_glob,
                self.tv25_glob,
                self.rvec_glob,
                self.tvec_glob,
                self.rvec_cam,
                self.tvec_cam,
            ) = self.glob_pose(camnum, vidnum, name=name, load=1)
        print("Displaying Pose")

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        # print(np.linspace(1,max_frames,max_frames))

        ani = FuncAnimation(self.fig, self.showtags, frames=np.linspace(1, self.max_frames, self.max_frames, dtype=int))
        plt.show()

    def show_Ti(self, ax, tvec, rvec, scale=0.1, sign=1, ID="N"):

        dx, dy, dz = tvec[0], tvec[1], tvec[2]

        var = 1 * scale * sign

        xaxis = np.array([var, 0, 0])
        yaxis = np.array([0, var, 0])
        zaxis = np.array([0, 0, var])

        x = rvec.apply(xaxis).flatten()
        y = rvec.apply(yaxis).flatten()
        z = rvec.apply(zaxis).flatten()

        d = scale / 10

        ax.quiver([dx], [dy], [dz], x[0], x[1], x[2], colors=[1, 0, 0])

        ax.quiver([dx], [dy], [dz], y[0], y[1], y[2], colors="green")

        ax.quiver([dx], [dy], [dz], z[0], z[1], z[2], colors=[0, 0, 1])
        ax.text(dx, dy, dz, str(ID))

        return ax

    def glob_pose(self, camnum, vidnum, load=0, name=None):
        print("Getting global extrinsics")
        cam_ex, rerror = self.glob_extrinsics(camnum, view=0)
        rvec_glob = cam_ex["Rvec"]
        tvec_glob = cam_ex["Tvec"]

        R_glob = R.from_rotvec(rvec_glob.T)
        # print(R_glob)

        R_orient = R.from_matrix(np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))

        # print(R_orient)

        R_glob_cor = R_glob * R_orient

        if load == 0:
            tag_pose = self.get_pose(camnum, vidnum)
        else:
            tag_pose = self.load_tagpose(name=name)

        tag0 = tag_pose.loc[(tag_pose["ID"] == 0)]
        tag25 = tag_pose.loc[(tag_pose["ID"] == 25)]

        self.max_frames = np.max(tag25["frame"])

        rv0 = tag0["rvecs"]
        tv0 = tag0["tvec"]

        rv25 = tag25["rvecs"]
        tv25 = tag25["tvec"]

        print("Adjusting extrinsics")

        Rc = R_glob_cor.inv()
        Tc = Rc.apply(-tvec_glob.T)

        rv0 = np.array(rv0.to_list())
        rv0 = R.from_rotvec(rv0)
        tv0 = np.array(tv0.to_list())

        rv25 = np.array(rv25.to_list())
        rv25 = R.from_rotvec(rv25)
        tv25 = np.array(tv25.to_list())
        # print(rv0)

        R0 = R_glob_cor.inv() * rv0
        R25 = R_glob_cor.inv() * rv25
        # R0 = R_glob_cor.inv() * rv0.inv()
        # R25 = R_glob_cor.inv() * rv25.inv()

        T0 = Rc.apply(-tvec_glob.T + tv0)
        T25 = Rc.apply(-tvec_glob.T + tv25)

        Rg = R.from_matrix(np.eye(3))
        Tg = np.array([0, 0, 0])

        return R0, T0, R25, T25, Rg, Tg, Rc, Tc
        # return rv0, tv0, rv25, tv25, R_glob_cor, tvec_glob, R.from_matrix(np.eye(3)), np.array([0, 0, 0])

    def glob_extrinsics(self, camnum, view, chess=0):
        intrinsics = self.intrinsics["cam" + str(camnum)]

        if chess == 0:

            calib_vid = self.calib_names[camnum - 1]
            calib = calibrate(video=calib_vid, vid_intrinsic=intrinsics)
            img = calib.pull_frames_static(calib_vid)

            cam_ex, rerror = calib.calibrate_extrinsics(img, intrinsics, view=view)
        else:
            calib_status, calib_name = self.sim.check_vids(2)
            calib_vid = calib_name[camnum - 1]
            print(calib_vid)
            calib = calibrate(video=calib_vid, vid_intrinsic=intrinsics)
            img = calib.pull_frames_static(calib_vid)

            cam_ex, rerror = calib.calibrate_extrinsics_chess(img, intrinsics, view=view)

        return cam_ex, rerror

    def get_pose(self, camnum, vidnum, view=0):
        print("Getting tag extrinsics for camera ", str(camnum), "vidoe ", str(vidnum))
        tag_pose = pd.DataFrame()
        sim = self.sim

        status, name = sim.check_vids(vidnum)
        cap = cv2.VideoCapture(name[camnum - 1])

        intrinsics = self.intrinsics["cam" + str(camnum)]

        framenum = 0
        max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # # Check if camera opened successfully
        if status[camnum - 1] == 0:
            print("No Video")
        elif cap.isOpened() == False:
            print("Error opening video file")

        else:
            # Read until video is completed
            while cap.isOpened() and framenum < max_frames:

                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret == True:
                    # Display the resulting frame

                    frame, ids, tvecs, rvecs = self.apriltag_pose(frame, intrinsics)
                    framenum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                    if 0 in ids:
                        row0 = pd.DataFrame(
                            {"ID": [ids[0, 0]], "tvec": [tvecs[:, 0]], "rvecs": [rvecs[:, 0]], "frame": [framenum]}
                        )
                    else:
                        row0 = pd.DataFrame(
                            {
                                "ID": [0],
                                "tvec": [[0, 0, 0]],
                                "rvecs": [[0, 0, 0]],
                                "frame": [framenum],
                            }
                        )
                    if 25 in ids:
                        row25 = pd.DataFrame(
                            {"ID": [ids[1, 0]], "tvec": [tvecs[:, 1]], "rvecs": [rvecs[:, 1]], "frame": [framenum]}
                        )
                    else:
                        row0 = pd.DataFrame(
                            {
                                "ID": [25],
                                "tvec": [[0, 0, 0]],
                                "rvecs": [[0, 0, 0]],
                                "frame": [framenum],
                            }
                        )

                    tag_pose = pd.concat([tag_pose, row0], ignore_index=True)
                    tag_pose = pd.concat([tag_pose, row25], ignore_index=True)

                    # print(framenum)
                    if view == 1:
                        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
                        cv2.imshow("Frame", frame)

                        if cv2.waitKey(25) & 0xFF == ord("q"):
                            break

            cap.release()

            # Closes all the frames
            cv2.destroyAllWindows()

        return tag_pose

    def apriltag_pose(self, frame, intrinsics):
        ARUCO_PARAMETERS = aruco.DetectorParameters()
        ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

        ProjectImage = frame

        # grayscale image
        gray = cv2.cvtColor(ProjectImage, cv2.COLOR_BGR2GRAY)

        cameraMatrix = intrinsics["Mat"]
        distCoeffs = intrinsics["Dist"]

        # Detect Aruco markers
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
        # marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(image)
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

        # Outline all of the markers detected in our image
        # Uncomment below to show ids as well
        markerlength = 40 / 1000

        rvecs = np.zeros((3, len(ids)))
        tvecs = np.zeros((3, len(ids)))
        tot_error = 0

        for j in range(len(ids)):
            corner = corners[j]

            # rvec,tvec,objpts=aruco.estimatePoseSingleMarkers(corner, markerlength, cameraMatrix, distCoeffs)
            objpts = np.array(
                [
                    [-markerlength / 2, markerlength / 2, 0],
                    [markerlength / 2, markerlength / 2, 0],
                    [markerlength / 2, -markerlength / 2, 0],
                    [-markerlength / 2, -markerlength / 2, 0],
                ],
                dtype=np.float32,
            )

            if type(cameraMatrix) == list:
                cameraMatrix = np.matrix(cameraMatrix)

            if type(distCoeffs) == list:
                distCoeffs = np.matrix(distCoeffs)

            success, rvec, tvec = cv2.solvePnP(objpts, np.reshape(corner, (4, 1, 2)), cameraMatrix, distCoeffs)

            # print('Camera ',i, success, len(ids))

            tvecs[:, [j]] = tvec
            rvecs[:, [j]] = rvec

            imgpts_rpj, _ = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, distCoeffs)
            # print(imgpts_rpj)

            # ProjectImage = aruco.drawDetectedMarkers(ProjectImage, imgpts.reshape(-1, 2), borderColor=(0, 0, 255))

            # print(corner[0,:,:].shape, imgpts_rpj[:,0,:].shape)
            error = cv2.norm(corner[0, :, :], imgpts_rpj[:, 0, :], cv2.NORM_L2) / len(imgpts_rpj[:, 0, :])
            tot_error = tot_error + error

            length_of_axis = 0.05
            ProjectImage = cv2.drawFrameAxes(ProjectImage, cameraMatrix, distCoeffs, rvec, tvec, length_of_axis)

            # print( "total error: {}".format(error/len(imgpts)) )

            for point in imgpts_rpj.astype(int):
                ProjectImage = cv2.circle(ProjectImage, tuple(point[0]), 2, 255, -1)

        # print( "total error: {}".format(tot_error/len(ids)) )

        # cv2.imshow("ProjectImage", ProjectImage)
        return ProjectImage, ids, tvecs, rvecs

    def save_tagpose(self, camnum, vidnum, folder=None, name=None):
        if name == None:
            if folder == None:
                folder = self.cam_dir

            name = (
                folder
                + "\\"
                + self.trial_name
                + "_"
                + "cam"
                + str(camnum)
                + "_"
                + "vid"
                + str(vidnum)
                + "_tagpose.json"
            )

        # name='cam3_vid3_tagpose.json'
        tag_pose = self.get_pose(camnum, vidnum)
        print("Saving: ", name)
        tag_pose.to_json(name, orient="split")

    def save_intrinsics(self, folder=None, name=None):
        if name == None:
            if folder == None:
                folder = self.cam_dir

            name = folder + "\\" + self.trial_name + "_intrinsics.json"

        # name='cam3_vid3_tagpose.json'
        print("Saving: ", name)
        intrinsics = pd.DataFrame(self.intrinsics)
        intrinsics.to_json(name, orient="split")

    def load_tagpose(self, name):
        print("Loading: ", name)
        tag_pose = pd.read_json(name, orient="split")

        return tag_pose

    def load_intrinsics(self, folder=None, name=None):
        if name == None:
            if folder == None:
                folder = self.cam_dir

            name = folder + "\\" + self.trial_name + "_intrinsics.json"
        print("Loading: ", name)
        intrinsics = pd.read_json(name, orient="split")

        return intrinsics

        # print(self.intrinsics)

    def tag_COP(self, camnum, vidnum, name=None):
        if name == None:
            R0, T0, R25, T25 = self.glob_pose(camnum, vidnum)[:4]
        else:
            R0, T0, R25, T25 = self.glob_pose(camnum, vidnum, name=name, load=1)[:4]

        COM_0 = T0 + R0.apply(np.array([0, 0, -70]) / 1000)
        COM_25 = T25 + R25.apply(np.array([0, 0, -70]) / 1000)

        COP_0 = COM_0[:, 0:2] * 1.564
        COP_25 = COM_25[:, 0:2] * 2.187

        COP = COP_0 + COP_25

        return COP * 1000

    def comapre_COP(self, camnum, vidnum, file, name=None):
        COP = self.tag_COP(camnum, vidnum, name)

        Xcalc = COP[int(5.5 * 60) :, 0]
        Ycalc = COP[int(5.5 * 60) :, 1]

        order = 3
        fcut = 5

        b, a = scipy.signal.butter(order, fcut, fs=60)

        # Xcalc = signal.filtfilt(b, a, Xcalc)
        # Ycalc = signal.filtfilt(b, a, Ycalc)
        Xcalc = scipy.ndimage.median_filter(Xcalc, 15)
        Ycalc = scipy.ndimage.median_filter(Ycalc, 15)

        COP_gc = processCOP(file, 60)
        Xreal = COP_gc.Xfilt
        Yreal = COP_gc.Yfilt

        # cc = compareCOP(COP_X, COP_Y, X_gc, Y_gc)

        # fig, ax = plt.subplots(2, 1)

        plt.subplot(2, 1, 1)
        plt.plot(Xreal - np.mean(Xreal))
        plt.plot(Xcalc - np.mean(Xcalc))
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("X COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP X (mm)")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(Yreal - np.mean(Yreal))
        plt.plot(Ycalc - np.mean(Ycalc))
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("Y COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP Y (mm)")
        plt.grid()

        plt.tight_layout()
        plt.show()

        # print(4)


cnum = 3
vnum = 3

tt = COP_Tag(cam_dir=r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Cameras", name="sim_trunk", load=1)


nme = (
    r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\tagpose\\"
    + tt.trial_name
    + "_"
    + "cam"
    + str(cnum)
    + "_"
    + "vid"
    + str(vnum)
    + "_tagpose.json"
)

tt.comapre_COP(
    cnum, vnum, name=nme, file=r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\2025-6-11_2_28_Sim_flex_1_2_3.csv"
)

# tt.showtags_loop(cnum, vnum, name=nme)
# tt.tag_COP(cnum, vnum, name=nme)
# tt.get_pose(cnum, vnum, view=0)
# tt.save_tagpose(cnum, vnum)
