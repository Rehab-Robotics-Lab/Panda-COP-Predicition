import pandas as pd
import cv2 as cv2
import cv2.aruco as aruco
import pickle
import numpy as np
import datetime as dt
from Cmanage import c_manage
from ProcessPose_3D import processpose
from Calibrate import calibrate
import scipy
from CompareCOP import compareCOP
from ProcessCOP import processCOP
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class COP_Tag:
    def __init__(self, cam_dir, name, ID_up=0, ID_low=25, ID_len=40 / 1000, calibnum=1, load=0):
        self.trial_name = name
        self.cam_dir = cam_dir

        sim = c_manage(cam_direct=cam_dir, vid_name=name)

        calib_status, calib_names = sim.check_vids(calibnum)

        # sim.intrinsics_orientation(calib_names, calib_status)
        # print(sim.intrinsics_final)

        if name == "sim_trunk":
            self.intrinsics = self.load_intrinsics(name=None).to_dict()
        else:
            if load == 0:
                sim.intrinsics_orientation(view=0, vidnum=calibnum)
            elif load == 1:
                sim.load_params()

            self.intrinsics = sim.intrinsics_final

        self.calib_names, self.calib_status = calib_names, calib_status

        self.ID_up, self.ID_low, self.ID_len = ID_up, ID_low, ID_len
        self.sim = sim

    def showtags(self, j):
        plt.cla()

        tv_up = self.tv_up_glob[j]
        rv_up = self.rv_up_glob[j]
        tv_low = self.tv_low_glob[j]
        rv_low = self.rv_low_glob[j]

        self.ax.plot3D([tv_up[0], tv_low[0]], [tv_up[1], tv_low[1]], [tv_up[2], tv_low[2]], "o")

        # print(self.tvec_glob.flatten()[2] - tv_up[2], self.tvec_glob.flatten()[2] - tv_low[2])
        Rg = R.from_matrix(np.eye(3))
        Tg = np.array([0, 0, 0])

        self.ax = self.show_Ti(self.ax, Tg, Rg, scale=0.2, ID="G")
        self.ax = self.show_Ti(self.ax, self.tvec_cam.flatten(), self.rvec_cam, scale=0.2, ID="C")
        self.ax = self.show_Ti(self.ax, tv_up, rv_up, ID=self.ID_up)
        # com = tv_low + rv_low.apply(np.array([0, -10, -50]) / 1000)
        # self.ax = self.show_Ti(self.ax, com, rv_low, ID="m")
        self.ax = self.show_Ti(self.ax, tv_low, rv_low, ID=self.ID_low)

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
                self.rv_up_glob,
                self.tv_up_glob,
                self.rv_low_glob,
                self.tv_low_glob,
                self.rvec_cam,
                self.tvec_cam,
            ) = self.glob_pose(camnum, vidnum)
        else:
            (
                self.rv_up_glob,
                self.tv_up_glob,
                self.rv_low_glob,
                self.tv_low_glob,
                self.rvec_cam,
                self.tvec_cam,
            ) = self.glob_pose(camnum, vidnum, name=name, load=1)
        print("Displaying Pose")

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        # print(np.linspace(1,max_frames,max_frames))

        ani = FuncAnimation(
            self.fig, self.showtags, interval=1, frames=np.linspace(1, self.max_frames, self.max_frames, dtype=int)
        )
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

        R_orient = R.from_matrix(np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))

        R_glob_cor = R_glob * R_orient

        if load == 0:
            tag_pose = self.get_pose(camnum, vidnum)
        else:
            tag_pose = self.load_tagpose(name=name)

        tag_up = tag_pose.loc[(tag_pose["ID"] == self.ID_up)]
        tag_low = tag_pose.loc[(tag_pose["ID"] == self.ID_low)]

        self.max_frames = np.max(tag_low["frame"])

        rv_up = tag_up["rvecs"]
        tv_up = tag_up["tvec"]

        rv_low = tag_low["rvecs"]
        tv_low = tag_low["tvec"]

        print("Adjusting extrinsics")

        Rc = R_glob_cor.inv()
        Tc = Rc.apply(-tvec_glob.T)

        rv_up = np.array(rv_up.to_list())
        rv_up = R.from_rotvec(rv_up)
        tv_up = np.array(tv_up.to_list())

        rv_low = np.array(rv_low.to_list())
        rv_low = R.from_rotvec(rv_low)
        tv_low = np.array(tv_low.to_list())

        R_up = R_glob_cor.inv() * rv_up
        R_low = R_glob_cor.inv() * rv_low

        T_up = Rc.apply(-tvec_glob.T + tv_up)
        T_low = Rc.apply(-tvec_glob.T + tv_low)

        return R_up, T_up, R_low, T_low, Rc, Tc

    def combine_vecs(self, cams, vidnum, folder):
        print("Combining vecs")

        n = 100000000

        for camnum in cams:
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
            start, stop = self.pull_synch_time(camnum, vidnum)

            temp = self.load_tagpose(name, start)
            temp = temp.loc[(temp["ID"] == 0)]
            if camnum < 4:
                temp = temp.iloc[::2]

            n = min(temp.shape[0], n)

        Rot_up = np.zeros((n, 3, len(cams)))
        Tr_up = np.zeros((n, 3, len(cams)))
        Rot_low = np.zeros((n, 3, len(cams)))
        Tr_low = np.zeros((n, 3, len(cams)))
        # Rot_cam = np.zeros((3, len(cams)))
        # Tr_cam = np.zeros((3, len(cams)))

        print("N =", n)

        i = 0

        for camnum in cams:
            cam_ex, rerror = self.glob_extrinsics(camnum, view=0)
            rvec_glob = np.asarray(cam_ex["Rvec"])
            tvec_glob = np.asarray(cam_ex["Tvec"])

            R_glob = R.from_rotvec(rvec_glob.T)
            R_orient = R.from_matrix(np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
            R_glob_cor = R_glob * R_orient
            Rc = R_glob_cor.inv()

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
            start, stop = self.pull_synch_time(camnum, vidnum)
            tag_pose = self.load_tagpose(name, start)

            tag_up = tag_pose.loc[(tag_pose["ID"] == self.ID_up)]
            tag_low = tag_pose.loc[(tag_pose["ID"] == self.ID_low)]

            rv_up, tv_up = tag_up["rvecs"], tag_up["tvec"]
            rv_low, tv_low = tag_low["rvecs"], tag_low["tvec"]

            if camnum < 4:
                rv_up = rv_up[::2]
                tv_up = tv_up[::2]

                rv_low = rv_low[::2]
                tv_low = tv_low[::2]

            # print("Big sape ", (np.array(rv_up[:n].to_list())))

            rv_up, tv_up = self.interpolate_tag_pose(rv_up, tv_up, n)
            rv_low, tv_low = self.interpolate_tag_pose(rv_low, tv_low, n)

            # rv_up = np.array(rv_up[:n].to_list())
            rv_up = R.from_rotvec(rv_up)
            # tv_up = np.array(tv_up[:n].to_list())

            # rv_low = np.array(rv_low[:n].to_list())
            rv_low = R.from_rotvec(rv_low)
            # tv_low = np.array(tv_low[:n].to_list())

            R_up = R_glob_cor.inv() * rv_up
            R_low = R_glob_cor.inv() * rv_low

            T_up = Rc.apply(-tvec_glob.T + tv_up)
            T_low = Rc.apply(-tvec_glob.T + tv_low)

            Rot_up[:, :, i] = R.as_rotvec(R_up)
            Tr_up[:, :, i] = T_up
            Rot_low[:, :, i] = R.as_rotvec(R_low)
            Tr_low[:, :, i] = T_low

            i = i + 1

        Rot_up = R.from_rotvec(np.mean(Rot_up, axis=2))
        Tr_up = np.mean(Tr_up, axis=2)
        Rot_low = R.from_rotvec(np.mean(Rot_low, axis=2))
        Tr_low = np.mean(Tr_low, axis=2)

        return Rot_up, Tr_up, Rot_low, Tr_low

    def interpolate_tag_pose(self, Rvec, Tvec, n):

        Rv = np.array(Rvec[:n].to_list())
        Tv = np.array(Tvec[:n].to_list())

        Rv_sum = np.sum(Rv, axis=1)

        if (Rv_sum == 0).any():
            # print("Interpolation detected")
            idx = np.where(Rv_sum == 0)[0]
            # print(idx)

            nan_array = [np.nan, np.nan, np.nan]

            Rv[idx, :] = nan_array
            Tv[idx, :] = nan_array

            Rv = pd.DataFrame(Rv)
            Rvec_corrected = Rv.interpolate(method="linear", axis=0).to_numpy()
            Tv = pd.DataFrame(Tv)
            Tvec_corrected = Tv.interpolate(method="linear", axis=0).to_numpy()

        else:
            Rvec_corrected, Tvec_corrected = Rv, Tv

        return Rvec_corrected, Tvec_corrected

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
        print("Getting tag extrinsics for camera ", str(camnum), "video ", str(vidnum))
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
                    ids = np.atleast_1d(ids)

                    if self.ID_up in ids:
                        idx_up = np.where(ids == self.ID_up)[0].item()

                        row_up = pd.DataFrame(
                            {
                                "ID": ids[idx_up],
                                "tvec": [tvecs[:, idx_up]],
                                "rvecs": [rvecs[:, idx_up]],
                                "frame": [framenum],
                            }
                        )
                    else:
                        row_up = pd.DataFrame(
                            {
                                "ID": [self.ID_up],
                                "tvec": [[0, 0, 0]],
                                "rvecs": [[0, 0, 0]],
                                "frame": [framenum],
                            }
                        )
                    if self.ID_low in ids:
                        idx_low = np.where(ids == self.ID_low)[0].item()

                        row_low = pd.DataFrame(
                            {
                                "ID": ids[idx_low],
                                "tvec": [tvecs[:, idx_low]],
                                "rvecs": [rvecs[:, idx_low]],
                                "frame": [framenum],
                            }
                        )
                    else:
                        row_low = pd.DataFrame(
                            {
                                "ID": [self.ID_low],
                                "tvec": [[0, 0, 0]],
                                "rvecs": [[0, 0, 0]],
                                "frame": [framenum],
                            }
                        )

                    tag_pose = pd.concat([tag_pose, row_up], ignore_index=True)
                    tag_pose = pd.concat([tag_pose, row_low], ignore_index=True)

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
        markerlength = self.ID_len

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

    def save_tagpose(self, camnum, vidnum, folder, name=None):
        if name == None:

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

    def load_tagpose(self, name, start=None):
        # print("Loading: ", name)
        tag_pose = pd.read_json(name, orient="split")

        if start != None:
            tag_pose = tag_pose.loc[tag_pose["frame"] >= start]

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

    def adjust_Fn(self, M_low, M, Fn):
        m_head, m_up, m_low = M[0], M[1], M[2]

        Fn_mean = np.mean(Fn[0])
        # Fn_mean = np.mean(Fn)
        m_mat = 1.325

        # Fn_tot = Fn / Fn_mean * (m_head + m_up + m_low)
        # Fn_tot = Fn - Fn_mean + (m_head + m_up + m_low + m_mat)
        # F_mat=Fn-(m_head + m_up + m_low)

        # or

        # plt.plot(Fn_tot / Fn_mean * (M_head + M_up + M_low))
        # plt.plot(np.ones_like(Fn_tot) * (M_head + M_up + M_low))
        # plt.show()
        # quit()

        M_low_d = Fn - (m_head + m_up + m_mat)
        # print(np.mean(M_low_d))
        # print(Fn_mean, m_low)
        # M_low_d = Fn - Fn_mean + m_low\
        # print((np.shape(M_low)[0], np.shape(M_low_d)[0]))
        n = min(np.shape(M_low)[0], np.shape(M_low_d)[0])

        M_low_d = M_low_d[0:n]

        return M_low_d

    def tag_COP(self, cams, vidnum, folder, posefile=None, copfile=None, addHead=True):
        R_up, T_up, R_low, T_low = self.combine_vecs(cams, vidnum, folder)

        rtrunk = 91.39
        g = 9.81

        # posefile = None
        copfile = None

        m_head = 1.031
        m_up = 1.564
        m_low = 2.187
        # m_low = 1.175

        # print(m_head)

        M = [m_head, m_up, m_low]

        n_tag = np.shape(T_low)[0]
        # print(n_tag)

        M_head = np.ones((n_tag)) * m_head
        M_up = np.ones((n_tag)) * m_up
        M_low = np.ones((n_tag)) * m_low

        COM_up = T_up
        COM_low = T_low + R_low.apply(np.array([0, 7.7, (rtrunk - 80)]) / 1000)

        COP_up = COM_up[:, 0:2].T * M_up
        COP_low = COM_low[:, 0:2].T * M_low

        COM_head_proj = T_up + R_up.apply(np.array([0, -130, 0]) / 1000)
        # COM_head2 = scipy.ndimage.median_filter(COM_head2, size=[150, 1])
        COM_head_proj = scipy.ndimage.median_filter(COM_head_proj, size=[60, 1])

        n = n_tag

        if copfile != None:
            COP_gc = processCOP(copfile, 60)
            Fn_tot = COP_gc.Rfilt[::2]

            # temp_COP = COM_low[:, 0:2].T * M_low

            M_low_d = self.adjust_Fn(M_low, M, Fn_tot)
            n = np.shape(M_low_d)[0]

            COP_low = COM_low[0:n, 0:2].T * M_low_d
            M_low = M_low_d

        if posefile == None and addHead == False:
            COP_head = np.zeros_like(COP_up)
            m_head = 0
        elif posefile == None and addHead == True:
            COP_head = COM_head_proj[:, 0:2].T * m_head
        else:
            # initilazlise pose file
            pose = processpose(posefile)
            pose.IK_init()

            # find head location
            T_head = pose.mid_face.T
            n_pose = np.shape(T_head)[0]

            # find minimum length
            n = min(n, n_pose)

            # find location of neck in pose and tag frames
            T_neck_pose = np.matrix([pose.X[:, 1], pose.Y[:, 1], pose.Z[:, 1]]).T

            T_neck_tag = T_up + R_up.apply(np.array([0, -55, 0]) / 1000)

            # find offset of neck locations
            # T_h_tag-T_h_pose==T_n_tag-T_n_pose
            T_diff = np.mean(T_neck_pose[0:15, :]) - np.mean(T_neck_tag[0:15, :])

            # Offset head location to be in the tag frame
            T_head_off = T_head[0:n, :] - T_diff
            # COM_head = np.ones_like(T_head_off) * np.mean(T_head_off, axis=0).T
            COM_head = scipy.ndimage.median_filter(T_head_off, size=[50, 1])
            # COM_head = COM_head + (70 / 1000)
            COP_head = COM_head[:, 0:2].T * m_head

            # plt.plot(COM_head[0:n, 0].T * 1000)
            # plt.plot(COM_head2[0:n, 0].T * 1000)

            # plt.legend(["pose", "proj"])
            # plt.show()
            # COP_head[0:n, 0] = COP_up[0:n, 0]

        COP = (COP_head[:, 0:n] * g + COP_up[:, 0:n] * g + COP_low[:, 0:n] * g) / (
            M_head[0:n] * g + M_up[0:n] * g + M_low[0:n] * g
        )

        # COP2 = COP_head[:, 0:n] + COP_up[:, 0:n] + COP_low[:, 0:n]
        # COP[0, :] = COP2[0, :]

        # plt.plot(COP_head[0, 0:n] * 1000)
        # plt.plot(COP_up[0, 0:n].T * 1000)
        # plt.plot(COP_low[0, 0:n].T * 1000)
        # plt.plot(COP[0, 0:n].T * 1000)

        # plt.legend(["head", "up", "lower", "tot"])
        # plt.show()
        # quit()

        return COP.T * 1000

    def pose_COP(self, posefile, copfile=None):

        pose = processpose(posefile)
        pose.IK_init()

        # plt.plot(pose.X[:, 0].T)
        # plt.plot(pose.mid_shoulder[0, :].T)
        # plt.plot(pose.mid_hip[0, :].T)
        # plt.show()

        ## SHOULDERS
        shoulder_angles = np.stack((pose.thet1s[0, :], pose.thet2s[0, :]))
        R_up = R.from_euler("zy", shoulder_angles.T)
        T_up = pose.mid_shoulder.T

        ## HIPS
        hip_angles = np.stack((pose.thet1h[0, :], pose.thet2h[0, :], pose.thet3h[0, :]))
        R_low = R.from_euler("xzy", hip_angles.T)
        T_low = pose.mid_hip.T

        T_head = pose.mid_face.T
        n_pose = np.shape(T_head)[0]

        rtrunk = 91.39

        COM_head = T_head

        # COM_up = T_up
        COM_up = T_up + R_up.apply(np.array([0, -75, 0]) / 1000)

        COM_low = T_low + R_low.apply(np.array([0, 0, (rtrunk - 80)]) / 1000)

        m_head = 1.031
        m_up = 1.564
        m_low = 2.187

        M_head = np.ones((n_pose)) * m_head
        M_up = np.ones((n_pose)) * m_up
        M_low = np.ones((n_pose)) * m_low

        # copfile = None

        if copfile != None:
            M = [m_head, m_up, m_low]

            COP_gc = processCOP(copfile, 60)
            Fn_tot = COP_gc.Rfilt[::2]

            M_low = self.adjust_Fn(M_low, M, Fn_tot)

            n = np.shape(M_low)[0]
        else:
            n = n_pose

        COP_head = COM_head[0:n, 0:2].T * m_head
        COP_up = COM_up[0:n, 0:2].T * m_up
        COP_low = np.multiply(COM_low[0:n, 0:2].T, M_low)

        COP = (COP_head + COP_up + COP_low) / (M_head[0:n] + M_up[0:n] + M_low[0:n])

        # Tsh = np.linalg.inv(T_ups) @ T_uph
        # Th = T_ups @ Tsh
        return COP.T * 1000

    def compare_angles(self, cams, folder, vidnum, posefile, filter_window=10):
        Rup_tag, Tup_tag, Rlow_tag, Tlow_tag = self.combine_vecs(cams, vidnum, folder)

        tag_to_world = R.from_matrix(np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))

        # print(np.shape(Rup_tag))

        Rup_tag = Rup_tag * tag_to_world
        Rlow_tag = Rlow_tag * tag_to_world

        pose = processpose(posefile)
        pose.IK_init()

        ## SHOULDERS
        shoulder_angles = np.stack((pose.thet1s[0, :], pose.thet2s[0, :]))
        Rup_pose = R.from_euler("zy", shoulder_angles.T)
        Tup_pose = pose.mid_shoulder.T

        ## HIPS
        hip_angles = np.stack((pose.thet1h[0, :], pose.thet2h[0, :], pose.thet3h[0, :]))
        Rlow_pose = R.from_euler("xzy", hip_angles.T)
        Tlow_pose = pose.mid_hip.T

        pose_ang_low = Rlow_pose.as_euler("xzy", degrees=True)
        tag_ang_low = Rlow_tag.as_euler("xzy", degrees=True)
        tag_ang_low = scipy.ndimage.median_filter(self.interp_tagpose(tag_ang_low), [filter_window, 1])

        # print(Rup_tag[0], Rup_pose[0])

        n = min(np.shape(pose_ang_low)[0], np.shape(tag_ang_low)[0])

        t = (np.linspace(0, n / 30, num=n)).reshape(n, 1)

        print(np.mean(pose_ang_low, axis=0), np.nanmean(tag_ang_low, axis=0))
        # plt.plot(t, tag_ang_low[0:n, 2])

        plt.subplot(3, 1, 1)
        plt.plot(t, pose_ang_low[0:n, 0])
        plt.plot(t, tag_ang_low[0:n, 0])
        plt.legend(["Pose", "Tag"])
        plt.title("X")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(t, pose_ang_low[0:n, 1])
        plt.plot(t, tag_ang_low[0:n, 1])
        plt.legend(["Pose", "Tag"])
        plt.title("Z")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.grid()

        plt.subplot(3, 1, 3)
        plt.plot(t, pose_ang_low[0:n, 2])
        plt.plot(t, tag_ang_low[0:n, 2])
        plt.legend(["Pose", "Tag"])
        plt.title("Y")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.grid()

        plt.tight_layout()
        plt.show()

        # print(np.shape(pose_ang_low), np.shape(tag_ang_low))

    def interp_tagpose(self, angles, thresh=2.5):
        z_ang = np.abs(scipy.stats.zscore(angles, axis=0))
        # z_y = np.abs(scipy.stats.zscore(Y, axis=1))

        jump = np.where((z_ang >= thresh))

        angles[jump] = np.nan

        ang_interp = angles
        ang_interp = pd.DataFrame(angles).interpolate(method="linear", axis=1, limit_direction="both").to_numpy()

        return ang_interp

    def pull_synch_time(self, cam_num, vid_num):
        col_names = [
            ["cam1_start", "cam1_stop"],
            ["cam2_start", "cam2_stop"],
            ["cam3_start", "cam3_stop"],
            ["cam4_start", "cam4_stop"],
            ["cam5_start", "cam5_stop"],
            ["cam6_start", "cam6_stop"],
            ["cam7_start", "cam7_stop"],
        ]

        df = pd.read_excel(r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\synchlight_time.xlsx")

        # print(trial)

        vid = df.loc[(df["vid"] == vid_num) & (df["name"] == self.trial_name)]

        start = vid[col_names[cam_num - 1][0]].item()
        stop = vid[col_names[cam_num - 1][1]].item()

        return start, stop

    def comapre_COP(self, camnum, vidnum, file, folder, posefile):
        COP = self.tag_COP(camnum, vidnum, folder, posefile=posefile, copfile=file)
        # COP = self.pose_COP(posefile, copfile=file)

        # start, stop = self.pull_synch_time(camnum, vidnum)

        Xcalc = COP[:, 0]
        Ycalc = COP[:, 1]

        # order = 3
        # fcut = 5

        # b, a = scipy.signal.butter(order, fcut, fs=60)

        # Xcalc = signal.filtfilt(b, a, Xcalc)
        # Ycalc = signal.filtfilt(b, a, Ycalc)
        Xcalc = scipy.ndimage.median_filter(Xcalc, 15)
        Ycalc = scipy.ndimage.median_filter(Ycalc, 15)

        COP_gc = processCOP(file, 60)

        Xreal = COP_gc.Xfilt[::2]
        Yreal = COP_gc.Yfilt[::2]

        # plt.plot(Xreal)
        # plt.show()
        # quit()

        # cc = compareCOP(COP_X, COP_Y, X_gc, Y_gc)

        start = 200
        stop = -1

        Xreal, Yreal = Xreal[start:stop], Yreal[start:stop]
        Xcalc, Ycalc = Xcalc[start:stop], Ycalc[start:stop]

        # win = 2

        # Xcalc = scipy.ndimage.median_filter(Xcalc, win)
        # Ycalc = scipy.ndimage.median_filter(Ycalc, win)

        # print(np.mean(Xcalc), np.mean(Xreal))
        # print(np.mean(Ycalc) - np.mean(Yreal))

        cc = compareCOP(Xcalc, Ycalc, Xreal, Yreal, cam=1)
        cc.comp_XY()
        # cc.comp_ellipse()
        # cc.plot_cop_anim()

        # print(cc.metrics())
        print(cc.diff_metric())
        # print(4)


tt = COP_Tag(
    cam_dir=r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Cameras",
    name="sim_trunk_limbs2",
    load=1,
    ID_up=0,
    ID_low=50,
    ID_len=30 / 1000,
)

folder = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\tagpose"

# cams = [1, 2, 3, 4, 5]
cams = [4]


copfile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_limbs2_cop_vid12_flex_nolimb.csv"
posefile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\3D Pose Test\sim_trunk_limbs2_vid12_cams_2_6_both.csv"
vnum = 12


# copfile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_limbs_cop_vid4_side.csv"
# copfile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_limbs_cop_vid5_rot.csv"
# copfile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_limbs_cop_vid6_flex.csv"
# copfile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_clothed_cop_vid6_all.csv"

# copfile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_clothed_cop_vid7_side_limbs.csv"
# posefile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\3D Pose Test\sim_trunk_clothed_vid7_cams_5_6_both.csv"

# posefile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\3D Pose Test\sim_trunk_limbs_vid4_cams_2_6_both.csv"

# posefile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\3D Pose Test\sim_trunk_limbs_vid4_cams_2_6_both.csv"
# copfile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_limbs_cop_vid4_side.csv"


# posefile = None
tt.comapre_COP(cams, vnum, file=copfile, folder=folder, posefile=posefile)
tt.compare_angles(cams, vidnum=vnum, folder=folder, posefile=posefile)

# tt.save_tagpose(2, vnum, folder=r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\tagpose")


# for vnum in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
#     for cnum in [1, 2, 3, 4, 5, 7]:
#         tt.save_tagpose(cnum, vnum, folder=r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\tagpose")
