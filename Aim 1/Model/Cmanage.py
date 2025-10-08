import json
import math
import datetime
from io import StringIO
import subprocess

# from exiftool import ExifToolHelper

import os.path
import pandas as pd
import numpy as np
import cv2 as cv2
import cv2.aruco as aruco

import librosa
import librosa.display

import moviepy as mp
import matplotlib.pylab as plt

from ultralytics import YOLO
from Calibrate import calibrate
from scipy import signal
from scipy.fftpack import fft
from itertools import groupby
from operator import itemgetter

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy import VideoFileClip, concatenate_videoclips
import ffmpeg


####Class for managing and preocessing camera data from box
##Inputs, userdirectory of box on compueter subject Aim, subject ID, trial month, day and year
class c_manage:
    def __init__(self, cam_direct, vid_name):

        self.cam_direct = cam_direct
        self.vid_name = vid_name

        # Variables and paramters for ARUCO calibration board
        ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        self.ARUCO_DICT = ARUCO_DICT

        # Create grid board object we're using in our stream
        self.board = aruco.GridBoard(
            (5, 7),
            markerSeparation=0.008,
            markerLength=0.03,
            dictionary=ARUCO_DICT,
        )

    # Getting name and existance status for each camera for a given video number
    def check_vids(self, vid_num):
        cam_direct = self.cam_direct
        vid_name = self.vid_name

        vid_stat = np.zeros(7)
        vid_names = []
        for i in range(7):
            cam = i + 1
            name = (
                cam_direct
                + "\\Camera "
                + str(cam)
                + "\\"
                + vid_name
                + "_cam"
                + str(cam)
                + "_vid"
                + str(vid_num)
                + ".MP4"
            )
            stat = os.path.exists(name)

            if stat == 0:
                name = (
                    cam_direct
                    + "\\Camera "
                    + str(cam)
                    + "\\"
                    + vid_name
                    + "_cam"
                    + str(cam)
                    + "_vid"
                    + str(vid_num)
                    + ".mp4"
                )

            stat = os.path.exists(name)
            vid_stat[i] = stat
            vid_names.append(name)

        return vid_stat, vid_names

    # Function for retreving middle frame from video
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

    # function to manually naviatge frames to find synch light start and stop based on intial guess
    def manual_synch(self, video, start):

        if os.path.exists(video):
            cap = cv2.VideoCapture(video)

            fps = cap.get(cv2.CAP_PROP_FPS)

            # starting frame comes from initial guess
            frame = start

            # make frame 0 if number is negative
            if frame < 0:
                frame = 0

            # intitializing variable for loop end condition
            next = 1

            print("---------FPS: ", cap.get(cv2.CAP_PROP_FPS))

            while next == 1:

                print("FRAME ", round(frame))
                # pulling specified frame from vidoe
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

                ret, image = cap.read()

                width = cap.get(3)  # float `width`
                height = cap.get(4)  # float `height`

                # resizing window to 1/3 it's size
                image = cv2.resize(image, (int(width / 3), int(height / 3)))

                # Window name in which image is displayed
                window_name = "image"

                # Displaying the image
                cv2.imshow(window_name, image)

                # go back 1 frames if "a" key is pressed
                if cv2.waitKey(0) & 0xFF == ord("a"):
                    frame = frame - 1
                    next = 1
                # go forward 1 frames if "d" key is pressed
                elif cv2.waitKey(0) & 0xFF == ord("d"):
                    frame = frame + 1
                    next = 1
                # go forward FPS/2 (15 or 30) frames if "c" key is pressed
                elif cv2.waitKey(0) & 0xFF == ord("c"):
                    frame = frame + round(fps / 2)
                    next = 1
                # go back FPS/2 (15 or 30) frames if "z" key is pressed
                elif cv2.waitKey(0) & 0xFF == ord("z"):
                    frame = frame - round(fps / 2)
                    next = 1
                # go forward FPS/6 (5 or 10) frames if "e" key is pressed
                elif cv2.waitKey(0) & 0xFF == ord("e"):
                    frame = frame + round(fps / 6)
                    next = 1
                # go back FPS/6 (5 or 10) frames if "q" key is pressed
                elif cv2.waitKey(0) & 0xFF == ord("q"):
                    frame = frame - round(fps / 6)
                    next = 1
                # If any other key is pressed, end loop/program
                else:
                    next = 0

                # closing all open windows
                cv2.destroyAllWindows()

    # Function using ML model to detect when synch light is on or off in frame
    # option to rotate video to try object dectection for different results
    def visual_synch(self, video, rotate, dur=10, disp=1):

        # retrieving model (location should be changed for different user)
        model = YOLO(r"C:\Users\franc\Documents\GitHub\Panda-COP-Predicition\Aim 1\Model\synch_detect.pt")
        classNames = ["Synch Light Off", "Synch Light On"]
        # if video exists
        if os.path.exists(video):
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            tot_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            tot_t = tot_f / fps

            if tot_t > 130:
                dur = (tot_t - (120)) + 2

            if tot_f / fps < 15:
                print("Video less than 15s")
                return [3, 3], 0

            # print("dur", dur)

            # maximum frame for "dur" amount of seconds
            max_frame = round(fps * dur)

            width = cap.get(3)  # float `width`
            height = cap.get(4)  # float `height`

            # cap.set(3, int(width / 3))
            # cap.set(4, int(height / 3))

            # initializing start frame
            frame = 0
            # initialzing variable for storing previous frame classification
            prev_cls = 0

            # initializing variables for final result
            start = 0
            stop = 0
            success = 0

            # will only display until max frame is reached
            while True and frame < max_frame:
                succ, img = cap.read()

                if succ == 1:
                    frame = frame + 1
                    # rotate video is specified in function input
                    if rotate == 1:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        img = cv2.resize(img, (int(height / 3), int(width / 3)))
                    else:
                        img = cv2.resize(img, (int(width / 3), int(height / 3)))

                    results = model(img, stream=1, verbose=False)

                    # drawing bounding box for results
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            w, h = x2 - x1, y2 - y1

                            # classification for result (0=synch light off , 1=synch light off)
                            cls = int(box.cls[0])

                            # checking if
                            diff = cls - prev_cls
                            # if classification changes from OFF to ON mark that as a start
                            if diff == 1:
                                start = frame
                                print("------------------------START", start)
                            # if classification changes from ON to OFF, and it's been on for roughly a second, mark that as a stop
                            if (diff == -1) & (frame - start >= fps - 2) & (frame - start <= fps + 2):
                                stop = frame
                                print("------------------------STOP", stop)
                                print("VIDEO: ", start, stop)
                                success = 1

                                cv2.destroyAllWindows
                                # return results if stop is found
                                return [start, stop], success

                            # update previous classification
                            prev_cls = cls

                            # color for boundng box
                            colour = self.getColours(cls)

                            # draw the rectangle
                            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

                            conf = math.ceil((box.conf[0] * 100)) / 100

                            # cvzone.putTextRect(img, f"{name} " f"{conf}", (max(0, x1), max(35, y1)), scale=0.5)
                            # cv2.putText(frame, f"{classNames[int(box.cls[0])]} {box.conf[0]:.2f}", (x1, y1))
                    if disp == 1:
                        cv2.imshow("Image", img)
                        cv2.waitKey(1)

                else:
                    # if there's an error reading the frame
                    print(frame)

            cv2.destroyAllWindows
            # returns incorrect start and stop time, and success is inficated as 0
            return [start, stop], success
        else:
            return [0, 0], 0

    # Pull and save audio wav file from MP4 video
    def save_wav(self, video):
        # name of saved wav file, should be changed for different user
        name = r"C:\Users\\franc\\Documents\\GitHub\\PANDA-Gym-Data-Processing\\synch_light\\test.wav"

        # save audio as wav file from mp4, with temporary name
        clip = mp.VideoFileClip(video)
        clip.audio.write_audiofile(name, codec="pcm_s16le")

        # return name/location of saved wav file
        return name

    # Find start/stop time based on synch light beep
    def audio_synch(self, video, save, thresh=0.95, val=30, dur=10):
        try:
            # if video exists
            if os.path.exists(video):
                success = 1

                cap = cv2.VideoCapture(video)
                fps = round(cap.get(cv2.CAP_PROP_FPS))

                # if save variable is 1, save wav file, if not check existing file
                if save == 1:
                    audio = self.save_wav(video)
                else:
                    # audio file name
                    audio = r"C:\Users\franc\Documents\GitHub\PANDA-Gym-Data-Processing\synch_light\test.wav"

                # load wav file
                y, sr = librosa.load(audio, duration=dur)
                # t = np.linspace(0, (len(y) - 1), num=len(y)) / sr * 30

                # empty matrix for storing start and stop times
                start_stop = []

                # bandpass filter only looking between 2000-3000 Hz
                order = 2
                fcut_l = 2048
                fcut_h = 3000

                b, a = signal.butter(order, [fcut_l, fcut_h], fs=sr, btype="band")
                # band pass pilter when Mel spectogram is used
                # b, a = signal.butter(order, [5000, 9000], fs=sr, btype="band")

                y = signal.filtfilt(b, a, y)

                # run fourier transform on data
                d = librosa.stft(y, n_fft=2048)

                # Mel spectogram
                # S = librosa.feature.melspectrogram(
                #     y=y,
                #     sr=sr,
                #     n_mels=128 * 2,
                # )

                # find time scale for spectogram
                dt = np.shape(d)[1] / dur

                # convert to db
                sdb = librosa.amplitude_to_db(np.abs(d), ref=np.max)
                # conversion for mel spectogram
                # sdb = librosa.amplitude_to_db(S, ref=np.max)

                # find offset of when audio starts in file
                # looks minumum column where there's no sounds
                t_off = np.where(sdb > np.min(sdb))[1]
                t_off = np.min(t_off)

                # set anything less than -"val" to 5, isolating desired sound
                sdb[sdb < -val] = 5

                # look for anywhere along the x axis where value is less than 2/3 of -val (desired db)
                streak_x = np.where(sdb < -val / 1.5)[0]

                # store the unique number of rows with desired value
                cols = np.unique(streak_x)

                # calculating thresh*100 % of minimum value
                ymin = np.min(sdb) * thresh

                # iterating through unque columns
                for num in cols:
                    # among unique columns, look for which rows where the sound is between 0-ymin
                    streak = np.where((sdb[num, :] < 0) & (sdb[num, :] > ymin))[0]
                    # print(streak)

                    # loop to find continious numbers
                    for k, g in groupby(enumerate(streak), lambda x: x[0] - x[1]):

                        group = map(itemgetter(1), g)

                        group = list(map(int, group))

                        # look if difference between first and last number in continious row
                        diff = group[-1] - group[0]
                        # if difference is rougly between 1 second +5, -10
                        if (diff >= dt - 10) & (diff <= dt + 5):
                            start_stop.append((group[0], group[-1]))
                            # prints row number and start stop ranges
                            # useful for accuracy verification (start and stop times should be simialr across row if not, results will be inaccurate )
                            print("Row: ", num, " Start_Stop", [group[0], group[-1]])

                # Plot spectogram
                fig, ax = plt.subplots(figsize=(10, 5))
                img = librosa.display.specshow(sdb, x_axis="time", y_axis="log", ax=ax)
                ax.set_title("Spectogram ", fontsize=20)
                fig.colorbar(img, ax=ax, format=f"%0.2f")

                # find mean of start and stop times
                start_stop = np.mean(start_stop, 0)

                # start stop time converted to frames while accounting for time offset
                # start, stop = round((start_stop[0] - t_off) / dt * fps), round((start_stop[1] - t_off) / dt * fps)

                # start stop time converted to frames while accounting for 1/2 of time offset
                start2, stop2 = round((start_stop[0] - (t_off / 2)) / dt * fps), round(
                    (start_stop[1] - (t_off / 2)) / dt * fps
                )

                # print("AUDIO: ", start, stop)
                print("AUDIO2: ", start2, stop2)

                # Vertical lines displaying start and stop times, useful for accuracy verification
                plt.vlines(
                    x=start_stop[0] / dt,
                    ymin=0,
                    ymax=8200,
                    linewidth=1.5,
                    colors="g",
                )

                plt.vlines(
                    x=start_stop[1] / dt,
                    ymin=0,
                    ymax=8200,
                    linewidth=1.5,
                    colors="g",
                )

                # Vertical line for displaying time offset
                # plt.vlines(x=t_off / dt, ymin=0, ymax=8200, linewidth=1, colors="b", linestyles="dotted")

                plt.show()

                return [start2, stop2], success

            # return wrong start,stop time if some kind of error occurs, and indiacte success as 0
        except:
            success = 0
            print("Error finding correct reults")
            return [0, 0], success

    def start_stop(self, video, duration=10, display=1):
        # check video for start/stop frame
        v_start_stop, v_success = self.visual_synch(video, rotate=0, dur=duration, disp=display)

        # if initial video check is inaccurate,rotate video, it can make a difference
        if v_success == 0:
            print("trying again rotated")
            v_start_stop, v_success = self.visual_synch(video, rotate=1, dur=duration, disp=display)

        # if video is still unable to retuen results check audio
        if v_success == 0:
            print("Visula synch failed")
            return [0, 0], v_success
        # # if video gets results, return video results
        # else:
        return v_start_stop, v_success

    # get bounding box color
    def getColours(self, cls_num):

        base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        color_index = cls_num % len(base_colors)
        increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
        color = [
            base_colors[color_index][i] + increments[color_index][i] * (cls_num // len(base_colors)) % 256
            for i in range(3)
        ]
        return tuple(color)

    def tringulate(self, vid_num, delay=0):
        stat, names = self.check_csvs(vid_num)

        cams = np.array([1, 2, 3, 4, 5, 6, 7])

        stat = stat.astype(bool)

        available = cams[stat]
        print(available)

        fps = np.array([60, 60, 30, 30, 30, 30, 30])
        skip = delay * fps[stat]

        names = np.array(names)[stat]

        starts = np.zeros_like(available)
        stops = np.zeros_like(available)

        for i in range(len(available)):
            starts[i], stops[i] = self.pull_synch_time(available[i], vid_num)

        if os.path.exists(self.cam_direct + "\\" + filename + "_intrinsics.json") and os.path.exists(
            self.cam_direct + "\\" + filename + "_extrinsics.json"
        ):
            self.load_params()
        else:
            self.intrinsics_orientation(view=0)
            self.save_params()
        # intrinsics = self.intrinsics_final
        # extrinsics = self.extrinsics_final

        T = tringulatepose(
            names=names,
            cams=available,
            start=starts + skip,
            intrinsics=self.intrinsics_final,
            extrinsics=self.extrinsics_final,
        )

        T.check_combos()
        # T.trinagulate_all(1, 4)
        # T.SBA()

        names = np.array(self.name_4)[stat]
        self.T = T
        self.cams = available

        #

    def overlay_reproj(self, vid_num):

        available = self.cams

        # print(np.asarray(names))
        # print(np.asarray(status))

        for j in range(len(available)):
            # idx = np.where(available == available[j])[0][0]
            print(available[j], names[available[j] - 1])
            self.T.overlay_pose(available[j], vidname=names[available[j] - 1], compare=1)

    # function to find orientation of video based on static calibration
    def intrinsics_orientation(self, names, status):

        intrinsics = {}
        extrinsics = {}

        # look for all cameras
        for i in range(7):
            # i = 3
            print("Camera: ", i + 1)
            if status[0] == 1:
                # make 4 copies of image
                vid = names[i]

                cam = i + 1
                fov, lens, rot = self.check_metadata(vid)

                # use 4 images to calibration instatiage calibration object (based on calibration class)
                calib = calibrate(vid, lens, fov)

                # returns calibration results
                orient, ext, int = calib.intrinsics_orientation(view=0)
                print(orient)

                intrinsics.update({"cam" + str(cam): int})
                extrinsics.update({"cam" + str(cam): ext})

        # update intrinsics and extrinsics
        self.extrinsics_final = extrinsics
        self.intrinsics_final = intrinsics

        return orient

    # def intrinsics_orr(self):
    #     status = self.stat_1
    #     names = self.name_1

    #     # look for all cameras
    #     for i in range(7):
    #         # i = 3
    #         print("Camera: ", i + 1)
    #         if status[0] == 1:
    #             # make 4 copies of image
    #             image = self.pull_frames_static(names[i])
    #             image1 = self.pull_frames_static(names[i])
    #             image2 = self.pull_frames_static(names[i])
    #             image3 = self.pull_frames_static(names[i])

    #             # use 4 images to calibration instatiage calibration object (based on calibration class)
    #             calib = calibrate(self.userdirect, image, image1, image2, image3)

    #             # r, extr, intr = calib.intrinsics_orientation(i + 1)
    #             # print(r)

    #             # returns calibration results
    #             cam_ex1, rerror = calib.calibrate_extrinsics(i + 1, image, self.intrinsics[0])
    #             cam_ex2, rerror = calib.calibrate_extrinsics(i + 1, image1, self.intrinsics[1])
    #             cam_ex3, rerror = calib.calibrate_extrinsics(i + 1, image2, self.intrinsics[2])
    #             cam_ex4, rerror = calib.calibrate_extrinsics(i + 1, image3, self.intrinsics[3])

    #             rvs = [cam_ex1["Rvec"].T, cam_ex2["Rvec"].T, cam_ex3["Rvec"].T, cam_ex4["Rvec"].T]

    #             # print(rvs)

    #             for rv in rvs:
    #                 rv[rv < 0] = -1
    #                 rv[rv > 0] = 1

    #             print(rvs)

    def check_metadata(self, video):

        fov = "None"
        lens = "None"
        rot = 0

        # if video exists
        if os.path.exists(video):

            exiftool_command = [
                "exiftool",
                "-FieldOfView",
                "-Rotation",
                "-Model",
                "-LensSerialNumber",
                "-ee",
                "-csv",
                video,
            ]
            # exiftool_command = ["exiftool", "-FieldOfView", "-ee", video]
            k = subprocess.run(exiftool_command, shell=True, capture_output=True, text=True)
            out = pd.read_csv(StringIO(k.stdout))

            if "FieldOfView" in out:
                fov = out["FieldOfView"].item()

            if "LensSerialNumber" in out:
                lens = out["LensSerialNumber"].item()

            if "Rotation" in out:
                rot = out["Rotation"].item()

        else:
            print("Video Not Avaialble")

        return fov, lens, rot

    def json_serialize(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # function to save intrinsic and extrinsic parameters
    def save_params(self):
        # savefolder = r"C:\\Users\\franc\Documents\\GitHub\\PANDA-Gym-Data-Processing\\calib_videos"
        savefolder = self.cam_direct

        with open(savefolder + "\\intrinsics_sim.json", "w") as outfile:
            print("Saving Intrinsics")
            json.dump(self.intrinsics_final, outfile, default=self.json_serialize)

        with open(savefolder + "\\extrinsics_sim.json", "w") as outfile:
            print("Saving Extrinsics")
            json.dump(self.extrinsics_final, outfile, default=self.json_serialize)


# sim = c_manage(cam_direct=r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Cameras", vid_name="sim_trunk")
# nme = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Cameras\Camera 2\sim_trunk_cam2_vid5.MP4"
# sim.manual_synch(nme, 140)
# _, vid_names = sim.check_vids(4)
# print(np.array(vid_names))
# sim.start_stop(r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Cameras\Camera 1\sim_trunk_cam1_vid3.MP4")
