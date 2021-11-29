#!/usr/bin/python
import math

import numpy as np
import cv2  # OpenCV 4.5.4-dev
import argparse
import dewarp
import feature_matching
import optimal_seamline
import blending
import cropping
import os
import uuid

import time

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

# --------------------------------
# output video resolution
W = 2560
H = 1280
# --------------------------------
# field of view, width of de-warped image
FOV = 194.0
W_remap = 1380  # input width + overlap
# --------------------------------
# params for template matching
templ_shape = (60, 16)
offsetYL = 160
offsetYR = 160
maxL = 80
maxR = 80
# --------------------------------
# params for optimal seamline and multi-band blending
W_lbl = 120
blend_level = 7
# --------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
# --------------------------------

start = time.time()
average_time = []


def modify_start(t):
    global start
    start = t


def image_show(img, controllable_size=False):
    id = str(uuid.uuid4())

    if controllable_size:
        # 讓視窗可以自由縮放大小
        cv2.namedWindow(id, cv2.WINDOW_NORMAL)

    # 顯示圖片
    cv2.imshow(id, img)

    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Hcalc(cap, xmap, ymap):
    """Calculate and return homography for stitching process."""
    Mlist = []
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for frame_no in np.arange(0, frame_count, int(frame_count / 10)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            # defish / unwarp
            cam1 = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
            cam2 = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)
            cam1_gray = cv2.cvtColor(cam1, cv2.COLOR_BGR2GRAY)
            cam2_gray = cv2.cvtColor(cam2, cv2.COLOR_BGR2GRAY)

            # shift the remapped images along x-axis
            shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
            shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = cam2
            shifted_cams[:H, :int(W_remap / 2)] = cam1[:, int(W_remap / 2):]
            shifted_cams[:H, W - int(W_remap / 2):] = cam1[:, :int(W_remap / 2)]

            # find matches and extract pairs of correspondent matching points
            matchesL = feature_matching.getMatches_goodtemplmatch(
                cam1_gray[offsetYL:H - offsetYL, int(W / 2):],
                cam2_gray[offsetYL:H - offsetYL, :W_remap - int(W / 2)],
                templ_shape, maxL)
            matchesR = feature_matching.getMatches_goodtemplmatch(
                cam2_gray[offsetYR:H - offsetYR, int(W / 2):],
                cam1_gray[offsetYR:H - offsetYR, :W_remap - int(W / 2)],
                templ_shape, maxR)
            matchesR = matchesR[:, -1::-1]

            matchesL = matchesL + (int((W - W_remap) / 2), offsetYL)
            matchesR = matchesR + (int((W - W_remap) / 2) + int(W / 2), offsetYR)
            zipped_matches = list(zip(matchesL, matchesR))
            matches = np.int32([e for i in zipped_matches for e in i])
            pts1 = matches[:, 0]
            pts2 = matches[:, 1]

            # find homography from pairs of correspondent matchings
            M, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
            Mlist.append(M)
    M = np.average(np.array(Mlist), axis=0)
    print(M)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return M


def Hcalc_image(frame1, frame2, xmap, ymap):
    """Calculate and return homography for stitching process."""
    Mlist = []

    # defish / unwarp
    cam1 = cv2.remap(frame1, xmap, ymap, cv2.INTER_LINEAR)
    cam2 = cv2.remap(frame2, xmap, ymap, cv2.INTER_LINEAR)
    cam1_gray = cv2.cvtColor(cam1, cv2.COLOR_BGR2GRAY)
    cam2_gray = cv2.cvtColor(cam2, cv2.COLOR_BGR2GRAY)

    # shift the remapped images along x-axis
    shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
    shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = cam2
    shifted_cams[:H, :int(W_remap / 2)] = cam1[:, int(W_remap / 2):]
    shifted_cams[:H, W - int(W_remap / 2):] = cam1[:, :int(W_remap / 2)]

    # find matches and extract pairs of correspondent matching points
    matchesL = feature_matching.getMatches_goodtemplmatch(
        cam1_gray[offsetYL:H - offsetYL, int(W / 2):],
        cam2_gray[offsetYL:H - offsetYL, :W_remap - int(W / 2)],
        templ_shape, maxL)
    matchesR = feature_matching.getMatches_goodtemplmatch(
        cam2_gray[offsetYR:H - offsetYR, int(W / 2):],
        cam1_gray[offsetYR:H - offsetYR, :W_remap - int(W / 2)],
        templ_shape, maxR)
    matchesR = matchesR[:, -1::-1]

    matchesL = matchesL + (int((W - W_remap) / 2), offsetYL)
    matchesR = matchesR + (int((W - W_remap) / 2) + int(W / 2), offsetYR)
    zipped_matches = list(zip(matchesL, matchesR))
    matches = np.int32([e for i in zipped_matches for e in i])
    pts1 = matches[:, 0]
    pts2 = matches[:, 1]

    # find homography from pairs of correspondent matchings
    M, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
    Mlist.append(M)

    M = np.average(np.array(Mlist), axis=0)
    print(M)

    return M


def main_image(input_image_path, output_dir):
    frame = cv2.imread(input_image_path)
    frame1 = frame[:, :1280]
    frame2 = frame[:, 1280:]

    with open('parameters/raw_parameters.npy', 'wb') as f:
        # obtain xmap and ymap
        # Ws = W_remap = image width + overlap width
        # Wd = Hd = image height
        xmap, ymap = dewarp.buildmap(Ws=W_remap, Hs=H, Wd=H, Hd=H, fov=FOV)
        np.save(f, xmap)
        np.save(f, ymap)

        # calculate homography
        M = Hcalc_image(frame1, frame2, xmap, ymap)
        np.save(f, M)

        # calculate vertical boundary of warped image, for later cropping
        top, bottom = cropping.verticalBoundary(M, W_remap, W, H)
        with open('parameters/raw_parameters.pkl', 'wb') as outp:
            pickle.dump(top, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(bottom, outp, pickle.HIGHEST_PROTOCOL)

        # estimate empty (invalid) area of warped2
        EAof2 = np.zeros((H, W, 3), np.uint8)
        EAof2[:, int((W - W_remap) / 2) + 1:int((W + W_remap) / 2) - 1] = 255
        EAof2 = cv2.warpPerspective(EAof2, M, (W, H))
        np.save(f, EAof2)

        # de-warp
        cam1 = cv2.remap(frame1, xmap, ymap, cv2.INTER_LINEAR)
        cam2 = cv2.remap(frame2, xmap, ymap, cv2.INTER_LINEAR)

        # shift the remapped images along x-axis
        shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
        shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = cam2
        shifted_cams[:H, :int(W_remap / 2)] = cam1[:, int(W_remap / 2):]
        shifted_cams[:H, int(W - W_remap / 2):] = cam1[:, :int(W_remap / 2)]

        # warp cam2 using homography M
        warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
        warped1 = shifted_cams[:H]

        # crop to get a largest rectangle, and resize to maintain resolution
        warped1 = cv2.resize(warped1[top:bottom], (W, H))
        warped2 = cv2.resize(warped2[top:bottom], (W, H))

        cv2.imwrite(output_dir + 'shifted-1.png', warped1)
        cv2.imwrite(output_dir + 'warped-2.png', warped2)

        # image labeling (find minimum error boundary cut)
        mask, minloc_old = optimal_seamline.imgLabeling(
            warped1[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
            warped2[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
            warped1[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
            warped2[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
            (W, H), int(W_remap / 2) - W_lbl, W - int(W_remap / 2))
        np.save(f, mask)

        labeled = warped1 * mask + warped2 * (1 - mask)

        # fill empty area of warped1 and warped2, to avoid darkening
        warped1[:, int(W_remap / 2):W - int(W_remap /
                                            2)] = warped2[:, int(W_remap / 2):W - int(W_remap / 2)]
        warped2[EAof2 == 0] = warped1[EAof2 == 0]

        # multi band blending
        blended = blending.multi_band_blending(
            warped1, warped2, mask, blend_level)

        # write results from phases
        cv2.imwrite(output_dir + 'equirectangular-1.png', cam1)
        cv2.imwrite(output_dir + 'equirectangular-2.png', cam2)
        cv2.imwrite(output_dir + 'shifted.png', shifted_cams)
        cv2.imwrite(output_dir + 'output-labeled.png', labeled.astype(np.uint8))
        cv2.imwrite(output_dir + 'output-blended.png', blended.astype(np.uint8))

        end = time.time()
        print(end - start, 'second')


def main_image_simplified(input1, input2):
    # Load parameters
    with open('parameters/parameters.npy', 'rb') as f:
        xmap = np.load(f)
        ymap = np.load(f)
        M = np.load(f)
        EAof2 = np.load(f)
        mask = np.load(f)

        with open('parameters/parameters.pkl', 'rb') as inp:
            top = pickle.load(inp)
            bottom = pickle.load(inp)

            modify_start(time.time())
            frame1 = cv2.imread(input1)
            frame2 = cv2.imread(input2)

            if frame1.shape != frame2.shape:
                print('Images size are different.')
                return

            # obtain xmap and ymap
            # Ws = W_remap = image width + overlap width
            # Wd = Hd = image height
            '''xmap, ymap = dewarp.buildmap(Ws=W_remap, Hs=H, Wd=H, Hd=H, fov=FOV)'''

            # calculate homography
            '''M = Hcalc_image(frame1, frame2, xmap, ymap)'''

            # calculate vertical boundary of warped image, for later cropping
            '''top, bottom = cropping.verticalBoundary(M, W_remap, W, H)'''

            # estimate empty (invalid) area of warped2
            '''EAof2 = np.zeros((H, W, 3), np.uint8)
            EAof2[:, int((W - W_remap) / 2) + 1:int((W + W_remap) / 2) - 1] = 255
            EAof2 = cv2.warpPerspective(EAof2, M, (W, H))'''

            # de-warp
            cam1 = cv2.remap(frame1, xmap, ymap, cv2.INTER_LINEAR)
            cam2 = cv2.remap(frame2, xmap, ymap, cv2.INTER_LINEAR)

            # shift the remapped images along x-axis
            shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
            shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = cam2
            shifted_cams[:H, :int(W_remap / 2)] = cam1[:, int(W_remap / 2):]
            shifted_cams[:H, int(W - W_remap / 2):] = cam1[:, :int(W_remap / 2)]

            # warp cam2 using homography M
            warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
            warped1 = shifted_cams[:H]

            # crop to get a largest rectangle, and resize to maintain resolution
            warped1 = cv2.resize(warped1[top:bottom], (W, H))
            warped2 = cv2.resize(warped2[top:bottom], (W, H))

            # image labeling (find minimum error boundary cut)
            '''mask, minloc_old = optimal_seamline.imgLabeling(
                warped1[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
                warped2[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
                warped1[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
                warped2[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
                (W, H), int(W_remap / 2) - W_lbl, W - int(W_remap / 2))'''

            # Mask manually
            '''mask = np.zeros(warped1.shape, np.uint8)
            mask[:, 1278:3888] = 1'''

            labeled = warped1 * mask + warped2 * (1 - mask)  # labeled = warped1 * mask + warped2 * (1 - mask)

            '''# fill empty area of warped1 and warped2, to avoid darkening
            warped1[:, int(W_remap / 2):W - int(W_remap /
                    2)] = warped2[:, int(W_remap / 2):W - int(W_remap / 2)]
            warped2[EAof2 == 0] = warped1[EAof2 == 0]

            # multi band blending
            blended = blending.multi_band_blending(
                warped1, warped2, mask, blend_level)'''

            print(type(xmap), type(ymap), type(M), type(top), type(bottom), type(mask))
            # write results from phases
            cv2.imwrite(dir_path + '/output/image-simplified.png', labeled.astype(np.uint8))

            # release everything if job is finished
            cv2.destroyAllWindows()

            end = time.time()
            print(end - start, 'second')


def main_video(input, output):
    cap = cv2.VideoCapture(input)

    # define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, 30.0, (W, H))

    # obtain xmap and ymap
    xmap, ymap = dewarp.buildmap(Ws=W_remap, Hs=H, Wd=1280, Hd=1280, fov=FOV)

    # calculate homography
    M = Hcalc(cap, xmap, ymap)

    # calculate vertical boundary of warped image, for later cropping
    top, bottom = cropping.verticalBoundary(M, W_remap, W, H)

    # estimate empty (invalid) area of warped2
    EAof2 = np.zeros((H, W, 3), np.uint8)
    EAof2[:, int((W - W_remap) / 2) + 1:int((W + W_remap) / 2) - 1] = 255
    EAof2 = cv2.warpPerspective(EAof2, M, (W, H))

    # process the first frame
    ret, frame = cap.read()
    if ret:
        # de-warp
        cam1 = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
        cam2 = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)

        # shift the remapped images along x-axis
        shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
        shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = cam2
        shifted_cams[:H, :int(W_remap / 2)] = cam1[:, int(W_remap / 2):]
        shifted_cams[:H, int(W - W_remap / 2):] = cam1[:, :int(W_remap / 2)]

        # warp cam2 using homography M
        warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
        warped1 = shifted_cams[:H]

        # crop to get a largest rectangle, and resize to maintain resolution
        warped1 = cv2.resize(warped1[top:bottom], (W, H))
        warped2 = cv2.resize(warped2[top:bottom], (W, H))

        # image labeling (find minimum error boundary cut)
        mask, minloc_old = optimal_seamline.imgLabeling(
            warped1[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
            warped2[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
            warped1[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
            warped2[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
            (W, H), int(W_remap / 2) - W_lbl, W - int(W_remap / 2))

        labeled = warped1 * mask + warped2 * (1 - mask)

        # fill empty area of warped1 and warped2, to avoid darkening
        warped1[:, int(W_remap / 2):W - int(W_remap /
                                            2)] = warped2[:, int(W_remap / 2):W - int(W_remap / 2)]
        warped2[EAof2 == 0] = warped1[EAof2 == 0]

        # multi band blending
        blended = blending.multi_band_blending(
            warped1, warped2, mask, blend_level)

        # Show 1st frame
        '''cv2.imshow('p', blended.astype(np.uint8))
        cv2.waitKey(0)'''

        # Write results from 1st frame
        out.write(blended.astype(np.uint8))
        '''cv2.imwrite(dir_path + '/output/0.png', cam1)
        cv2.imwrite(dir_path + '/output/1.png', cam2)
        cv2.imwrite(dir_path + '/output/2.png', shifted_cams)
        cv2.imwrite(dir_path + '/output/3.png', warped2)
        cv2.imwrite(dir_path + '/output/4.png', warped1)
        cv2.imwrite(dir_path + '/output/5.png', frame)
        cv2.imwrite(dir_path + '/output/labeled.png', labeled.astype(np.uint8))
        cv2.imwrite(dir_path + '/output/blended.png', blended.astype(np.uint8))'''

    # process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # de-warp
            cam1 = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
            cam2 = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)

            # shift the remapped images along x-axis
            shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
            shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = cam2
            shifted_cams[:H, :int(W_remap / 2)] = cam1[:, int(W_remap / 2):]
            shifted_cams[:H, W - int(W_remap / 2):] = cam1[:, :int(W_remap / 2)]

            # warp cam2 using homography M
            warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
            warped1 = shifted_cams[:H]

            # crop to get a largest rectangle
            # and resize to maintain resolution
            warped1 = cv2.resize(warped1[top:bottom], (W, H))
            warped2 = cv2.resize(warped2[top:bottom], (W, H))

            # image labeling (find minimum error boundary cut)
            mask, minloc_old = optimal_seamline.imgLabeling(
                warped1[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
                warped2[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
                warped1[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
                warped2[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
                (W, H), int(W_remap / 2) - W_lbl, W - int(W_remap / 2), minloc_old)

            labeled = warped1 * mask + warped2 * (1 - mask)

            # fill empty area of warped1 and warped2, to avoid darkening
            '''warped1[:, int(W_remap / 2):W - int(W_remap /
                                                2)] = warped2[:, int(W_remap / 2):W - int(W_remap / 2)]
            warped2[EAof2 == 0] = warped1[EAof2 == 0]

            # multi band blending
            blended = blending.multi_band_blending(
                warped1, warped2, mask, blend_level)'''

            # write the remapped frame
            out.write(labeled.astype(np.uint8))

            # Timer to count require time of each frame
            now = time.time()
            print(now - start, 'second')
            average_time.append(now - start)
            modify_start(now)

            # Show each frame
            '''cv2.imshow('warped', labeled.astype(np.uint8))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break'''
        else:
            break

    # release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main_video_with_lookup_table(input, output):
    with open('parameters/parameters.npy', 'rb') as f:
        # lookup table parameters
        xmap = np.load(f)
        ymap = np.load(f)
        M = np.load(f)
        EAof2 = np.load(f)
        mask = np.load(f)

        with open('parameters/parameters.pkl', 'rb') as inp:
            # lookup table parameters
            top = pickle.load(inp)
            bottom = pickle.load(inp)

            cap = cv2.VideoCapture(input)

            # define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, fourcc, 30.0, (W, H))

            # process the first frame
            ret, frame = cap.read()
            if ret:
                # de-warp
                cam1 = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
                cam2 = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)

                # shift the remapped images along x-axis
                shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
                shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = cam2
                shifted_cams[:H, :int(W_remap / 2)] = cam1[:, int(W_remap / 2):]
                shifted_cams[:H, int(W - W_remap / 2):] = cam1[:, :int(W_remap / 2)]

                # warp cam2 using homography M
                warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
                warped1 = shifted_cams[:H]

                # crop to get a largest rectangle, and resize to maintain resolution
                warped1 = cv2.resize(warped1[top:bottom], (W, H))
                warped2 = cv2.resize(warped2[top:bottom], (W, H))

                # image labeling (find minimum error boundary cut)
                labeled = warped1 * mask + warped2 * (1 - mask)

                # fill empty area of warped1 and warped2, to avoid darkening
                '''warped1[:, int(W_remap / 2):W - int(W_remap /
                        2)] = warped2[:, int(W_remap / 2):W - int(W_remap / 2)]
                warped2[EAof2 == 0] = warped1[EAof2 == 0]'''

                # multi band blending
                '''blended = blending.multi_band_blending(
                    warped1, warped2, mask, blend_level)'''

                # Show 1st frame
                '''cv2.imshow('p', blended.astype(np.uint8))
                cv2.waitKey(0)'''

                # Write results from 1st frame
                out.write(labeled.astype(np.uint8))
                '''cv2.imwrite(dir_path + '/output/0.png', cam1)
                cv2.imwrite(dir_path + '/output/1.png', cam2)
                cv2.imwrite(dir_path + '/output/2.png', shifted_cams)
                cv2.imwrite(dir_path + '/output/3.png', warped2)
                cv2.imwrite(dir_path + '/output/4.png', warped1)
                cv2.imwrite(dir_path + '/output/5.png', frame)
                cv2.imwrite(dir_path + '/output/labeled.png', labeled.astype(np.uint8))
                cv2.imwrite(dir_path + '/output/blended.png', blended.astype(np.uint8))'''

            # process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # de-warp
                    cam1 = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
                    cam2 = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)

                    # shift the remapped images along x-axis
                    shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
                    shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = cam2
                    shifted_cams[:H, :int(W_remap / 2)] = cam1[:, int(W_remap / 2):]
                    shifted_cams[:H, W - int(W_remap / 2):] = cam1[:, :int(W_remap / 2)]

                    # warp cam2 using homography M
                    warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
                    warped1 = shifted_cams[:H]

                    # crop to get a largest rectangle
                    # and resize to maintain resolution
                    warped1 = cv2.resize(warped1[top:bottom], (W, H))
                    warped2 = cv2.resize(warped2[top:bottom], (W, H))

                    # image labeling (find minimum error boundary cut)
                    labeled = warped1 * mask + warped2 * (1 - mask)

                    # fill empty area of warped1 and warped2, to avoid darkening
                    '''warped1[:, int(W_remap / 2):W - int(W_remap /
                                                        2)] = warped2[:, int(W_remap / 2):W - int(W_remap / 2)]
                    warped2[EAof2 == 0] = warped1[EAof2 == 0]'''

                    # multi band blending
                    '''blended = blending.multi_band_blending(
                        warped1, warped2, mask, blend_level)'''

                    # write the remapped frame
                    out.write(labeled.astype(np.uint8))

                    # Timer to count require time of each frame
                    now = time.time()
                    print(now - start, 'second')
                    average_time.append(now - start)
                    modify_start(now)

                    # Show each frame
                    '''cv2.imshow('warped', labeled.astype(np.uint8))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break'''
                else:
                    break

            # release everything if job is finished
            cap.release()
            out.release()
            cv2.destroyAllWindows()


def main_video_with_optimized_lookup_table(input, output):
    with open('parameters/parameters.npy', 'rb') as f:
        # lookup table parameters
        _ = np.load(f)
        _ = np.load(f)
        M = np.load(f)
        _ = np.load(f)
        mask = np.load(f)

        with open('parameters/shifted_cams_map.npy', 'rb') as f:
            # lookup table parameters
            x_map_shifthed_cams = np.load(f)
            y_map_shifthed_cams = np.load(f)

            with open('parameters/parameters.pkl', 'rb') as inp:
                # lookup table parameters
                top = pickle.load(inp)
                bottom = pickle.load(inp)

                cap = cv2.VideoCapture(input)

                # define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output, fourcc, 30.0, (W, H))

                # process the first frame
                ret, frame = cap.read()
                if ret:
                    # shift the remapped images along x-axis
                    shifted_cams = cv2.remap(frame, x_map_shifthed_cams, y_map_shifthed_cams, cv2.INTER_LINEAR)

                    # warp cam2 using homography M
                    warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
                    warped1 = shifted_cams[:H]

                    # crop to get a largest rectangle, and resize to maintain resolution
                    warped1 = cv2.resize(warped1[top:bottom], (W, H))
                    warped2 = cv2.resize(warped2[top:bottom], (W, H))

                    # image labeling (find minimum error boundary cut)
                    labeled = warped1 * mask + warped2 * (1 - mask)

                    # fill empty area of warped1 and warped2, to avoid darkening
                    '''warped1[:, int(W_remap / 2):W - int(W_remap /
                            2)] = warped2[:, int(W_remap / 2):W - int(W_remap / 2)]
                    warped2[EAof2 == 0] = warped1[EAof2 == 0]'''

                    # multi band blending
                    '''blended = blending.multi_band_blending(
                        warped1, warped2, mask, blend_level)'''

                    # Show 1st frame
                    '''cv2.imshow('p', blended.astype(np.uint8))
                    cv2.waitKey(0)'''

                    # Write results from 1st frame
                    out.write(labeled.astype(np.uint8))
                    '''cv2.imwrite(dir_path + '/output/0.png', cam1)
                    cv2.imwrite(dir_path + '/output/1.png', cam2)
                    cv2.imwrite(dir_path + '/output/2.png', shifted_cams)
                    cv2.imwrite(dir_path + '/output/3.png', warped2)
                    cv2.imwrite(dir_path + '/output/4.png', warped1)
                    cv2.imwrite(dir_path + '/output/5.png', frame)
                    cv2.imwrite(dir_path + '/output/labeled.png', labeled.astype(np.uint8))
                    cv2.imwrite(dir_path + '/output/blended.png', blended.astype(np.uint8))'''

                # process each frame
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # shift the remapped images along x-axis
                        shifted_cams = cv2.remap(frame, x_map_shifthed_cams, y_map_shifthed_cams, cv2.INTER_LINEAR)

                        # warp cam2 using homography M
                        warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
                        warped1 = shifted_cams[:H]

                        # crop to get a largest rectangle
                        # and resize to maintain resolution
                        warped1 = cv2.resize(warped1[top:bottom], (W, H))
                        warped2 = cv2.resize(warped2[top:bottom], (W, H))

                        # image labeling (find minimum error boundary cut)
                        labeled = warped1 * mask + warped2 * (1 - mask)

                        # fill empty area of warped1 and warped2, to avoid darkening
                        '''warped1[:, int(W_remap / 2):W - int(W_remap /
                                                            2)] = warped2[:, int(W_remap / 2):W - int(W_remap / 2)]
                        warped2[EAof2 == 0] = warped1[EAof2 == 0]'''

                        # multi band blending
                        '''blended = blending.multi_band_blending(
                            warped1, warped2, mask, blend_level)'''

                        # write the remapped frame
                        out.write(labeled.astype(np.uint8))

                        # Timer to count require time of each frame
                        now = time.time()
                        print(now - start, 'second')
                        average_time.append(now - start)
                        modify_start(now)

                        # Show each frame
                        '''cv2.imshow('warped', labeled.astype(np.uint8))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break'''
                    else:
                        break

                # release everything if job is finished
                cap.release()
                out.release()
                cv2.destroyAllWindows()


def main_write_paramters(input):
    with open('parameters/parameters.npy', 'wb') as f:
        cap = cv2.VideoCapture(input)

        # obtain xmap and ymap
        xmap, ymap = dewarp.buildmap(Ws=W_remap, Hs=H, Wd=1280, Hd=1280, fov=FOV)
        np.save(f, xmap)
        np.save(f, ymap)

        # calculate homography
        M = Hcalc(cap, xmap, ymap)
        np.save(f, M)

        # calculate vertical boundary of warped image, for later cropping
        top, bottom = cropping.verticalBoundary(M, W_remap, W, H)
        with open('parameters/parameters.pkl', 'wb') as outp:
            pickle.dump(top, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(bottom, outp, pickle.HIGHEST_PROTOCOL)

        # estimate empty (invalid) area of warped2
        EAof2 = np.zeros((H, W, 3), np.uint8)
        EAof2[:, int((W - W_remap) / 2) + 1:int((W + W_remap) / 2) - 1] = 255
        EAof2 = cv2.warpPerspective(EAof2, M, (W, H))
        np.save(f, EAof2)

        # process the first frame
        ret, frame = cap.read()
        if ret:
            # de-warp
            cam1 = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
            cam2 = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)

            # shift the remapped images along x-axis
            shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
            shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = cam2
            shifted_cams[:H, :int(W_remap / 2)] = cam1[:, int(W_remap / 2):]
            shifted_cams[:H, int(W - W_remap / 2):] = cam1[:, :int(W_remap / 2)]

            # warp cam2 using homography M
            warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
            warped1 = shifted_cams[:H]

            # crop to get a largest rectangle, and resize to maintain resolution
            warped1 = cv2.resize(warped1[top:bottom], (W, H))
            warped2 = cv2.resize(warped2[top:bottom], (W, H))

            # image labeling (find minimum error boundary cut)
            mask, minloc_old = optimal_seamline.imgLabeling(
                warped1[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
                warped2[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
                warped1[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
                warped2[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
                (W, H), int(W_remap / 2) - W_lbl, W - int(W_remap / 2))
            np.save(f, mask)

            labeled = warped1 * mask + warped2 * (1 - mask)

            # fill empty area of warped1 and warped2, to avoid darkening
            warped1[:, int(W_remap / 2):W - int(W_remap /
                                                2)] = warped2[:, int(W_remap / 2):W - int(W_remap / 2)]
            warped2[EAof2 == 0] = warped1[EAof2 == 0]

            # multi band blending
            blended = blending.multi_band_blending(
                warped1, warped2, mask, blend_level)


def create_shift_map():
    # Coordinate Maps
    xmap, ymap = dewarp.buildmap(Ws=W_remap, Hs=H, Wd=1280, Hd=1280, fov=FOV)

    x_map_shifted_cams = np.zeros((H * 2, W), np.float32)
    x_map_shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = xmap
    x_map_shifted_cams[:H, :int(W_remap / 2)] = xmap[:, int(W_remap / 2):]
    x_map_shifted_cams[:H, int(W - W_remap / 2):] = xmap[:, :int(W_remap / 2)]

    y_map_shifted_cams = np.zeros((H * 2, W), np.float32)
    y_map_shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = ymap
    y_map_shifted_cams[:H, :int(W_remap / 2)] = ymap[:, int(W_remap / 2):]
    y_map_shifted_cams[:H, int(W - W_remap / 2):] = ymap[:, :int(W_remap / 2)]

    x_map_shifted1 = x_map_shifted_cams[:H]
    y_map_shifted1 = y_map_shifted_cams[:H]

    x_map_shifted2 = x_map_shifted_cams[H:]
    y_map_shifted2 = y_map_shifted_cams[H:]

    return x_map_shifted1, y_map_shifted1, x_map_shifted2, y_map_shifted2


# Return x-coordinate map & y-coordinate map for CV.WARP_PERSPECTIVE
def create_warp_perspective_map(homography):
    transformationMatrix = np.copy(homography)

    _, inverseTransMatrix = cv2.invert(transformationMatrix)

    srcTM = np.copy(inverseTransMatrix)

    map_x = np.zeros((1280, 2560), dtype=np.float32)
    map_y = np.zeros((1280, 2560), dtype=np.float32)

    M11 = srcTM[0][0]
    M12 = srcTM[0][1]
    M13 = srcTM[0][2]
    M21 = srcTM[1][0]
    M22 = srcTM[1][1]
    M23 = srcTM[1][2]
    M31 = srcTM[2][0]
    M32 = srcTM[2][1]
    M33 = srcTM[2][2]

    for y in range(len(map_x)):
        fy = y
        for x in range(len(map_x[0])):
            fx = x
            w = ((M31 * fx) + (M32 * fy) + M33)
            # w = w != 0.0f ? 1.f / w : 0.0f;
            new_x = ((M11 * fx) + (M12 * fy) + M13) * w
            new_y = ((M21 * fx) + (M22 * fy) + M23) * w
            map_x[y][x] = new_x
            map_y[y][x] = new_y

    # Optimization
    '''
    transformation_x, transformation_y = cv2.convertMaps(map_x, map_y, cv2.CV_8UC1)

    with open('parameters/m.npy', 'wb') as f:
        np.save(f, transformation_x)
        np.save(f, transformation_y)

    print(transformation_x)
    print(transformation_y)'''

    return map_x, map_y


# Merge the map of shifted & warp perspective of frame 2
def create_merged_map(x_map_shift_camera, y_map_shift_camera, x_map_warp_perspective, y_map_warp_perspective):
    x_map_merged = np.zeros(x_map_warp_perspective.shape, dtype=np.float32)
    y_map_merged = np.zeros(y_map_warp_perspective.shape, dtype=np.float32)

    for i in range(len(x_map_warp_perspective)):
        for j in range(len(x_map_warp_perspective[0])):
            if 0 <= y_map_warp_perspective[i][j] < H - 1 and 0 <= x_map_warp_perspective[i][j] < W - 1:
                # For merging x_map_shifted2 with x_map_warp_perspective
                x_map_intercept1 = linear_interpolation(
                    x=y_map_warp_perspective[i][j],
                    x1=math.floor(y_map_warp_perspective[i][j]),
                    x2=math.floor(y_map_warp_perspective[i][j]) + 1,
                    v1=x_map_shift_camera[math.floor(y_map_warp_perspective[i][j])][math.floor(x_map_warp_perspective[i][j])],
                    v2=x_map_shift_camera[math.floor(y_map_warp_perspective[i][j]) + 1][math.floor(x_map_warp_perspective[i][j])]
                )

                x_map_intercept2 = linear_interpolation(
                    x=y_map_warp_perspective[i][j],
                    x1=math.floor(y_map_warp_perspective[i][j]),
                    x2=math.floor(y_map_warp_perspective[i][j]) + 1,
                    v1=x_map_shift_camera[math.floor(y_map_warp_perspective[i][j])][math.floor(x_map_warp_perspective[i][j]) + 1],
                    v2=x_map_shift_camera[math.floor(y_map_warp_perspective[i][j]) + 1][math.floor(x_map_warp_perspective[i][j]) + 1]
                )

                x_map_interpolation = linear_interpolation(
                    x=x_map_warp_perspective[i][j],
                    x1=math.floor(x_map_warp_perspective[i][j]),
                    x2=math.floor(x_map_warp_perspective[i][j]) + 1,
                    v1=x_map_intercept1,
                    v2=x_map_intercept2
                )

                x_map_merged[i][j] = x_map_intercept2

                # For merging y_map_shifted2 with y_map_warp_perspective
                y_map_intercept1 = linear_interpolation(
                    x=y_map_warp_perspective[i][j],
                    x1=math.floor(y_map_warp_perspective[i][j]),
                    x2=math.floor(y_map_warp_perspective[i][j]) + 1,
                    v1=y_map_shift_camera[math.floor(y_map_warp_perspective[i][j])][math.floor(x_map_warp_perspective[i][j])],
                    v2=y_map_shift_camera[math.floor(y_map_warp_perspective[i][j]) + 1][math.floor(x_map_warp_perspective[i][j])]
                )

                y_map_intercept2 = linear_interpolation(
                    x=y_map_warp_perspective[i][j],
                    x1=math.floor(y_map_warp_perspective[i][j]),
                    x2=math.floor(y_map_warp_perspective[i][j]) + 1,
                    v1=y_map_shift_camera[math.floor(y_map_warp_perspective[i][j])][math.floor(x_map_warp_perspective[i][j]) + 1],
                    v2=y_map_shift_camera[math.floor(y_map_warp_perspective[i][j]) + 1][math.floor(x_map_warp_perspective[i][j]) + 1]
                )

                y_map_interpolation = linear_interpolation(
                    x=x_map_warp_perspective[i][j],
                    x1=math.floor(x_map_warp_perspective[i][j]),
                    x2=math.floor(x_map_warp_perspective[i][j]) + 1,
                    v1=y_map_intercept1,
                    v2=y_map_intercept2
                )

                y_map_merged[i][j] = y_map_interpolation

    return x_map_merged, y_map_merged


def linear_interpolation(x, x1, x2, v1, v2):
    return v1 + (((x - x1) * (v2 - v1)) / (x2 - x1))


'''if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(
        description="A summer research project to seamlessly stitch \
                     dual-fisheye video into 360-degree videos")
    ap.add_argument('input', metavar='INPUT.XYZ',
                    help="path to the input dual fisheye video")
    ap.add_argument('-o', '--output', metavar='OUTPUT.XYZ', required=False,
                    default=dir_path + '/output/output.MP4',
                    help="path to the output equirectangular video")

    args = vars(ap.parse_args())
    main(args['input'], args['output'])'''


# Prepare input images:
# 1. Must be square
# 2. Height & Width must be multiple of 2^7 (for image pyramid)

# Resize the image to 2560. Fill in the background to black. Make it a square image
# --------------------------------
# output video resolution
# W = 5120
# H = 2560
# --------------------------------
# field of view, width of de-warped image
# FOV = 202.0
# W_remap = 2760  # input width + overlap (200)
# --------------------------------'''

# Create & save shift maps
'''x_map_shifted1, y_map_shifted1, x_map_shifted2, y_map_shifted2 = create_shift_map()

with open('parameters/shift_parameters.npy', 'wb') as f1:
    np.save(f1, x_map_shifted1)
    np.save(f1, y_map_shifted1)
    np.save(f1, x_map_shifted2)
    np.save(f1, y_map_shifted2)'''

# Create & save merged maps
'''with open('parameters/raw_parameters.npy', 'rb') as f2:
    # Create & save warp perspective maps
    _ = np.load(f2)
    _ = np.load(f2)
    M = np.load(f2)

    x_map_warp_perspective, y_map_warp_perspective = create_warp_perspective_map(M)

    with open('parameters/warp_perspective_parameters.npy', 'wb') as f3:
        np.save(f3, x_map_warp_perspective)
        np.save(f3, y_map_warp_perspective)

    x_map_merged, y_map_merged = create_merged_map(x_map_shifted2 + 1280, y_map_shifted2, x_map_warp_perspective, y_map_warp_perspective)

    with open('parameters/merged_parameters.npy', 'wb') as f4:
        np.save(f4, x_map_merged)
        np.save(f4, y_map_merged)'''

frame = cv2.imread('input/cap.png')

with open('parameters/shift_parameters.npy', 'rb') as f1:
    x_map_shifted1 = np.load(f1)
    y_map_shifted1 = np.load(f1)
    x_map_shifted2 = np.load(f1)
    y_map_shifted2 = np.load(f1)

with open('parameters/merged_parameters.npy', 'rb') as f2:
    x_map_merged = np.load(f2)
    y_map_merged = np.load(f2)

shifted1 = cv2.remap(frame, x_map_shifted1, y_map_shifted1, cv2.INTER_LINEAR)
cv2.namedWindow('Shifted 1', cv2.WINDOW_NORMAL)
cv2.imshow('Shifted 1', shifted1)
cv2.waitKey(0)
cv2.destroyAllWindows()

shifted2 = cv2.remap(frame, x_map_shifted2 + 1280, y_map_shifted2, cv2.INTER_LINEAR)
cv2.namedWindow('Shifted 2', cv2.WINDOW_NORMAL)
cv2.imshow('Shifted 2', shifted2)
cv2.waitKey(0)
cv2.destroyAllWindows()

merged2 = cv2.remap(frame, x_map_merged, y_map_merged, cv2.INTER_LINEAR)
cv2.namedWindow('Merged 2', cv2.WINDOW_NORMAL)
cv2.imshow('Merged 2', merged2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output/shifted-1.png', shifted1)
cv2.imwrite('output/warped-1.png', shifted2)
cv2.imwrite('output/merged-2.png', merged2)





