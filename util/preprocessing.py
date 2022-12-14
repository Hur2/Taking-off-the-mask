import os
import numpy as np
import scipy.ndimage
import PIL.Image

import mediapipe as mp
import cv2

def image_align(src_file, dst_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=True):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm = np.array(face_landmarks)
        lm_chin          = lm[0  : 17, :2]  # left-right
        lm_eyebrow_left  = lm[17 : 22, :2]  # left-right
        lm_eyebrow_right = lm[22 : 27, :2]  # left-right
        lm_nose          = lm[27 : 31, :2]  # top-down
        lm_nostrils      = lm[31 : 36, :2]  # top-down
        lm_eye_left      = lm[36 : 42, :2]  # left-clockwise
        lm_eye_right     = lm[42 : 48, :2]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60, :2]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68, :2]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
        
        # convert to 'png'
        name, ext = os.path.splitext(dst_file)
        ext_len = -len(ext)
        dst_file = dst_file[:ext_len] + '.png'

        # Save aligned image.
        img.save(dst_file, 'PNG')


def folders_image_align(src, dst):
    
    output_size = 512
    transform_size = 512
    no_padding = False

    if not os.path.exists(dst):
        os.mkdir(dst)

    for img_name in os.listdir(src):
        raw_img_path = os.path.join(src, img_name)
        
        face_landmarks = mask_landmark_extract(raw_img_path, True)
        aligned_face_path = os.path.join(dst, f'align-{img_name}')
        
        image_align(raw_img_path, aligned_face_path, face_landmarks, output_size, transform_size, no_padding)
            
def mask_landmark_extract(src, align=False):
    
    mask_location = []
    dlib_68_location = []

    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
  
        image = cv2.imread(src)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks : 
            print("ERROR")
            return False

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        
        mask_landmark = [
            127, 234, 93, 132, 58, 172, 136,150, 176, 148, 152, 377, 
            400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 168
        ]
        landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]


        loc_x, loc_y = "", ""
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks: 
                if align:
                    for idx in landmark_points_68:
                        loc_x = face_landmarks.landmark[idx].x * image.shape[1]
                        loc_y = face_landmarks.landmark[idx].y * image.shape[0]
                        dlib_68_location.append((loc_x, loc_y))
                else:
                    for idx in mask_landmark:
                        loc_x = face_landmarks.landmark[idx].x * image.shape[1]
                        loc_y = face_landmarks.landmark[idx].y * image.shape[0]
                        mask_location.append((loc_x, loc_y+25))
        if align:
            return dlib_68_location
        return mask_location

def mask_image_generate(src, dst):

    for img_name in os.listdir(src):
        
        if '.png' not in img_name:
            continue

        src_img_path = os.path.join(src, img_name)
        dst_img_path = os.path.join(dst, img_name)
    
        landmark_points = mask_landmark_extract(src_img_path)
        points = np.array(landmark_points, np.int32)

        mask = np.zeros((512,512,3), dtype=np.int32)
        black_color = (255,255,255)
        img = cv2.fillConvexPoly(mask, points, black_color)

        cv2.imwrite(dst_img_path,img)