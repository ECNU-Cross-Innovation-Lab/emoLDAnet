import os
import pandas as pd
from FER.recognizer import Recognizer
from skimage import io
import csv
from utils import toOCEAN,toPAD,toDTASL
from tqdm import tqdm
import cv2
import numpy as np
from mtcnn import FaceDetector
from mtcnn.utils import convert_to_square
from PIL import Image
import time
import matplotlib.pyplot as plt

labels = ['Subject','Frame','Angry', 'Disgust', 'Fear', 
    'Happy', 'Sad', 'Surprise', 'Neutral','Pleasure', 'Arousal', 'Dominance',
    '-', '-', '-', '-', '-']

def wash_names(video_names):
    temp = []
    for name in video_names:
        if name.endswith("mp4") or name.endswith("mkv") or name.endswith("mov") or name.endswith("webm"):
            temp.append(name)
    return temp


def video_to_image_time(video_path, t_freq):
    '''

    Args:
        video_path: The path of the directory of the videos
        t_freq: For example: t_freq = 1 means the frames are taken to recognize every 1 second.

    Returns:
        all the frames taken
    '''
    camera = cv2.VideoCapture(video_path)
    count = 0
    selected_frms = []
    while 1:
        camera.set(cv2.CAP_PROP_POS_MSEC, t_freq * 1000 * count)
        res, img = camera.read()
        print('\r' + str(count * t_freq) + str(res), end='', flush=True)
        if not res:
            break
        img = Image.fromarray(img)
        selected_frms.append(img)
        count += 1
    print("\n{0} frames are selected.".format(len(selected_frms)))
    return selected_frms


def process_video(video_path, result_csv):
    '''

    Args:
        video_path: The path of the directory of the videos
        result_path: The path of the result

    Returns:
        None
        Generates full_result.csv under result_path. This file contains the recognizing results allover the videos
    '''
    recognizer = Recognizer()
    detector = FaceDetector()
    video_names = os.listdir(video_path)
    video_names = wash_names(video_names)
    print('Constructing dataset...')
    with open(result_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        names = ['Name', 'frame', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Pleasure',
                 'Arousal', 'Dominance', '-', '-', '-', '-',
                 '-', 'Depression', 'Anxiety', '-', '-', 'Loneliness', '-']
        writer.writerow(names)
        for video_name in tqdm(video_names):
            start = time.time()
            end = video_name.split('.')[-1]
            video_name = video_name[:-(1 + len(end))]
            print('Processing ' + video_name + '...')

            selected_frms = video_to_image_time(video_path + video_name + "." + end, 0.2)
            print('Getting faces...')
            faces = detector.new_detect(selected_frms)
            print('Get {} faces...'.format(len(faces)))
            print('Recognizing expressions...')
            models = ['resnet18_1', 'mobilenet_1', 'squeezenet_1', 'densenet_1', 'VGG11_1', 'VGG13_1', 'VGG16_1', 'VGG19_1', 
            'resnet18_2', 'mobilenet_2', 'squeezenet_2', 'densenet_2', 'VGG11_2', 'VGG13_2', 'VGG16_2', 'VGG19_2', ]
            row = []
            for i in range(len(faces)):
                row.append([video_name, i])

            for model in models:
                occs = recognizer.new_recognize(faces, model)

                i = 0
                for occ in occs:
                    pad = toPAD(occ)
                    ocean = toOCEAN(pad)
                    dtasl = toDTASL(pad)
                    row[i].extend(occ)
                    row[i].extend(pad)
                    row[i].extend(ocean)
                    row[i].extend(dtasl)
                    i += 1
                print(model, end=' ')
            print('\n')
            for r in row:
                writer.writerow(r)
            end = time.time()
            print("time:" + str(end - start))
