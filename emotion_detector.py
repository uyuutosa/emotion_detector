import cv2
import time
import numpy as np
from numpy import *
from PIL import Image
from tqdm import tqdm
import Optimizer_with_theano as op
import Optimizer_with_theano.Datasets as data
from matplotlib.pyplot import *
import pickle
import os
import glob as g


# Emotion Detector
class emotion_detector():
    def __init__(self, num_datasets=1000, cnum = 0, image_size=(20,20)):
        self.num_datasets = num_datasets
        self.cnum = cnum
        self.image_size = image_size

    def get_train_data(self, parent_dir, is_mouth=False):
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        self.dir_path = parent_dir
        input("Ready to laugh...")
        self.take_photos("laugh", is_mouth)
        input("Ready to anger...")
        self.take_photos("anger", is_mouth)
        input("Ready to sad...")
        self.take_photos("sad", is_mouth)
        input("Ready to fun...")
        self.take_photos("fun", is_mouth)

    def set_data(self, dir_path="default"):
        xlst = []
        ylst = []
        for i, emo in enumerate(["laugh", "anger", "sad", "fun"]):
            for p in g.glob(dir_path+"/{}/*".format(emo)):
                xlst += [cv2.resize(cv2.imread(p,0), (200,200))]
                ylst += [i]

        self.x_arr = array(xlst, dtype=float).reshape(len(xlst), -1)
        self.x_arr /= 255
        self.y_arr = data.gen_one_hot(array(ylst)[:,None])

    def train(self, lr=0.001, n_batch=10, n_epoch=100, n_view=10):
        #self.y_arr = data.gen_one_hot(arange(4).repeat(self.num_datasets)[:, None])
        self.o = op.optimizer(self.x_arr, self.y_arr, n_batch=n_batch)
        self.o = self.o.dense(10,act="relu")
        self.o = self.o.dense(100,act="relu")
        self.o = self.o.dense(200,act="relu")
        self.o = self.o.dense(300,act="relu")
        self.o = self.o.dense(400,act="relu")
        self.o = self.o.dense(4).softmax()
        self.o = self.o.loss_cross_entropy()
        self.o = self.o.opt_Adam(lr).compile()
        self.o = self.o.optimize(n_epoch, n_view)

    def cam_test(self):
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(self.cnum)
        tof, frame = vc.read()
        return tof

    def take_photos(self, file_dir, is_mouth=True, fname="file"):
        dirpath = self.dir_path + "/{}".format(file_dir)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(self.cnum)
        haarFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        haarMouth = cv2.CascadeClassifier('haarcascade_mouth.xml')

        lst = []
        rval, frame = vc.read()
        #tof, self.frame = vc.read()

        print("Please patient for few minutes..")
        with tqdm(total=self.num_datasets) as pbar:
            cnt = 0
            for i in range(self.num_datasets):
                rval, frame = vc.read()
                cv2.imshow("preview", frame)

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # running the classifiers
                detectedFace = haarFace.detectMultiScale(frame_gray)
                if not len(detectedFace):
                    continue

                x, y, width, height = detectedFace[array([x[2]*x[3] for x in detectedFace]).argmax()]

                face = frame_gray[y:y+height, x:x+width]
                cv2.imshow("face", face)
                cv2.imwrite(dirpath + "/{}_{:04d}.jpg".format(fname, cnt), face)
                #cv2.waitKey(100)

                pbar.update(1)
                cv2.waitKey(1)
                cnt += 1


    def dump_pickle(self, fname="emotion.pkl"):
        with open(fname, mode='wb') as f:
            pickle.dump(self, f)
            print("Pickle data is dumped as {}.".format(fname))


    def load_pickle(self, fname="emotion.pkl"):
        with open(fname, mode='rb') as f:
            return pickle.load(f)

    def detect(self, is_mouth=True):
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(self.cnum)
        haarFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        haarMouth = cv2.CascadeClassifier('haarcascade_mouth.xml')
        emolst = ["laugh", "anger", "sad", "fun"]
        lst = []
        rval, frame = vc.read()


        # 動画の読み込みと動画情報の取得
        fps    = vc.get(cv2.CAP_PROP_FPS)
        height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width  = vc.get(cv2.CAP_PROP_FRAME_WIDTH)

        # 出力先のファイルを開く
        out = cv2.VideoWriter("output.mv4", int(cv2.VideoWriter_fourcc('m', 'p', '4', 'v')), fps, (int(width), int(height)))

        while True:
            cv2.imshow("preview", frame)
            out.write(frame)
            num = cv2.waitKey(1)
            if num == "q":
                break
            rval, frame = vc.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detectedFace = haarFace.detectMultiScale(frame_gray)
            if not len(detectedFace):
                continue

            x, y, width, height = detectedFace[array([x[2]*x[3] for x in detectedFace]).argmax()]

            face = frame_gray[y:y+height, x:x+width]

            if not is_mouth:
                lst += [cv2.resize(face,self.image_size)]
            elif is_mouth:
                rects = haarMouth.detectMultiScale(face, 1.3, minSize=(50,50))
                self.rects = rects
                if len(rects):
                    rect = rects[array([x[1] for x in rects]).argmax()]
                    mouth = face[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[1]]
                    #cv2.imshow("mouth", mouth)
                    ans = emolst[int(self.o.pred_func(cv2.resize(mouth,self.image_size).reshape(1,-1))[0])]
                    cv2.putText(frame, ans, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)#, cv2.CV_AA)
            #cv2.imshow("face", face)


        cv2.destroyWindow("preview")
        out.release()
        return lst
