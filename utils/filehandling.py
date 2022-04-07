import os
import cv2 as cv
import torch
import numpy

def LoadFilesWithExtensions(path, extensions):
    ''':arg path->str : path of directory,
        :arg extensions->list(str) of extensions
        :returns->List(str) of files with the given extensions'''

    ret = []
    for root, dirs, files in os.walk(path):
        for file in files:
            for ext in extensions:
                if(file.endswith(ext)):
                    ret.append(os.path.join(root, file))
                    break
    return ret


def draw_rectangles(img, preds, thres=0.5):
    H, W, C = img.shape
    seeds = 0
    img_n = img

    if preds != None:
        no_tensors = preds["boxes"].size()
        for i in range(no_tensors[0]):
            if(preds["scores"][i] > thres):
                seeds += 1
                color = tuple(numpy.random.randint(0, 255, size=3).astype(numpy.uint8))
                color = (0, 0, 255)
                color_text = (255 - color[0], 255 - color[1], 255 - color[2])
                box_np = preds["boxes"][i].numpy().astype(int)
                x_min, y_min, x_max, y_max = list(box_np)
                img_n = cv.rectangle(img_n - numpy.zeros(img_n.shape), (x_min, y_min), (x_max, y_max), color, 2)
                label = preds["labels"][i].numpy().astype(int)
                img_n = cv.putText(img_n,
                                   "label : " + str(label),
                                   (x_min, y_min),
                                   cv.FONT_HERSHEY_DUPLEX,
                                   0.35,
                                   color_text,
                                   1)
                if "scores" in preds:
                    img_n = cv.putText(img_n,
                                       "score : " + str(preds["scores"][i]),
                                       (x_min, y_max),
                                       cv.FONT_HERSHEY_DUPLEX,
                                       0.35,
                                       color_text,
                               1)

    return img_n.astype(numpy.uint8)


class FrameSelectorBase(object):


    def __int__(self):
        self.start_frame = 0
        self.end_frame = 0
        self.list_of_videos = []
        print("Hi Parent")

    def preprocess(self, img):
        return img

    def setFrameOneAndEndFrame(self, ListOfVideos, FramesStart, FramesEnd):
        self.list_of_videos = ListOfVideos
        self.start_frame = FramesStart
        self.end_frame = FramesEnd
        self.seeked = False
        self.framesOverHead = 0
        self.pos = 0
        self.video_pos = 0

    def __len__(self):
        return self.end_frame - self.start_frame


    def getFrames(self):
        self.frames = 0

        while (not self.seeked):
            self.cap = cv.VideoCapture(self.list_of_videos[self.video_pos])
            self.frames = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
            if (self.start_frame > self.framesOverHead + self.frames):
                self.framesOverHead += self.frames
                self.cap.release()
                self.video_pos += 1
            else:
                self.cap.set(cv.CAP_PROP_POS_FRAMES, self.start_frame - self.framesOverHead)
                self.seeked = True
        i = self.cap.get(cv.CAP_PROP_POS_FRAMES)

        while self.pos <= self.end_frame - self.start_frame:
            i += 1
            if(i>self.frames):
                self.video_pos += 1
                self.cap.release()
                self.cap = cv.VideoCapture(self.list_of_videos[self.video_pos])
                self.frames = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
            self.pos += 1
            _, img = self.cap.read()
            yield self.preprocess(img)
        return


    def GetNumberedFramesFromListOfVideos(self, ListOfVideos, FramesStart, FramesEnd):
            #End Frame Is inclussive

            framesOverHead = 0
            seeked = False
            pos = 0
            ret = []

            for video in ListOfVideos:

                cap = cv.VideoCapture(video)
                frames = cap.get(cv.CAP_PROP_FRAME_COUNT)

                if(not seeked):
                    if(FramesStart>framesOverHead+frames):
                        framesOverHead += frames
                        continue
                    else:
                        cap.set(cv.CAP_PROP_POS_FRAMES, FramesStart-framesOverHead)
                        seeked = True

                if(seeked):
                    i = cap.get(cv.CAP_PROP_POS_FRAMES)
                    while (i < frames) and pos <= FramesEnd-FramesStart:
                        i += 1
                        pos += 1
                        _, img = cap.read()
                        img = self.preprocess(img)
                        ret.append(img)
                cap.release()
                if(pos == FramesEnd - FramesStart):
                    break

            return ret

    def setListOfVideos(self, list_of_videos):
        self.list_of_videos = list_of_videos

if __name__ == "__main__":
    path = "/home/harsha/Desktop/seedIdentification/SeedSpace2_ImagesSplitted/"
    ret = LoadFilesWithExtensions(path, ["avi", "tdms"])
    print(ret)