import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import torch


class SeedPredictor:

    def __init__(self, name):
        self.name = name

    def predict(self, input):
        #TODO impliment the abstarct method
        print("predict method not implimented")

class RetinaNetSeedPredictor(SeedPredictor):

    def __init__(self, name, retinanet_path, device=0):

        super().__init__(name)
        self.retina_net = torch.load(retinanet_path)
        self.device = device

    def predict(self, input):
                
        tensors = list(map(lambda cv2_frame: torch.Tensor(cv2_frame[::-1] / 255).to(self.device).permute(2, 0, 1), input))
        self.retina_net.eval()
        
        with torch.no_grad():
            outs = self.retina_net(tensors)
        
        return outs


class SeedAttributes:
    
    def __init__(self, seed_loc, distance, velocity, acceleration, gps, frame_no):
        
        self.seed_loc     = seed_loc
        self.velocity     = velocity
        self.acceleration = acceleration
        self.gps          = gps
        self.distance     = distance
        self.frame_no     = frame_no



class distance_struct:

    def __init__(self, position=0):

        self.frame_no = position
        self.seed_attributes = []
        self.cum_distance = 0
        self.first_seed = False
        self.found_seed = False
        self.frames_seeds_dict = {}

    def filter_out_seen_seeds(self, pos_array, p, frame_length):
        '''Takes in seed positions array, distance seen in the previous image if it has a seed in it and the frame length'''
        '''Returns the indices which have larger than the distance seen in the previous frame '''
        
        cum_dist_filter = np.array(pos_array + self.cum_distance > 0.03)
        
        pos_filter = pos_array > p * frame_length
        
        return np.where(np.logical_and(cum_dist_filter, pos_filter))

    
    def insert(self, pos_array, velocity, acceleration, frame_length, gps=None):

        dist = velocity * (0.025) + 0.5 * acceleration * (0.025 * 0.025)

        residual_distance = 0

        p = (1 - (dist / frame_length)) * (self.found_seed)
        self.found_seed = False
        
        if (pos_array.size > 0):
            
            self.frames_seeds_dict[self.frame_no] = pos_array
            
            filter_array = self.filter_out_seen_seeds(pos_array, p, frame_length)
            
            pos_array = pos_array[filter_array]

            # Filter out the seeds seen with in the last frame
            if (pos_array.size > 0):

                pos_array = np.sort(pos_array)
                residual_distance = pos_array[-1]

                if (self.first_seed):
                    self.seed_attributes.append(SeedAttributes(distance=self.cum_distance + pos_array[0], seed_loc=pos_array[0], velocity=velocity, acceleration=acceleration, gps=gps, frame_no=self.frame_no))

                # if there are more than one unseen seed in the image
                if (pos_array.size > 1):

                    diff_list = list(np.diff(pos_array))

                    for i, dist in enumerate(diff_list):
                        self.seed_attributes.append(
                            SeedAttributes(distance= dist, seed_loc=pos_array[1+i],
                                           velocity=velocity, acceleration=acceleration, gps=gps, frame_no = self.frame_no))


                self.cum_distance = 0

            self.first_seed = True
            self.found_seed = True


        self.cum_distance += dist - residual_distance if self.first_seed else 0

        self.frame_no += 1


class SeedSpacingCalculator:

    def __init__(self, video_file_path, csv_data_frame, Model, x_factor=0.13335):

        self.model = Model
        self.cap = cv.VideoCapture(video_file_path)
        self.csv_file = csv_data_frame
        self.frame_idx = 0
        self.seeked = 0
        self.frame_width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.x_factor = x_factor

    def seek(self, percent=0.25):

        total_frames = self.cap.get(cv.CAP_PROP_FRAME_COUNT)

        if (percent > 1):
            print("Unable to Seek As percentage is greater than 1, p = {}".format(percent))
            return

        self.seeked = int(total_frames * percent)
        self.frame_idx = self.seeked
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.seeked)

    def get_seed_loc(self, frame, threshold):

        single_img_size = int(self.frame_height // 4)

        frames = [cv.cvtColor(frame[i * single_img_size:(i + 1) * single_img_size, :, :], cv.COLOR_BGR2RGB) for i in
                  range(4)]

        outs = self.model.predict(frames)

        predictions = {}

        for i, out in enumerate(outs):
            
            scores = torch.gt(out["scores"], threshold)
            predictions[i] = self.pixel_space_to_world_cords(out["boxes"][scores, :].cpu())

        return predictions

    def get_accel(self):

        vel_now = 0.277778 * self.csv_file['Amplitude - Vel'][self.frame_idx]

        if self.frame_idx + 4 >= len(self.csv_file):
            return 0

        vel_next = 0.277778 * self.csv_file['Amplitude - Vel'][self.frame_idx + 4]

        return (vel_next - vel_now) / 0.1


    def seek_frame(self, frame_idx):

        total_frames = self.cap.get(cv.CAP_PROP_FRAME_COUNT)

        if (frame_idx > total_frames):
            print("Unable to Seek Frame index to be seeked is {} greater than total frames".format(frame_idx,
                                                                                                   total_frames))
            return

        self.frame_idx = frame_idx
        self.seeked = frame_idx
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)

    def pixel_space_to_world_cords(self, tensor_pixel_space):

        if (tensor_pixel_space.nelement() == 0):
            return tensor_pixel_space.numpy()

        if (tensor_pixel_space.shape[0] > 0):
            x = ((tensor_pixel_space[:, 2] + tensor_pixel_space[:, 0]) / (2 * self.frame_width)) * self.x_factor

        else:
            return tensor_pixel_space.numpy()

        return x.numpy()

    def getSpacing(self, max_count=None, threshold=0.5, reset=True):

        if (not max_count):
            max_count = self.cap.get(cv.CAP_PROP_FRAME_COUNT) - self.seeked

        distances = {}
        for camera in range(4):
            distances[camera] = distance_struct(self.seeked)
        print(self.frame_idx - self.seeked)
        while (self.frame_idx - self.seeked < max_count):
            ret, frame = self.cap.read()
            if (not ret):
                print("Loop Broken Before Reaching The End Point")
                break

            velocity = 0.277778 * self.csv_file['Amplitude - Vel'][self.frame_idx]
            gps = (self.csv_file['Amplitude - Long'], self.csv_file['Amplitude - Long'])
            acceleration = self.get_accel()
            self.frame_idx += 1
            predictions = self.get_seed_loc(frame, threshold)

            for camera in predictions.keys():
                distances[camera].insert(predictions[camera], velocity, acceleration, self.x_factor, gps)

            output_msg = ""

            for camera in range(4):
                output_msg += " CAM {} POP : {}".format(camera, len(distances[camera].seed_attributes) + 1)
            print("FN {} {}".format(self.frame_idx, output_msg), end="\r")

        if (reset):
            self.frame_idx = 0
            self.seeked = 0
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        return distances

if "__main__" == __name__:
    pass