import cv2
import numpy as np

class VideoFile():
    def __init__(self, src, width, height):
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(src)

    # Displaces video position by the specified number of frames
    def set_pos(self, disp, unit='ms'):
        if unit == 'ms':
            self.cap.set(0, disp)
        elif unit == 'frames':
            self.cap.set(1, disp)
        else:
            raise ValueError

    # Extracts frames from clip using OpenCV
    def vid2npy(self, n_frames):
        frames =[]
        for _ in range(n_frames):
            ret, frame = self.cap.read()
            if ret == True:
                frame = cv2.resize(frame, (self.width,self.height), interpolation=cv2.INTER_AREA)
                frames.append(frame)
            else:
                break
        
        output = np.array(frames)
        output = (output / 255) * 2 - 1

        frame_count = output.shape[0]

        # If n_frames is underfilled, zero-pads remaining frames
        if frame_count < n_frames:
            pad = np.zeros(n_frames - frame_count, self.width, self.height, 3)
            output = np.concatenate((output, pad), axis=0)

        return output