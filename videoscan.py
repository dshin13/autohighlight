# below is a script to write for a runnable script:

from utils.event_detector import videoscan
import numpy as np

videopath = '../SoccerNet-code/data/leave_out/2016-09-13 - 21-45 Barcelona 7 - 0 Celtic/1.mkv'
out = videoscan(videopath, './weights/0618_finetune/weights.07-0.64.hdf5', 75)
np.save('barca_celtics_half_1.npy', out)
