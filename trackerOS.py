#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 
import numpy as np 
import time 
import sdl2 
import sdl2.ext
# from lane_detection import color_frame_pipeline
from camera import CameraCalibration
from gradients import get_edges
from perspective import flatten_perspective
from tracker import LaneTracker


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


import glob

W = 1920/2
H = 1080/2
sdl2.ext.init()

window = sdl2.ext.Window('tracker', size=(W,H), position=(500, 500))
window.show()

calibrate = CameraCalibration(glob.glob('/Users/michaelsafir/Downloads/detecting-road-features-master/data/camera_cal/calibration*.jpg'), retain_calibration_images=True)


class FeatureExtractor(object):
	GX = 16//2
	GY = 12//2
	"""docstring for FeatureExtractor"""
	def __init__(self):
		self.orb = cv2.ORB_create(100)
		self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		self.last = None

	def extract(self, img):
		# sy = img.shape[0]//self.GY
		# sx = img.shape[1]//self.GX
		# akp = []
		# for ry in range(0, img.shape[0], sy):
		# 	for rx in range(0, img.shape[1], sx):
		# 		img_chunk = img[ry:ry+sy, rx:rx+sx]
		# 		kp = self.orb.detect(img_chunk, None)
		# 		for p in kp:
		# 			p.pt = (p.pt[0] + rx, p.pt[1] + ry)
		# 			akp.append(p)
		# return akp

		feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=1)
		kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
		kps, des = self.orb.compute(img,kps)
		

		matches = None
		if self.last is not None:
			matches = self.bf.match(des, self.last['des'])
			matches = zip([kps[m.queryIdx] for m in matches], [self.last['kps'][m.trainIdx] for m in matches])

		self.last = {'kps' : kps, 'des' : des}
		
		return matches
		
		

fe = FeatureExtractor()

def process(load):

	img = cv2.resize(load, (W, H))


	# calibrated = calibrate(img)
	# lane_tracker = LaneTracker(calibrated)
	# overlay_frame = lane_tracker.process(calibrated, draw_lane=True, draw_statistics=False)
	# img = overlay_frame

	# out_image = color_frame_pipeline([img], solid_lines=True)
	# img = out_image

	matches = fe.extract(img)

	if matches is None:
		return

	for pt1, pt2 in matches:
		u1,v1 = map(lambda x: int(round(x)), pt1.pt)
		u2,v2 = map(lambda x: int(round(x)), pt2.pt)
		cv2.circle(img, (u1,v1) , color=(0,255,0), radius=3)
		# cv2.line(img, (u1, v2), (u2, v2), color=(255,0,0))


	

	events = sdl2.ext.get_events()
	for event in events:
		if event.type == sdl2.SDL_QUIT:
			print("OUT DONE")
			exit(0)
	# print(dir(window))

	surf = sdl2.ext.pixels3d(window.get_surface())
	surf[:, :, 0:3] = img.swapaxes(0,1)


	window.refresh()



# cap = cv2.VideoCapture(1)


if __name__ == '__main__':

	while True:
		cap = cv2.VideoCapture(0)
		# cap = cv2.VideoCapture('images/video2.mp4')
		fourcc = cv2.VideoWriter_fourcc(*'DIVX')
		out = cv2.VideoWriter('output3.avi', fourcc, 20.0, (640,480))




		while cap.isOpened():
			ret, frame = cap.read()

			

			if ret == True:

				process(frame)


				# frame_new = cv2.flip(frame,0)
				out.write(frame)
				

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			else:
				break

		cap.release()
		out.release()
		cv2.destroyAllWindows()


	# while True:
	# 	img = cv2.imread('images/image1.jpeg')
	# 	process(img)
	# 	time.sleep(0.3)

	# 	img = cv2.imread('images/image2.jpeg')
	# 	process(img)
	# 	time.sleep(0.3)


	# 	img = cv2.imread('images/image3.jpeg')
	# 	process(img)
	# 	time.sleep(0.3)


