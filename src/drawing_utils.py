import numpy as np
import cv2


def draw_label(I, l, color=(255, 0, 0), thickness=1, lp=False, lcar=None, C=None):
	if not lp:
		wh = np.array(I.shape[1::-1]).astype(float)
		tl = tuple((l.tl() * wh).astype(int).tolist())
		br = tuple((l.br() * wh).astype(int).tolist())
		cv2.rectangle(I, tl, br, color, thickness=thickness)
	else:
		wh = np.array(I.shape[1::-1]).astype(float)
		cwh = np.array(C.shape[1::-1]).astype(float)
		tl = tuple(np.add((l.tl() * cwh).astype(int).tolist(), (lcar.tl() * wh).astype(int).tolist()))
		br = tuple(np.add((l.br() * cwh).astype(int).tolist(), (lcar.tl() * wh).astype(int).tolist()))
		cv2.rectangle(I, tl, br, color, thickness=thickness)


def write2img(Img, label, strg, txt_color=(0, 0, 0), bg_color=(255, 255, 255), font_size=1):
	wh_img = np.array(Img.shape[1::-1])

	font = cv2.FONT_HERSHEY_SIMPLEX

	wh_text, v = cv2.getTextSize(strg, font, font_size, 3)
	bl_corner = label.tl() * wh_img

	tl_corner = np.array([bl_corner[0], bl_corner[1] - wh_text[1]]) / wh_img
	br_corner = np.array([bl_corner[0] + wh_text[0], bl_corner[1]]) / wh_img
	bl_corner /= wh_img

	if (tl_corner < 0.).any():
		delta = 0. - np.minimum(tl_corner, 0.)
	elif (br_corner > 1.).any():
		delta = 1. - np.maximum(br_corner, 1.)
	else:
		delta = 0.

	tl_corner += delta
	br_corner += delta
	bl_corner += delta

	tpl = lambda x: tuple((x * wh_img).astype(int).tolist())

	cv2.rectangle(Img, tpl(tl_corner), tpl(br_corner), bg_color, -1)
	cv2.putText(Img, strg, tpl(bl_corner), font, font_size, txt_color, 3)
