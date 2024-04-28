import os
import numpy as np
import cv2

def read_svg(svg_path):
    rect = []
    circles = []
    with open(svg_path, 'r') as fp:
        for line in fp:
            if '<rect' in line:
                items = line.split(' ')
                for item in items:
                    if 'width' in item:
                        width = float(item.split('"')[1])
                    if 'height' in item:
                        height = float(item.split('"')[1])
                rect = [width, height]
            if '<circle' in line:
                items = line.split(' ')
                for item in items:
                    if 'cx' in item:
                        cx = float(item.split('"')[1])
                    if 'cy' in item:
                        cy = float(item.split('"')[1])
                    if 'r=' in item:
                        r = float(item.split('"')[1])
                circles.append([cx, cy, r])
    return rect, circles
    

def draw_board(rect, circles, scale=10):
    board = np.zeros([int(rect[1]*scale), int(rect[0]*scale)], dtype = np.uint8)
    for circle in circles:
        cx = int(circle[0]*scale)
        cy = int(circle[1]*scale)
        r = int(circle[2]*scale)
        cv2.circle(board, (cx, cy), r, (255), -1)
    return board



rect, circles = read_svg('board_svg/300x240.svg')

board = draw_board(rect, circles)

cv2.imwrite('board_300x240.png', board)
