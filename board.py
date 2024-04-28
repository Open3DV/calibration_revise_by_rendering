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

def generate_obj_points(circles):
    # first sort the circles by their y coordinate, then by their x coordinate
    sorted_circles = sorted(circles, key=lambda x: (x[1], x[0]))
    

    #sorted_circles = sorted(circles, key=lambda x: x[0])
    #sorted_circles = sorted(sorted_circles, key=lambda x: x[1])
    # generate the obj points
    obj_points = []
    for circle in sorted_circles:
        cx = circle[0]
        cy = circle[1]
        obj_points.append([cx, cy, 0])
    return np.float32(obj_points)

def read_board_svg(svg_path):
    rect, circles = read_svg(svg_path)
    obj_points = generate_obj_points(circles)
    board = draw_board(rect, circles)
    return obj_points, board

obj_points, board = read_board_svg('board_svg/300x240.svg')

# rect, circles = read_svg('board_svg/300x240.svg')

# print(circles)

# board = draw_board(rect, circles)

# cv2.imwrite('board_300x240.png', board)
