import cv2
from image_processing.camera_control import give_da_stream
import image_processing.puzzle_image_processing as imgp
from KukaCommunication.kuka_control import PuzzleSolver
import numpy as np
import os
import time

if __name__ == '__main__':

    path = 'image_processing/images/'
    puzzleSolved = False
    kuka = PuzzleSolver()
    kuka.go2Origin()

    while not puzzleSolved:
        for img in give_da_stream():
            if np.any(img):
                try:
                    puzzlepieces, slots = imgp.init_pieces_and_slots(img)
                    if (imgp.check_initialization(puzzlepieces, slots)):
                        puzzleSolved = len(puzzlepieces) == 0
                        if not puzzleSolved:
                            kuka.pick(xy=puzzlepieces[0].handle_center)
                            kuka.place(xy=puzzlepieces[0].match.center, angle= kuka.convert_angle(puzzlepieces[0].angle),doShaking='O')
                            kuka.go2Origin()
                            print("puzzlepiece placed")
                        else:
                            print("Puzzle solved.")
                            break
                except:
                    print("Error while processing the picture.")