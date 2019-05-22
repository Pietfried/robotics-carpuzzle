import cv2
import main_image_processing

normal_image = cv2.imread("images/image13.jpg")
puzzlepieces, slotpieces = main_image_processing.init_pieces_and_slots(normal_image)

if (main_image_processing.check_initialization(puzzlepieces, slotpieces)):
    i = 1
    for piece in puzzlepieces:
        #take puzzlepiece by piece.handle_center
        #go up and turn by piece.angle
        #drop at slot.center
        #go home
        print("do piece:", i)
        i += 1
else:
    print("initialization failed")