import cv2
import main_image_processing

normal_image = cv2.imread("images/image21.jpg")
puzzlepieces, slotpieces = main_image_processing.init_pieces_and_slots(normal_image)

main_image_processing.show_matches(puzzlepieces, normal_image)

v = main_image_processing.check_initialization(puzzlepieces, slotpieces)
print("succesful:", v)

main_image_processing.pretty_print(puzzlepieces)

# if (main_image_processing.check_initialization(puzzlepieces, slotpieces)):
#     i = 1
#     for piece in puzzlepieces:
#         #take puzzlepiece by piece.handle_center
#         #go up and turn by piece.angle
#         #drop at slot.center
#         #go home
#         print("do piece:", i)
#         i += 1
# else:
#     print("initialization failed")