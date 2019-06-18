import cv2
from image_processing import puzzle_image_processing

normal_image = cv2.imread("images/image.jpg")
#puzzlepieces, slotpieces = main_image_processing.init_pieces_and_slots(normal_image)

piece_contours = puzzle_image_processing.get_piece_contours(puzzle_image_processing.process_img(normal_image, 50))

#main_image_processing.show_contours_onebyone()

print(len(piece_contours))

puzzle_image_processing.draw_contours(piece_contours, normal_image)
puzzle_image_processing.show(normal_image)

for contour in piece_contours:
    handle_center = puzzle_image_processing.get_handle_coordinates(contour, normal_image)
    print("coordinate:", handle_center)

for contour in piece_contours:
    contour = puzzle_image_processing.get_handle_circle(contour, normal_image)
    puzzle_image_processing.draw_contours([contour], normal_image)

puzzle_image_processing.show(normal_image)


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