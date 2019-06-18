import cv2
import numpy as np
import math

#GLOBAL VARIABLES

DISTANCE_CAMERA_TO_BUTTOM = 27624 # 120cm
HANDLE_HEIGHT = 460 # 2cm
IMAGE_CENTER = (512, 384)

# Classes
class PuzzlePiece:
    def __init__(self, contour, handle_center, match, angle=None):
        """
        Constructor for a PuzzlePiece
        :param contour: the contour of the puzzlepiece
        :param handle_center: the center of the handle of the puzzlepiece
        :param match: the match of the puzzlepiece
        :type match: SlotPiece
        :param angle: the angle of the puzzlepiece in relation to the slotpiece match.
        """
        self.contour = contour
        self.handle_center = handle_center
        self.match = match  # slotpiece
        self.angle = angle

    def pretty_print(self):
        """
        method to pretty print the puzzlepiece.
        """
        print("type: piece")
        if self.contour is not None:
            print("Contour: found")
        else:
            print("Contour: not found")
        print("handle_center:", self.handle_center)
        print("angle:", self.angle)

class SlotPiece:
    def __init__(self, contour, match):
        """
        Constructor for a SlotPiece
        :param contour: the contour of the slotpiece
        :param match: the match of the slotpiece
        :type match: PuzzlePiece
        """
        self.contour = contour
        self.center = None
        self.match = match  # puzzlepiece

    def pretty_print(self):
        """
        method to pretty print the slotpiece.
        :return:
        :rtype:
        """
        print("type: slot")
        if self.contour is not None:
            print("Contour: found")
        else:
            print("Contour: not found")
        print("center:", self.center)

class IMG_Processing:
    def __init__(self):
        pass

def show(img):
    """
    method to show the given image. The image will dissapear when pressing any button.
    :param img: the image to show
    """
    cv2.imshow("img", img)
    cv2.waitKey(0)


def show_contours_onebyone(pieces, img):
    """
    method to show the contours of the given pieces. The next contour will appear when any button is pressed.
    :param pieces: PuzzlePiece or SlotPiece objects
    :param img: the image to show the contours in
    """
    for piece in pieces:
        contour = piece.contour
        cv2.drawContours(img, [contour], -1, (0, 255, 0), -1)
        cv2.imshow("Contour image", img)
        cv2.waitKey(0)


def find_board_contour_idx(contours):
    """
    method to find the idx of the board contour by retrieving the contour with the highest length.
    :param contours: a bunch of contours
    :return the index of the board contour
    """
    idx = 0
    save = 0
    for contour in contours:
        if (cv2.arcLength(contour, True) > 2000):
            save = idx
        idx = idx + 1
    return save


def find_board_contour(img):
    """
    methhod to find a board contour in the given image.
    :param img: the image containing the puzzle board
    :return: the board contour
    """
    img = process_img(img, 50)
    contours = get_contours_ccomp(img)
    idx = 0
    curr_contour = None
    for contour in contours:
        if (cv2.arcLength(contour, True) > 2000):
            curr_contour = contour
        idx = idx + 1
    return curr_contour


def get_contours_external(img):
    """
    method to get the contours of a given image by using RETR_EXTERNAL. This will retrieve only external contours.
    :param img: The image should already been processed for contour detection.
    :return: contours
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def get_contours_ccomp(img):
    """
    method to get the contours of a given image by using RETR_CCOMP. This will retrieve internal and external contours.
    :param img: The image should already been processed for contour detection.
    :type img:
    :return:
    :rtype:
    """
    contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_board_area(contour):
    """
    method to get the area of the given contour
    :param contour: the contour
    :return: the area of the contour
    """
    return cv2.contourArea(contour)


def get_cut_contour(contour, img):
    """
    method to get a cut contour image.
    :param contour: the contour to be cut out.
    :param img: the image containing the contour. The image is not processed.
    :return: a cut contour image.
    """
    stencil = np.zeros(img.shape).astype(img.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [contour], color)
    result = cv2.bitwise_and(img, stencil)
    return result


def get_piece_contours(img):
    """
    method to get the piece contours in the given image
    :param img: the image.
    :return: the contours of the pieces
    """

    #img = (img - 255)
    piece_contours = []
    all_contours = get_contours_ccomp(img)

    board_area = get_board_area(all_contours[find_board_contour_idx(all_contours)])
    for contour in all_contours:
        if (cv2.contourArea(contour) > 10000
                and cv2.contourArea(contour) < board_area and cv2.contourArea(
                contour) > 0):
            piece_contours.append(contour)
    return piece_contours


def get_slot_contours(img):
    """
    method to get the slot contours of the given image.
    :param img: the image that is not processed.
    :return: the contours of the slotpieces.
    """

    slot_contours = []
    cut_board_img = process_for_slots(img)

    contours = get_contours_external(cut_board_img)

    for contour in contours:
        if (cv2.contourArea(contour) > 10000 and cv2.contourArea(contour) > 0):
            slot_contours.append(contour)
    return slot_contours

def find_match(contour, contours):
    """
    method to find the match of a contour. The match is found by using the cv2.matchShapes() method.
    :param contour: the contour to find a match for.
    :param contours: the contours including the match
    :type contours:
    :return: the contour that matches best
    """
    best_match = 1
    best_contour = None
    for i in range(len(contours)):
        match_value = cv2.matchShapes(contours[i], contour, 3, 0.0)
        if (match_value <= best_match):
            best_match = match_value
            best_contour = contours[i]

    if (match_value <= 1):
        return best_contour
    else:
        print("Error. Match value is too high.")
        return best_contour

def show_matches(puzzlepieces, img):
    """
    method to show all matches of the puzzlepieces and slots.
    :param puzzlepieces:
    :param img: the image in which the matches will be displayed.
    """
    for puzzlepiece in puzzlepieces:
        img = img.copy()
        slot_contour = puzzlepiece.match.contour
        piece_contour = puzzlepiece.contour
        cv2.drawContours(img, [slot_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(img, [piece_contour], -1, (0, 255, 0), 2)
        cv2.imshow("Matches", img)
        cv2.waitKey(0)

def find_center(contour):
    """
    method to find the center of the given contour.
    :param contour:
    :return: the center point
    """
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"]) if M["m00"] else 0
    cY = int(M["m01"] / M["m00"]) if M["m00"] else 0
    return (cX, cY)


def draw_contours(contours, img):
    """
    method to draw contours to an image.
    :param contours:
    :type contours:
    :param img: the image to draw the contours in.
    """
    cv2.drawContours(img, contours, -1, (0, 255, 255), 2)

def black_background(img, contour):
    """
    method to black the background of the given image except for the contour
    :param img:
    :type img:
    :param contour:
    :type contour:
    :return: the image with a blacked background.
    """
    stencil = np.zeros(img.shape).astype(img.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [contour], color)
    result = cv2.bitwise_and(img, stencil)
    return result

def get_handle_circle(contour, img):
    """
    method to get the circle of the handle of a puzzlepiece
    :param contour: the contour of a puzzlepiece
    :param img: the image that is not processed.
    :return: the coordinates of the handle.
    """

    cut_piece_img = get_cut_contour(contour, img)
    gray_img = cv2.cvtColor(cut_piece_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY)
    thresh = black_background(thresh, contour)

    contours = get_contours_ccomp(thresh)
    new_contours = []

    #show(thresh)

    for curr_contour in contours:
        #print("area:", cv2.contourArea(curr_contour, True))
        #print("center difference:", abs(find_center(contour)[0] - find_center(curr_contour)[0]))
        #print("Länge:", cv2.arcLength(curr_contour, True))
        #draw_contours([curr_contour], cut_piece_img)
        #show(cut_piece_img)
        if (isValidCircle(curr_contour, contour)):
            new_contours.append(curr_contour)

    if (len(new_contours) == 1):
        return new_contours[0]
    else:
        print(len(new_contours))
        print("Error! Did not find circle correctly.")
        return None


def isValidCircle(curr_contour, contour):
    """
    method to check if the circle is valid by checking if the area, the length and the distance to the center of the contour is in the expected range.
    :param curr_contour: the contour of the possible handle
    :param contour: the contour of the whole puzzlepiece
    :return:
    """
    center = find_center(contour)
    curr_center = find_center(curr_contour)
    area = cv2.contourArea(curr_contour, True)
    # valid circle if: area is ok, length is ok and distance to center is ok
    if (cv2.arcLength(curr_contour, True) > 50 and cv2.arcLength(curr_contour, True) < 70 and abs(
            curr_center[0] - center[0]) < 20 and abs(
        curr_center[1] - center[1]) < 20 and abs(area) > 245 and abs(area) < 320):
        return True
    else:
        return False


def get_handle_coordinates(contour, img):
    """
    method to get the coordinates of the handle.
    :param contour: the contour of the handle.
    :param img:
    :return: the coordinates of the handle
    """
    handle_circle = get_handle_circle(contour, img)
    if handle_circle is not None:
        wrong_center = find_center(handle_circle)
        correct_center = correctParallaxEffect(wrong_center)

        return correct_center
    else:
        return (0,0)

def show_handle_circles(puzzlepieces, img):
    """
    method to show all handle circles in the given image.
    :param puzzlepieces:
    :param img:
    """
    for piece in puzzlepieces:
        contour = get_handle_circle(piece.contour, img)
        draw_contours([contour], img)

    show(img)

def show_handle_centers(puzzlepieces, img):
    for piece in puzzlepieces:
        cv2.circle(img, piece.handle_center, 1, (0,255,255), 0)

    show(img)

def show_slot_centers(slots, img):
    for slot in slots:
        cv2.circle(img, slot.center, 1, (0,255,255), 0)

    show(img)


def rotateImage(image, angle, image_center):
    """
    method to rotate an image according to the given angle and the given image center
    :param image:
    :param angle:
    :param image_center: the center of rotation
    :return: the rotated image
    """
    # image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate_piece(puzzlepiece, angle, img):
    """
    method to rotate the given puzzlepiece according to the given angle in the given image.
    :param puzzlepiece:
    :param angle:
    :param img:
    :return: the rotated image
    """
    mask = get_mask_image(puzzlepiece, img)
    processed_img = process_img(mask.copy(), 40)
    contours = get_contours_external(processed_img)
    center = find_center(contours[0])

    return rotateImage(mask, angle, center)

def get_mask_image(piece, img):
    """
    method to crop the given piece and to put it into the middle of a mask.
    :param piece:
    :param img:
    :return: an image of the piece in a masked image
    """
    cropped = get_cropped_contour(piece, img)
    mask = np.zeros((512, 512, 3), np.uint8)
    x_offset = y_offset = 200

    mask[y_offset:y_offset + cropped.shape[0], x_offset:x_offset + cropped.shape[1]] = cropped

    return mask


def get_base_edge(rect):
    """
    method to get the long side of the given rectangle
    :param rect:
    :type rect:
    :return: two coordinates representing the line of the base edge.
    """
    box = get_rect_contour(rect)
    fixed_point = (box[0][0], box[0][1])
    point_distance_list = []
    for i in range(4):
        curr_point = (box[i][0], box[i][1])
        curr_distance = get_distance(fixed_point, curr_point)
        point_distance_list.append((curr_point, curr_distance))
    point_distance_list = sorted(point_distance_list, key=lambda x: x[1])
    return (fixed_point, point_distance_list[2][0])


def get_distance(point1, point2):
    """
    method to get the distance between two points.
    :param point1:
    :type point1:
    :param point2:
    :type point2:
    :return:
    :rtype:
    """
    distance = math.sqrt(math.pow((point2[0] - point1[0]), 2) + math.pow((point2[1] - point1[1]), 2))
    return distance


def get_slope(line):
    """
    method to get the slope of the given line
    :param line:
    :type line:
    :return: the slope
    """
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[1][0]
    y2 = line[1][1]

    if (x2 == x1):
        return ((y2 - y1) / 1)
    else:
        return ((y2 - y1) / (x2 - x1))


def draw_rect(img, rect):
    """
    method to draw a rectangle in the given image.
    :param img:
    :param rect:
    """
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], -1, (255, 255, 0), 2)


def get_rect_contour(rect):
    """
    method to get the contour of the given rect.
    :param rect:
    :return: the contour of the rect
    """
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def draw_edge(img, edge):
    """
    method to draw an edge in the given image.
    :param img:
    :param edge:
    """
    cv2.line(img, edge[0], edge[1], (0, 0, 255), 3)


def get_angle(edge1, edge2):
    """
    method to get the angle between two edges.
    :param edge1:
    :param edge2:
    :return: the angle between the edges in degrees.
    """
    slope1 = get_slope(edge1)
    slope2 = get_slope(edge2)

    rad_angle = math.atan((slope1 - slope2) / (1 + (slope1 * slope2)))
    degree_angle = (rad_angle * 180 / math.pi)
    return degree_angle


def get_piece_angle(puzzlepiece):
    """
    method to get the relative angle of the given puzzlepiece to its match.
    :param puzzlepiece:
    :return: the angle
    """
    edge1 = get_base_edge(get_rect(puzzlepiece.contour))
    edge2 = get_base_edge(get_rect(puzzlepiece.match.contour))
    angle = get_angle(edge1, edge2)
    return angle


def get_rect(contour):
    """
    method to get the minAreaRectangle of the given contour
    :param contour:
    :return: minAreaRect
    """
    return cv2.minAreaRect(contour)


def process_cropped(img):
    """
    method to process a cropped image. This method is used to find the handle circles.
    :param img: the image containing only one puzzlepiece
    :return: the processed image
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred_img = cv2.medianBlur(gray_img, 5)
    # edged_img = cv2.Canny(blurred_img, 63, 180) # these parameters are important. The image detection behaves differently when changing the contrast.
    ret, threshold = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)
    # threshold = 255 - threshold  # invert the coloring
    return threshold


def overlay_contours(contour1, contour2):
    """
    method to overlay to contours in a mask
    :param contour1:
    :param contour2:
    :return: the image with the mask and the two overlaying contours.
    """
    img = np.zeros((512, 512, 3), np.uint8)
    draw_contours(contour1, img)
    draw_contours(contour2, img)
    return img


def get_cropped_contour(piece, img):
    """
    method to get the cropped image of the piece contour
    :param piece:
    :param img:
    :return: the cropped image of the piece contour
    """
    cut_piece_img = get_cut_contour(piece.contour, img)
    x, y, w, h = cv2.boundingRect(piece.contour)
    cropped = cut_piece_img[y:y + h, x:x + w]
    return cropped


def get_overlay_area(puzzlepiece, angle, img):
    """
    This method calculates the overlay area of the puzzlepiece contour and its matched contour.
    :param puzzlepiece:
    :param angle:
    :param img:
    :return: the external area of two contours overlaying each other
    """
    rotated_img = rotate_piece(puzzlepiece, angle, img)
    processed = process_img(rotated_img, 40)
    processed = (255 - processed)
    contours_puzzlepiece = get_contours_external(processed)
    draw_contours(contours_puzzlepiece, rotated_img)

    slot_img = get_mask_image(puzzlepiece.match, img)
    processed = process_cropped(slot_img)
    contours_slotpiece = get_contours_external(processed)
    draw_contours(contours_slotpiece, slot_img)

    ## (2) Calc offset
    center_puzzle = find_center(contours_puzzlepiece[0])
    center_slot = find_center(contours_slotpiece[0])
    dx = center_puzzle[0] - center_slot[0]
    dy = center_puzzle[1] - center_slot[1]

    img = draw_overlaying_contours_to_mask([contours_puzzlepiece, contours_slotpiece], (dx, dy))
    # draw_rect(img, cv2.minAreaRect(contours_slotpiece[0]))
    # draw_rect(img, cv2.minAreaRect(contours_puzzlepiece[0]))
    # show(img)

    img = process_img(img, 20)
    img = (255 - img)
    overall_contour = get_contours_external(img)

    overall_img = draw_contours_to_mask([overall_contour])
    overall_contour_area = cv2.contourArea(overall_contour[0], True)

    return overall_contour_area


def draw_contours_to_mask(contours):
    """
    method to draw the given contours to a mask
    :param contours:
    :return: the contours in a mask
    """
    img = np.zeros((512, 512, 3), np.uint8)
    for i in range(len(contours)):
        cv2.drawContours(img, contours[i], -1, (0, 255, 0), 2)

    return img


def draw_overlaying_contours_to_mask(contours, offset):
    """
    method to draw the given contours to a mask
    :param contours:
    :param offset: the offset to put the contours to the center of the image
    :return: the contours in a mask
    """
    assert len(contours) == 2
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.drawContours(img, contours[0], -1, (0, 255, 0), 2)
    cv2.drawContours(img, contours[1], -1, (0, 255, 0), 2, offset=offset)

    return img


def adjust_angle(puzzlepiece, img):
    """
    method to detect the correct relative angle between puzzlepiece and slot. The pre-calculation of this method in this module calculates the angle
    between puzzlepiece and slot, without taking the alignment of the puzzlepiece into consideration. The pre-calculation could therefore result in
    the puzzlepiece to be upside down. To solve the problem, this method calculates the overlaying area of the contours with the pre-calculated angle
    and with the (pre-calculated angle + 180). If the puzzlepiece is upside down, the overlaying area of the contours will be bigger than if they
    match correctly. This way the correct alignment will be detected. This method sets the angle of the puzzlepiece accordingly.
    :param puzzlepiece:
    :type puzzlepiece:
    :param img:
    """
    area1 = get_overlay_area(puzzlepiece, puzzlepiece.angle, img)
    area2 = get_overlay_area(puzzlepiece, puzzlepiece.angle + 180, img)

    if (abs(area1) > abs(area2)):
        puzzlepiece.angle = puzzlepiece.angle + 180


def process_for_pieces(img):
    """
    method to process the given image for puzzlepiece detection. The threshold value of 50 is used.
    :param img:
    :return: the threshold of the given image.
    :rtype:
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred_img = cv2.medianBlur(gray_img, 5)
    # edged_img = cv2.Canny(blurred_img, 63, 180) # these parameters are important. The image detection behaves differently when changing the contrast.
    ret, threshold = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
    threshold = 255 - threshold  # invert the coloring
    return threshold


def process_for_slots(img):
    """
    method to process the given image for slot detection. The threshold value of 65 is used. This method cuts out the puzzleboard for further processing.
    :param img:
    :return: the cut board image.
    """
    cut_board_contour = find_board_contour(img)
    cut_board_img = get_cut_contour(cut_board_contour, img)
    cut_board_img = process_img(cut_board_img, 95) # this value is 65 for darker images
    cut_board_img = black_background(cut_board_img, cut_board_contour)
    cv2.drawContours(cut_board_img, [cut_board_contour], -1, (0, 0, 0), 2)
    return cut_board_img


def process_img(img, threshold_value):
    """
    method to process an image with dynamic threshold value
    :param img:
    :param threshold_value:
    :return: the theshold of the given image
    """

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred_img = cv2.medianBlur(gray_img, 5)
    # edged_img = cv2.Canny(blurred_img, 63, 180) # these parameters are important. The image detection behaves differently when changing the contrast.
    ret, threshold = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    threshold = 255 - threshold  # invert the coloring
    return threshold


def get_point_difference(point1, point2):
    """
    method to calculate the difference between two points.
    :param point1:
    :param point2:
    :return: the difference between two points
    """
    x = point1[0] - point2[0]
    y = point1[1] - point2[1]
    return (x, y)


def rotate_vector(vector, angle):
    """
    method to rotate a vector according to the given angle.
    :param vector:
    :param angle:
    :return: the rotated vector
    """
    angle = -angle + 180
    rad = math.radians(angle)
    # print("rad:", rad)
    x = vector[0]
    y = vector[1]

    x_prime = x * math.cos(rad) - y * math.sin(rad)
    y_prime = x * math.sin(rad) + y * math.cos(rad)

    return (x_prime, y_prime)


def get_slot_center(slotpiece):
    """
    method to get the center of a slot according the handle of the matching puzzlepiece.
    :param slotpiece:
    :return: the slot center.
    """
    puzzle_contour = slotpiece.match.contour
    puzzle_contour_center = find_center(puzzle_contour)
    handle_center = slotpiece.match.handle_center

    difference = get_point_difference(puzzle_contour_center, handle_center)
    vector = rotate_vector(difference, (slotpiece.match.angle))

    slot_contour = slotpiece.contour
    slot_contour_center = find_center(slot_contour)

    result = (slot_contour_center[0] + vector[0], slot_contour_center[1] + vector[1])

    return (int(result[0]), int(result[1]))


def rotate_contour(contour, angle):
    """
    method to rotate a contour according to the given angle. This method rotates around the center.
    :param contour:
    :param angle:
    :return: the rotated contour.
    """
    center = find_center(contour)
    new_contour = contour.copy()
    for i in range(len(contour)):
        point = get_point_difference(center, new_contour[i][0])
        new_contour[i][0] = rotate_vector(point, angle)
        # contour[i][0] = ((contour[i][0][0] + center[0]), contour[i][0][1] + center[1])
    return new_contour

def shift_contour(contour, center):
    """
    method to shift a contour to another center.
    :param contour: the contour to be shifted.
    :param center: the new center
    :return: the contour shifted to another center.
    """
    new_contour = contour.copy()
    for i in range(len(contour)):
        new_contour[i][0] = add_points(new_contour[i][0], center)
    return new_contour

def add_points(point1, point2):
    """
    method to add two points
    :param point1:
    :param point2:
    :return: the result
    """
    x = point1[0] + point2[0]
    y = point1[1] + point2[1]
    return (x, y)

def check_initialization(puzzlepieces, slotpieces):
    return check_contours(slotpieces, puzzlepieces) and check_centers(slotpieces, puzzlepieces)
    #check_angle(puzzlepieces)
    #check_matches(puzzlepieces)


def check_contours(slotpieces, puzzlepieces):
    if (len(slotpieces) != len(puzzlepieces)):
        return False
    for i in range(len(slotpieces)):
        if (slotpieces[i].contour is None or puzzlepieces[i].contour is None):
            return False
    return True

def check_centers(slotpieces, puzzlepieces):
    if (len(slotpieces) != len(puzzlepieces)):
        return False
    for i in range(len(slotpieces)):
        if (slotpieces[i].center == (0,0) or puzzlepieces[i].handle_center == (0,0)):
            return False
    return True

def pretty_print(puzzlepieces):
    for piece in puzzlepieces:
        piece.pretty_print()
        piece.match.pretty_print()
        print("****************************")

def correctParallaxEffect(wrong_center):
    distance_center_to_handle_center = get_distance(IMAGE_CENTER, wrong_center)
    parallaxError = (distance_center_to_handle_center / DISTANCE_CAMERA_TO_BUTTOM) * HANDLE_HEIGHT
    point_difference = get_point_difference(IMAGE_CENTER, wrong_center)

    directionVector = normalize_vector(point_difference)

    correctionVector  = (parallaxError*directionVector[0], parallaxError*directionVector[1])
    corrected_handle_center = add_points(wrong_center, correctionVector)
    return (int(corrected_handle_center[0]), int(corrected_handle_center[1]))

def normalize_vector(vector):
    amount = math.sqrt(vector[0]**2 + vector[1]**2)
    correctionVector = (vector[0]/amount, vector[1]/amount)
    return correctionVector



    pass

def init_pieces_and_slots(img):
    """
    method to init all puzzlepieces and slotpieces of the given image. This method will process and the image and it will retrieve and store all relevant information
    inside the PuzzlePiece and the SlotPiece objects.
    :param img: the original image
    :return: lists of puzzlepieces and slotpieces
    """
    slot_contours = get_slot_contours(img)
    piece_contours = get_piece_contours(process_img(img, 50))

    puzzlepieces = []
    slotpieces = []
    for i in range(len(piece_contours)):
        # initializing puzzlepiece
        match_contour = find_match(piece_contours[i], slot_contours)
        puzzlepiece = PuzzlePiece(piece_contours[i], get_handle_coordinates(piece_contours[i], img),
                                  SlotPiece(match_contour, None))

        # initializing slotpiece
        slotpiece = puzzlepiece.match
        slotpiece.match = puzzlepiece
        puzzlepiece.match = slotpiece
        puzzlepiece.angle = get_piece_angle(puzzlepiece)
        adjust_angle(puzzlepiece, img)

        puzzlepiece.angle = (puzzlepiece.angle + 360) % 360

        slotpiece.center = get_slot_center(slotpiece)
        slotpieces.append(slotpiece)
        puzzlepieces.append(puzzlepiece)

    return puzzlepieces, slotpieces

def overlay_piece_to_slot(piece):
    contour = rotate_contour(piece.contour, piece.angle)
    contour = shift_contour(contour, find_center(piece.match.contour))

    return contour

def draw_point(coordinate, img, color):
    cv2.circle(img, (int(coordinate[0]), int(coordinate[1])), 1, color, 0)

if __name__ == '__main__':
    img = cv2.imread('images/blackandwhite.jpg')
    #puzzlepieces, slots = init_pieces_and_slots(img)

    #for piece in puzzlepieces:
    #    vector = get_point_difference(find_center(piece.contour), piece.handle_center)
    #    vector = rotate_vector(vector, piece.angle)
    #    new_contour = overlay_piece_to_slot(piece)
    #    point =  add_points(find_center(new_contour), vector)

    #    draw_point(point, img, (0,255,255))
    #    draw_point(piece.match.center, img, (0,0,255))

    #show(img)

    show(img)
    thresh = process_img(img, 130)
    show(thresh)

    contours = get_contours_ccomp(thresh)

    for contour in contours:
        length = cv2.arcLength(contour, True)
        if (length > 300 and length < 350):
            print(length)
            draw_contours([contour], img)
            show(img)