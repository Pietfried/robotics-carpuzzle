import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import math
from scipy import ndimage

def show(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)

def show_images():
        cv2.imshow("normal_img", normal_img)
        cv2.waitKey(0)
        cv2.imshow("edged_img", threshold)
        cv2.waitKey(0)

def invert_coloring(img):
    return (img-255)

def show_contours_onebyone(contours):
        i = 0
        for contour in contours:
                cv2.drawContours(normal_img, [contour], -1, (0, 255, 0), 2)
                cv2.imshow("Contour image", normal_img)
                cv2.waitKey(0)
                i = i + 1

def find_board_contour(contours):
        board_contours = []
        i = 0
        for contour in contours:
                i = i+1
                if (cv2.arcLength(contour, True) > 200):
                        cv2.drawContours(normal_img, [contour], -1, (0, 255, 0), 2)
                board_contours.append(contour)
        
        return board_contours

def show_contours_onebyone(contours):
    for contour in contours:
        cv2.drawContours(normal_img, [contour], -1, (0, 255, 0), -1)
        cv2.imshow("Contour image", normal_img)
        cv2.waitKey(0)

def show_contours_with_hierarchy(contours):
    i = 0
    for contour in contours:
        cv2.drawContours(normal_img, [contour], -1, (0, 255, 0), 2)
        cv2.imshow("Contour image", normal_img)
        cv2.waitKey(0)
        i = i+1

def find_board_contour_idx(contours):
    idx = 0
    save = 0
    for contour in contours:
        if (cv2.arcLength(contour, True) > 2000):
            save = idx
        idx = idx + 1
    return save

def find_board_contour(contours):
    idx = 0
    save = 0
    for contour in contours:
        if (cv2.arcLength(contour, True) > 2000):
            save = contour
        idx = idx + 1
    return save


def remove_doubles(contours):
    new_contours = []
    threshold = 500
    i = 0
    while i < len(contours):
        if (cv2.contourArea(contours[i]) - cv2.contourArea(contours[i+1]) >= threshold and cv2.contourArea(contours[i+1]) >= 12000):
            new_contours.append(contours[i+1])
        if(cv2.contourArea(contours[i]) >= 400):
            new_contours.append(contours[i])
        i = i + 2
    return new_contours

def get_contours_external(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def get_contours_ccomp(img):
    contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_cut_board_img(img, contours):
    idx = find_board_contour_idx(contours)  # The index of the contour that surrounds your object
    mask = np.zeros_like(img)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, contours, idx, 255, -1)  # Draw filled contour in mask
    cv2.drawContours(mask, contours, idx, (0, 0, 0), 5)
    cut_board_image = np.zeros_like(img)  # Extract out the object and place into output image
    cut_board_image[mask == 255] = img[mask == 255]
    return cut_board_image

def get_board_area(contour):

    return cv2.contourArea(contour)

def get_cut_contour(contour):
    stencil = np.zeros(normal_img.shape).astype(normal_img.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [contour], color)
    result = cv2.bitwise_and(normal_img, stencil)
    return result

def get_piece_contours(img):
    img = invert_coloring(img)
    piece_contours = []
    all_contours = get_contours_external(img)
    board_area = get_board_area(all_contours[find_board_contour_idx(all_contours)])
    for contour in all_contours:
        if (cv2.contourArea(contour) > 10000 and cv2.contourArea(contour) < board_area and cv2.contourArea(contour) > 0):
            piece_contours.append(contour)
    return piece_contours

def get_slot_contours(img):
    slot_contours = []
    cut_board_img = process_for_slots(img)

    contours = get_contours_external(cut_board_img)
    show(cut_board_img)

    for contour in contours:
        if (cv2.contourArea(contour) > 10000 and cv2.contourArea(contour) > 0):
            slot_contours.append(contour)
    return slot_contours

def find_matchtes(slot_contours, piece_contours):
    matches = []
    best_match = 1
    best_contour = None
    for i in range (len(slot_contours)):
        for j in range(len(piece_contours)):
            match_value = cv2.matchShapes(slot_contours[i], piece_contours[j], 3, 0.0)
            if (match_value <= best_match):
                best_match = match_value
                best_contour = piece_contours[j]
        matches.append((slot_contours[i], best_contour))
        best_match = 1
    return matches

def find_match(contour, contours):
    best_match = 1
    best_contour = None
    for i in range(len(contours)):
        match_value = cv2.matchShapes(contours[i], contour, 3, 0.0)
        if (match_value <= best_match):
            best_match = match_value
            best_contour = contours[i]
    return best_contour


def show_matches(matches):
    for i in range (len(matches)):
        img = normal_img.copy()
        slot_contour = matches[i][0]
        piece_contour = matches[i][1]
        cv2.drawContours(img, [slot_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(img, [piece_contour], -1, (0, 255, 0), 2)
        cv2.imshow("Matches", img)
        cv2.waitKey(0)

def find_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

def draw_contours(contours, img):
    cv2.drawContours(img, contours, -1, (0, 255, 255), 2)

#blacks the background except for the given image
def black_background(img, contour):
    stencil = np.zeros(img.shape).astype(img.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [contour], color)
    result = cv2.bitwise_and(img, stencil)
    return result

def get_handle_circle(contour):
    cut_piece_img = get_cut_contour(contour)
    gray_img = cv2.cvtColor(cut_piece_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 35, 255, cv2.THRESH_BINARY)
    thresh = (255 - thresh) # switch black and white
    thresh = black_background(thresh, contour)
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1.4, 15, param1=20, param2=5, minRadius=13,
                               maxRadius=15)  # will change the values

    center = find_center(contour)
    handle_circle = None

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda x: x[0])

        # filter out the circles that are to far away from the center (e.g. wheels)
        for circle in circles:
            if (abs(circle[0] - center[0]) <= 10 and abs(circle[1] - center[1]) <= 10):
                handle_circle = circle

    return handle_circle

def get_handle_coordinates(contour):
    handle_circle = get_handle_circle(contour)
    if handle_circle is not None:
        return (handle_circle[0], handle_circle[1])
    else:
        return (0,0) #TODO: fix this

def show_handle_circles(piece_contours):
    for i in range(len(piece_contours)):
        circle = get_handle_circle(piece_contours[i])
        # this is only printing the circles to an image.

        if circle is not None:
            cv2.circle(normal_img, (circle[0], circle[1]), circle[2], (0, 255, 0), 1)
            cv2.putText(normal_img, "({},{})".format(circle[0], circle[1]),
                        (int(circle[0] - 20), int(circle[1]- 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
    show(normal_img)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


#Classes
class PuzzlePiece:
    def __init__(self, contour, handle, match, angle = None):
        self.contour = contour
        self.handle = handle
        self.match = match #slotpiece
        self.angle = angle

class SlotPiece:
    def __init__(self, contour, match):
        self.contour = contour
        self.match = match #puzzlepiece

def init_pieces_and_slots(piece_contours, slot_contours):
    puzzlepieces = []
    slotpieces = []
    for i in range(len(piece_contours)):
        #initializing puzzlepiece
        match_contour = find_match(piece_contours[i], slot_contours)
        puzzlepiece = PuzzlePiece(piece_contours[i], get_handle_coordinates(piece_contours[i]), SlotPiece(match_contour, None))

        #initializing slotpiece
        slotpiece = puzzlepiece.match
        slotpiece.match = puzzlepiece
        puzzlepiece.match = slotpiece
        puzzlepiece.angle = get_piece_angle(puzzlepiece)
        adjust_angle(puzzlepiece)
        slotpieces.append(slotpiece)
        puzzlepieces.append(puzzlepiece)

    return puzzlepieces, slotpieces

def rotateImage(image, angle, image_center):
    #image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def rotate_piece(puzzlepiece, angle):
    mask = get_mask_image(puzzlepiece)
    processed_img = process_img(mask.copy(), 40)
    contours = get_contours_external(processed_img)
    center = find_center(contours[0])

    return rotateImage(mask, angle, center)

#crops the given piece and puts it into the middle of a mask
def get_mask_image(piece):
    croped = get_cropped_contour(piece)
    mask = np.zeros((512, 512, 3), np.uint8)
    x_offset = y_offset = 200

    mask[y_offset:y_offset + croped.shape[0], x_offset:x_offset + croped.shape[1]] = croped

    return mask

def get_base_edge(rect):
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
    distance = math.sqrt(math.pow((point2[0] - point1[0]),2) + math.pow((point2[1] - point1[1]),2))
    return distance

def get_slope(line):
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[1][0]
    y2 = line[1][1]

    if (x2 == x1):
        return ((y2 - y1) / 1)
    else:
        return ((y2 - y1) / (x2 - x1))

def draw_rect(img, rect):
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], -1, (255,255,0), 2)

def get_rect_contour(rect):
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def draw_edge(img, edge):
    cv2.line(img, edge[0], edge[1], (0,0,255), 3)

def get_angle(edge1, edge2):
    slope1 = get_slope(edge1)
    slope2 = get_slope(edge2)

    rad_angle = math.atan((slope1-slope2)/(1+(slope1*slope2)))
    degree_angle = (rad_angle * 180 / math.pi)
    return degree_angle

def get_piece_angle(puzzlepiece):
    edge1 = get_base_edge(get_rect(puzzlepiece.contour))
    edge2 = get_base_edge(get_rect(puzzlepiece.match.contour))
    angle = get_angle(edge1, edge2)
    return angle

def get_rect(contour):
    return cv2.minAreaRect(contour)


def process_croped(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred_img = cv2.medianBlur(gray_img, 5)
    # edged_img = cv2.Canny(blurred_img, 63, 180) # these parameters are important. The image detection behaves differently when changing the contrast.
    ret, threshold = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)
    # threshold = 255 - threshold  # invert the coloring
    return threshold

def overlay_contours(contour1, contour2):
    img = np.zeros((512, 512, 3), np.uint8)
    draw_contours(contour1, img)
    draw_contours(contour2, img)
    return img

def get_cropped_contour(piece):
    cut_piece_img = get_cut_contour(piece.contour)
    x, y, w, h = cv2.boundingRect(piece.contour)
    croped = cut_piece_img[y:y + h, x:x + w]
    return croped

def get_overlay_area(puzzlepiece, angle):
    rotated_img = rotate_piece(puzzlepiece, angle)
    processed = process_img(rotated_img, 40)
    processed = (255-processed)
    contours_puzzlepiece = get_contours_external(processed)
    draw_contours(contours_puzzlepiece, rotated_img)

    slot_img = get_mask_image(puzzlepiece.match)
    processed = process_croped(slot_img)
    contours_slotpiece = get_contours_external(processed)
    draw_contours(contours_slotpiece, slot_img)

    ## (2) Calc offset
    center_puzzle = find_center(contours_puzzlepiece[0])
    center_slot = find_center(contours_slotpiece[0])
    dx = center_puzzle[0] - center_slot[0]
    dy = center_puzzle[1] - center_slot[1]

    img = draw_overlaying_contours_to_mask([contours_puzzlepiece, contours_slotpiece], (dx,dy))
    img = process_img(img, 20)
    img = (255-img)
    overall_contour = get_contours_external(img)

    overall_img = draw_contours_to_mask([overall_contour])
    overall_contour_area = cv2.contourArea(overall_contour[0], True)

    return overall_contour_area

def draw_contours_to_mask(contours):
    img = np.zeros((512, 512, 3), np.uint8)
    for i in range(len(contours)):
        cv2.drawContours(img, contours[i], -1, (0,255,0), 2)

    return img

def draw_overlaying_contours_to_mask(contours, offset):
    assert len(contours) == 2
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.drawContours(img, contours[0], -1, (0, 255, 0), 2)
    cv2.drawContours(img, contours[1], -1, (0, 255, 0), 2, offset=offset)

    return img

def adjust_angle(puzzlepiece):
    area1 = get_overlay_area(puzzlepiece, puzzlepiece.angle)
    area2 = get_overlay_area(puzzlepiece, puzzlepiece.angle + 180)

    if (abs(area1) > abs(area2)):
        puzzlepiece.angle = puzzlepiece.angle + 180

def process_for_pieces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred_img = cv2.medianBlur(gray_img, 5)
    # edged_img = cv2.Canny(blurred_img, 63, 180) # these parameters are important. The image detection behaves differently when changing the contrast.
    ret, threshold = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
    threshold = 255 - threshold  # invert the coloring
    return threshold

def process_for_slots(img):
    cut_board_contour = find_board_contour(get_contours_ccomp(img))
    cut_board_img = get_cut_contour(cut_board_contour)
    cut_board_img = process_img(cut_board_img, 65)
    cut_board_img = black_background(cut_board_img, cut_board_contour)
    cv2.drawContours(cut_board_img, [cut_board_contour], -1, (0, 0, 0), 2)
    return cut_board_img

def process_img(img, threshold_value):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred_img = cv2.medianBlur(gray_img, 5)
    # edged_img = cv2.Canny(blurred_img, 63, 180) # these parameters are important. The image detection behaves differently when changing the contrast.
    ret, threshold = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    threshold = 255 - threshold  # invert the coloring
    return threshold

##Main

#Building images
normal_img = cv2.imread('images/image9.jpg')
gray_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
#blurred_img = cv2.medianBlur(gray_img, 5)
#edged_img = cv2.Canny(blurred_img, 63, 180) # these parameters are important. The image detection behaves differently when changing the contrast.
ret, threshold = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
threshold = 255-threshold #invert the coloring

slot_contours = get_slot_contours(process_for_pieces(normal_img))
piece_contours = get_piece_contours(process_for_pieces(normal_img))

matches = find_matchtes(slot_contours, piece_contours)

show_matches(matches)
show_handle_circles(piece_contours)

puzzlepieces, slotpieces = init_pieces_and_slots(piece_contours, slot_contours)

for i in range(len(puzzlepieces)):
     print ("angle:", puzzlepieces[i].angle)
     cv2.putText(normal_img, str(("angle:", int(puzzlepieces[i].angle))), find_center(puzzlepieces[i].contour), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

show(normal_img)

cv2.destroyAllWindows()


