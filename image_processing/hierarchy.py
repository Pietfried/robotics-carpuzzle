import cv2
import numpy as np
import imutils

#Building images
normal_img = cv2.imread('images/image.jpg')
gray_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.medianBlur(gray_img, 5)
edged_img = cv2.Canny(blurred_img, 90, 300)

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
        if (cv2.arcLength(contour, True) > 2100):
            save = idx
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
    _ ,contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def get_contours_ccomp(img):
    _, contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
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

def get_piece_contours(img):
    piece_contours = []
    all_contours = get_contours_external(img)
    board_area = get_board_area(all_contours[find_board_contour_idx(all_contours)])
    for contour in all_contours:
        if (cv2.contourArea(contour) > 10000 and cv2.contourArea(contour) < board_area and cv2.contourArea(contour) > 0):
            piece_contours.append(contour)
    return piece_contours

def get_slot_contours(img):
    slot_contours = []
    cut_board_img = get_cut_board_img(img, get_contours_ccomp(edged_img))
    contours = get_contours_external(cut_board_img)
    for contour in contours:
        if (cv2.contourArea(contour) > 10000 and cv2.contourArea(contour) > 0):
            slot_contours.append(contour)
    return slot_contours

def get_areas(contours):
    areas = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))
    areas.sort()
    return areas

def get_perimeters(contours):
    perimiters = []
    for contour in contours:
        perimiters.append(cv2.arcLength(contour, True))
    perimiters.sort()
    return perimiters

def show_matches(piece_contours, slot_contours):
    for i in range(len(piece_contours)):
        cv2.drawContours(normal_img, piece_contours, i, (0, 255, 0), 2)
        cv2.drawContours(normal_img, slot_contours, i, (0, 255, 0), 2)
        cv2.imshow("Matches", normal_img)
        cv2.waitKey(0)

        piece_contours.so

slot_contours = get_slot_contours(edged_img)
piece_contours = get_piece_contours(edged_img)

slot_perimiters = get_perimeters(slot_contours)
piece_perimiters = get_perimeters(piece_contours)

print("slot", get_areas(slot_contours))
print("piece", get_areas(piece_contours))

show_matches(piece_contours, slot_contours)
cv2.destroyAllWindows()