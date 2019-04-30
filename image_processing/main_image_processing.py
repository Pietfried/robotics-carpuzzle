import cv2
import numpy as np
import imutils

#Building images
normal_img = cv2.imread('images/image5.jpg')
gray_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.medianBlur(gray_img, 5)
edged_img = cv2.Canny(blurred_img, 56, 180) # these parameters are important. The image detection behaves differently when changing the contrast.

ret, thresh = cv2.threshold(gray_img, 65, 255, cv2.THRESH_BINARY)
thresh = 255-thresh
edged_img = thresh

contours, hierarchy = cv2.findContours(edged_img, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

def show(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)

show(thresh)

def show_images():
        cv2.imshow("normal_img", normal_img)
        cv2.waitKey(0)
        cv2.imshow("edged_img", edged_img)
        cv2.waitKey(0)

def show_contours_onebyone():
        i = 0
        for contour in contours:
                cv2.drawContours(normal_img, [contour], -1, (0, 255, 0), 2)
                print("contour number:", i)
                print(hierarchy[0][i])
                cv2.imshow("Contour image", normal_img)
                cv2.waitKey(0)
                i = i + 1

def find_board_contour(contours):
        board_contours = []
        i = 0
        print(len(contours))
        for contour in contours:
                i = i+1
                print(i)
                if (cv2.arcLength(contour, True) > 200):
                        print("Contour found. Contour is", contour)
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
            print("found")
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
    img = cv2.imread("zebra.jpg")
    stencil = np.zeros(edged_img.shape).astype(edged_img.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [contour], color)
    result = cv2.bitwise_and(edged_img, stencil)
    return result

def get_piece_contours(img):
    img = (img-255)
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
    show(cut_board_img)
    contours = get_contours_external(cut_board_img)
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


slot_contours = get_slot_contours(edged_img)
piece_contours = get_piece_contours(edged_img)

matches = find_matchtes(slot_contours, piece_contours)

#show_contours_onebyone(slot_contours)
#show_contours_onebyone(piece_contours)

show_matches(matches)

cv2.destroyAllWindows()
