import cv2 
import numpy as np
from filter_tags import greyscale, gaussian_smoothing

def split_into_smaller_grid(warped_tags):
    gray_warped_tags = greyscale(warped_tags)
    _, binary = cv2.threshold(gray_warped_tags, 127, 255, cv2.THRESH_BINARY)

    grid = np.zeros((8,8), dtype = int)
    step = 20 #our window is 160 pixels and we want the 8x8 window, so 160/8 = 20

    #loop over the 8x8 grid to extract it
    for i in range(8):
        for j in range(8):
            y = i*step +(step//2)
            x = j*step +(step //2)

            if binary[y, x] > 127:
                grid[i,j] = 1
            else:
                grid[i,j] = 0

    return grid

def find_tag_id(grid):
    n_orientation = 0
    found_orientation = False

    for r in range(4):
        top_right = grid[2, 5]
        top_left = grid[2, 2]
        bottom_right = grid[5, 5]
        bottom_left = grid[5, 2]

        if bottom_right == 1 and (top_left+bottom_left+top_right == 0):
            found_orientation = True
            n_rotation = r
            break
        grid = np.rot90(grid) #rotate the grid if it is not upright

    if not found_orientation:
        return None, 0
    
    #reading inner 2x2 matrix for Ids 
    bit1 = grid[3, 3] 
    bit2 = grid[3, 4] 
    bit3 = grid[4, 4] 
    bit4 = grid[4, 3]

    tag_id = (bit1 * 1) + (bit2 * 2) + (bit3 * 4) + (bit4 * 8)
    return tag_id, n_rotation

    

