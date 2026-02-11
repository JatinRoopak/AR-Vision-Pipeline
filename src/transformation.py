import numpy as np

def homography(src_points, dst_points):
    if len(src_points) < 4 : return None

    A = []

    for i in range(len(src_points)):
        x, y = src_points[i][0], src_points[i][1]
        u, v = dst_points[i][0], dst_points[i][1]

        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])

    A = np.array(A)

    U, S, Vh = np.linalg.svd(A) #Solving using SVD method
    L = Vh[-1]

    H = L.reshape(3, 3)

    if H[2, 2] != 0:
        H = H / H[2, 2]

    return H

def bilinear_interpolation(image, x, y):
    h, w = image.shape[:2]

    x0 = int(np.floor(x))
    x1 = min(x0+1, w-1)
    y0 = int(np.floor(y))
    y1 = min(y0+1, h-1)

    dx = x-x0
    dy = y-y0

    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]

    top = (1-dx)*Ia + dx*Ib
    bottom = (1-dx)*Ic + dx*Id

    pixel = (1-dy)*top + dy*bottom

    return pixel.astype(np.uint8)


def warp_perspective(image, H, output_shape):
    height, width = output_shape
    warped_img = np.zeros((height, width, 3), dtype=np.uint8)

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        print("error: H is singular and cannot be inverted.")
        return warped_img
    
    #we loop over dsestination pixels and apply backwardwarping otherwise holes may come in the warp image
    for y in range(height):
        for x in range(width):
            dest_pt = np.array([x, y, 1])
            src_pt = H_inv @ dest_pt
            if src_pt[2] != 0:
                src_x = src_pt[0]/src_pt[2]
                src_y = src_pt[1]/src_pt[2]

                if x == 80 and y == 80:
                    print(f"Center Pixel (80,80) maps to Camera Coordinate: ({src_x:.2f}, {src_y:.2f})")
                    print(f"Camera Image Size: {image.shape[1]} x {image.shape[0]}")

                #checking boundary
                if 0<= src_x <image.shape[1]-1 and 0 <=src_y <image.shape[0]-1:
                    warped_img[y, x] = bilinear_interpolation(image, src_x, src_y)
    
    return warped_img

