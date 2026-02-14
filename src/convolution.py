import numpy as np

def convolve2d(image, kernel):
    k_height, k_width = kernel.shape
    pad_height, pad_width = k_height//2 , k_width//2

    #padding the image for corner condition
    padded_img = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    output = np.zeros(image.shape, dtype=np.float32)

    for y in range(k_height):
        for x in range(k_width):
            #we loop over kernel not image 
            image_slice = padded_img[y:y+image.shape[0], x:x+image.shape[1]]
            #multiply 
            output += image_slice*kernel[y, x]
    
    return output

