# import statements
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, img_as_ubyte

def part1():
    
    imfile = 'nuclei.png'
    I = io.imread(imfile)

    # Apply Gaussian filter
    sig = 2.5

    J = filters.gaussian(I, sig)

    # Plotting Input image and Blurred image
    plt.figure(figsize=(10,8))
    plt.subplot(121),plt.imshow(I,cmap='jet'), plt.title('Input Image')
    plt.subplot(122),plt.imshow(J,cmap='jet'),plt.title('Blurred Image')
    plt.show()


    # =========== 1. Create DoG volume ===========
    # In the list 'sigmas', 4 values of sigmas (in order) have been provided. You have to use these to create 3 DoG levels. 
    # Level 1 --> gaussian(J, first sigma value) - gaussian(J, second sigma value)
    # Level 2 --> gaussian(J,second sigma value) - gaussian(J, third sigma value) 
    # Level 3 ---> gaussian(J,third sigma value) - gaussian(J, fourth sigma value) 

    # NOTE: You can use filters.gaussian. You CANNOT use skimage.filters.difference_of_gaussians.
    # Each level should be saved in the corresponding index of variable DoG


    sigmas = [1.6 ** i for i in range(1,5)]
    [h, w] = J.shape
    DoG = np.zeros([h, w, 3])

    #TODO: Create DoG levels

    
    for i in range(3):
        DoG[:,:,i] = filters.gaussian(J,sigma = sigmas[i]) - filters.gaussian(J,sigma = sigmas[i+1])

    print(DoG)

    level1 = DoG[:, :, 0]
    level2 = DoG[:, :, 1]
    level3 = DoG[:, :, 2]

    # Plotting
    plt.figure(figsize=(10,8))
    plt.subplot(131), plt.imshow(level1,cmap='jet'), plt.title('Level 1')
    plt.subplot(132), plt.imshow(level2,cmap='jet'), plt.title('Level 2')
    plt.subplot(133), plt.imshow(level3,cmap='jet'), plt.title('Level 3')
    plt.show()

    # =========== 2. Obtain a rough estimate of blob center locations ===========
    scatter_size = 40
    scatter_col = 'r'

    # TODO: Detect regional minima within the DoG volume. You can check out scipy.ndimage.filters.minimum_filter. 

    local_minima = ...


    # Plotting
    plt.figure(figsize=(10,8))
    plt.subplot(131), plt.imshow(local_minima[..., 0],cmap='jet')
    plt.subplot(132), plt.imshow(local_minima[..., 1],cmap='jet')
    plt.subplot(133), plt.imshow(local_minima[..., 2],cmap='jet')
    plt.show()

    # TODO: Convert local_minima to a binary image A (Check the stackoverflow discussion linked on e-class for reference)

    A = ...


    # TODO: Collapse this 3D binary image into a single channel image and assign to variable B (Check out np.sum)

    B = ...


    # TODO: Show the locations of all non-zero entries in this collapsed array overlaid on the input image as red points.

    # Check out np.nonzero()
    [y,x] = ...


    # Plotting
    plt.figure(figsize=(10,8))
    plt.imshow(I,cmap='jet')
    plt.scatter(x, y, marker='.', color=scatter_col, s=scatter_size)
    plt.xlim([0, I.shape[1]])
    plt.ylim([0, I.shape[0]])
    plt.title('Rough blob centers detected in the image')
    plt.show()


    # =========== 3. Refine the blob centers using Li thresholding ===========

    # Apply Gaussian filtering using skimage.filters.gaussian with a suitably chosen sigma and convert to unit8
    J = img_as_ubyte(J)

    # TODO: Apply Li thresholding on the blurred image using filters.threshold_li to obtain the optimal threshold for this image
    threshold = ...

    # TODO: Remove all minima in the output image (B) of "Obtain a rough estimate of blob locations" (Part 1, q2) where pixel values 
    #          are less than the obtained threshold. Assign this output to variable final

    final = ...


    # TODO: Show the remaining minima locations overlaid on the input image as red points. Once again, you can use np.nonzero()
    [y, x] = ...


    # Plotting
    plt.figure(figsize=(10,8))
    plt.imshow(I,cmap='jet')
    plt.scatter(x, y, marker='.', color=scatter_col, s=scatter_size)
    plt.xlim([0, I.shape[1]])
    plt.ylim([0, I.shape[0]])
    plt.title('Refined blob centers detected in the image')
    plt.show()


    return final

# ----------------------------- PART 2 -----------------------------

#You do not need to modify this function
def getSmallestNeighborIndex(img, row, col):
    """
    Parameters : 
    img            - image
    row            - row index of pixel
    col            - col index of pixel
    
    Returns         :  The location of the smallest 4-connected neighbour of pixel at location [row,col]

    """

    min_row_id = -1
    min_col_id = -1
    min_val = np.inf
    h, w = img.shape
    for row_id in range(row - 1, row + 2):
        if row_id < 0 or row_id >= h:
            continue
        for col_id in range(col - 1, col + 2):
            if col_id < 0 or col_id >= w:
                continue
            if row_id == row and col_id == col:
                continue
            if is_4connected(row, col, row_id, col_id):
              if img[row_id, col_id] < min_val:
                  min_row_id = row_id
                  min_col_id = col_id
                  min_val = img[row_id, col_id]     
    return min_row_id, min_col_id


# TODO: Complete the function is_4connected
def is_4connected(row, col, row_id, col_id):

    """
    Parameters : 
    row           - row index of pixel
    col           - col index of pixel
    row_id        - row index of neighbour pixel 
    col_id        - col index of neighbour pixel 
    
    Return         :  Boolean. Whether pixel at location [row_id, col_id] is a 4 connected neighbour of pixel at location [row, col]

    """

    pass


# TODO: Complete the function getRegionalMinima
def getRegionalMinima(img):
    markers = np.zeros(img.shape, dtype=np.int32)
    h, w = img.shape
    
    #Your code here


    return markers

# TO - DO: Complete the function iterativeMinFollowing
def iterativeMinFollowing(img, markers):

    """
    Parameters : 
    img          - image
    markers      - returned from function getRegionalMinima(img)

    
    Returns       :  final labels (markers_copy)

    """
    markers_copy = np.copy(markers)
    h, w = img.shape
    
    # i here is for printing iteration
    #i=1
    
    while True:

        #Number of pixels unmarked (label value is still 0)
        n_unmarked_pix = 0
        
        for row in range(h):
            for col in range(w):
                
                #Your code here

                pass
        
        
        # NOTE!!: Please make sure to comment the below two print statements and i+=1 before submitting. 
        #Feel free to un-comment them while working on the assignment and observing how iterativeMinFollowing works
        #print(f"labels after iteration {i}:")
        #print(markers_copy)
        #i+=1
        
        print('n_unmarked_pix: ', n_unmarked_pix)
        
    return markers_copy