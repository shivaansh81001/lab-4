# import statements
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
from skimage.measure import find_contours
from A4_submission import getRegionalMinima, getSmallestNeighborIndex, iterativeMinFollowing, part1


# ======== DO NOT EDIT imreconstruct =========
def imreconstruct(marker, mask):
    curr_marker = (np.copy(marker)).astype(mask.dtype)
    kernel = np.ones([3, 3])
    while True:
        next_marker = cv2.dilate(curr_marker, kernel, iterations=1)
        intersection = next_marker > mask
        next_marker[intersection] = mask[intersection]
        if np.array_equal(next_marker, curr_marker):
            return curr_marker
        curr_marker = np.copy(next_marker)
    return curr_marker


# ======== DO NOT EDIT imimposemin =========
def imimposemin(marker, mask):
    # adapted from its namesake in MATLAB
    fm = np.copy(mask)
    fm[marker] = -np.inf
    fm[np.invert(marker)] = np.inf
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        range = float(np.max(mask) - np.min(mask))
        if range == 0:
            h = 0.1
        else:
            h = range * 0.001
    else:
        # Add 1 to integer images.
        h = 1
    fp1 = mask + h
    g = np.minimum(fp1, fm)
    
    return np.invert(imreconstruct(
        np.invert(fm.astype(np.uint8)), np.invert(g.astype(np.uint8))
    ).astype(np.uint8))

def part2(final):

    # Implement Minimum Following watershed segmentation to find the boundaries of the cells we detected in part 1.


    # TO - DO: Finish functions is_4connected(), getRegionalMinima() and iterativeMinFollowing() in A4_submission.py

    
    # =============== DO NOT EDIT ANY OF THE CODE BELOW ===============

    #Test if the functions are working as expected
    test_image = np.loadtxt('A4_test_image.txt')
    print("Test image:")
    print(test_image)

    #Testing getSmallestNeighborIndex. 
    print("\nTesting function getSmallestNeighborIndex...")
    [min_row, min_col] = getSmallestNeighborIndex(test_image, 0, 0)
    print(f"Location of the smallest 4-connected neighbour of pixel at location [0,0] with intensity value {test_image[0,0]}: {[min_row, min_col]} with value {test_image[min_row, min_col]}")

    print("\nTesting function getRegionalMinima...")
    markers = getRegionalMinima(test_image)
    print("markers:")
    print(markers)

    print("\nTesting function iterativeMinFollowing...")
    labels = iterativeMinFollowing(test_image, markers)
    print("Final labels:")
    print(labels)


    # Image reconstruct and draw their boundaries ()

    sigma = 2.5
    img_name = 'nuclei.png'
    img_gs = io.imread(img_name).astype(np.float32)

    img_blurred = cv2.GaussianBlur(img_gs, (int(2 * round(3 * sigma) + 1), int(2 * round(3 * sigma) + 1)), sigma
                         )#borderType=cv2.BORDER_REPLICATE

    [img_grad_y, img_grad_x] = np.gradient(img_blurred)
    img_grad = np.square(img_grad_x) + np.square(img_grad_y)

    # refined blob center locations generated generated in part 1
    blob_markers = final

    img_grad_min_imposed = imimposemin(blob_markers, img_grad)

    markers = getRegionalMinima(img_grad_min_imposed)
    plt.figure(0)
    plt.imshow(markers,cmap='jet')
    plt.title('markers')
    plt.show()

    labels = iterativeMinFollowing(img_grad_min_imposed, np.copy(markers))
    plt.figure(1)
    plt.imshow(labels,cmap='jet')
    plt.title('labels')
    plt.show()

    contours = find_contours(img_grad_min_imposed , 0.8)

    fig,ax=plt.subplots()
    ax.imshow(img_grad_min_imposed, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

if __name__  == "__main__":
    final = part1()
    part2(final)