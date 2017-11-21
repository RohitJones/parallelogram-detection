import numpy as np
import cv2

from PIL import Image
from scipy.stats import norm
from numba import jit
from itertools import combinations


class Line:
    def __init__(self, x1, y1, x2, y2, rho=None, theta=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.rho = rho
        self.theta = theta
    
    def __repr__(self):
        return "Line(x1:{}, y1:{}, x2:{}, y2:{})".format(self.x1, self.y1, self.x2, self.y2)
    
    def __str__(self):
        return "Line(x1:{}, y1:{}, x2:{}, y2:{})".format(self.x1, self.y1, self.x2, self.y2)
    
    def extract(self):
        return [[self.x1, self.y1], [self.x2, self.y2]]
    
    @staticmethod
    def __intersection_point(line1, line2, _mode=None):
        uan = (line2.x2-line2.x1)*(line1.y1-line2.y1)-(line2.y2-line2.y1)*(line1.x1-line2.x1)
        ubn = (line1.x2-line1.x1)*(line1.y1-line2.y1)-(line1.y2-line1.y1)*(line1.x1-line2.x1)

        denom = (line2.y2-line2.y1)*(line1.x2-line1.x1)-(line2.x2-line2.x1)*(line1.y2-line1.y1)

        if _mode == 'intersect':
            # intersecting lines
            return not denom == 0
        
        elif _mode == 'parallel':
            # parallel lines
            return denom == 0
        elif _mode == 'coincident':
            # coincident lines:
            return denom == uan == ubn == 0

        if denom == 0:
            raise Exception("{} and {} are parallel !".format(line1, line2))

        ua = uan/denom
        ub = ubn/denom         

        x = line1.x1 + ua * (line1.x2 - line1.x1)
        y = line1.y1 + ua * (line1.y2 - line1.y1)

        return np.int(np.round(x)), np.int(np.round(y))
    
    # intersect is used to determine if two lines, intersect each other.
    @staticmethod
    def intersect(line1, line2):
        def _ccw(A,B,C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        A, B = line1.extract()
        C, D = line2.extract()

        return _ccw(A,C,D) != _ccw(B,C,D) and _ccw(A,B,C) != _ccw(A,B,D)
    
    # intersect_beta is another method used to determine if two lines intersect each other.
    @classmethod
    def intersect_beta(cls, line1, line2):
        return cls.__intersection_point(line1, line2, _mode='intersect')
    
    # parallel_beta is used to determine if two lines are parallel each other.
    @classmethod
    def parallel_beta(cls, line1, line2):
        return cls.__intersection_point(line1, line2, _mode='parallel')
    
    # co_incident_beta is used to determine if two lines are coincidental.
    @classmethod
    def co_incident_beta(cls, line1, line2):
        return cls.__intersection_point(line1, line2, _mode='coincident')

    # get_intersection_point is used to determine the point of intersection of two lines. 
    @classmethod
    def get_intersection_point(cls, line1, line2):
        return cls.__intersection_point(line1, line2)
    
    
class parallelogram:
    def __init__(self, pgp1, pgp2):
        x1, y1 = Line.get_intersection_point(pgp1[0], pgp2[0])
        x2, y2 = Line.get_intersection_point(pgp1[0], pgp2[1])
        x3, y3 = Line.get_intersection_point(pgp1[1], pgp2[0])
        x4, y4 = Line.get_intersection_point(pgp1[1], pgp2[1])

        self.co_ordinates = [(x1,y1), (x2,y2), (x3,y3), (x4, y4)]
        
        self.line1 = Line(x1, y1, x2, y2)
        self.line3 = Line(x3, y3, x4, y4)
        
        # determines if the line fromed by these points are the diagonals of the parallelogram
        if Line.intersect(Line(x1,y1,x3,y3), Line(x2,y2,x4,y4)):
            self.line2 = Line(x1, y1, x4, y4)
            self.line4 = Line(x2, y2, x3, y3)
            
        else:
            self.line2 = Line(x1,y1,x3,y3)
            self.line4 = Line(x2,y2,x4,y4)
            
    def extract(self):
        return [self.line1, self.line2, self.line3, self.line4]

    
@jit
def rgb2grey(image_array):
    height, width, _ = image_array.shape
    out_img = np.zeros(shape=(height, width), dtype=np.uint8)

    for x in range(height):
        for y in range(width):
            r, g, b = image_array[x][y]
            out_img[x][y] = np.uint8((0.3 * r) + (0.59 * g) + (0.11 * b))

    return out_img


# custom convolution operator. takes two arrays as input. the second array is considered to be the mask
@jit
def convolution(image_array, mask):
    mk_height, mk_width = mask.shape
    im_height, im_width = image_array.shape
    out_img = np.zeros(shape=image_array.shape)

    # offset is calculated to avoid the edges of the image. As a result, the size of the image decreases 
    _offset = mk_height // 2

    for x in range(_offset, im_height - _offset):
        for y in range(_offset, im_width - _offset):
            # calculates the start and end position of the sliding window on the image.
            imgx = x - mk_height // 2
            imgy = y - mk_width // 2
            
            # compute the sum of the element by element product of the mask and sliding window
            out_img[x][y] = np.sum(image_array[imgx: imgx + mk_height, imgy: imgy + mk_width] * mask)

    return out_img[_offset: -_offset, _offset: -_offset]


# custom gaussian filter. generates a gaussian mask of the specified size.
@jit
def gaussian_filter(image_array, size=11, sigma=3):
    interval = (2 * sigma + 1.) / size
    x = np.linspace(-sigma - interval / 2., sigma + interval / 2., size + 1)
    kern1d = np.diff(norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()

    return convolution(image_array, kernel)


# custom median filter. the size of the neighborhood can be varied by the 'size' paramater
@jit
def median_filter(img_array, mask_size=5):
    _offset = mask_size // 2
    height, width = img_array.shape
    out_img = np.zeros(shape=(height, width), dtype=np.uint8)

    for x in range(_offset, height - _offset):
        for y in range(_offset, width - _offset):
            img_x = x - _offset
            img_y = y - _offset
            # finds the median value in the sliding window formed
            out_img[x, y] = np.median(img_array[img_x: img_x + mask_size, img_y: img_y + mask_size])

    return out_img[_offset:-_offset, _offset:-_offset]


# method to compute the edge orientations by dividing the y_gradients by the x_gradients.
@jit
def get_edge_orientations(grad_y, grad_x):
    angles = np.zeros(shape=grad_y.shape, dtype=np.float64)

    for x in range(angles.shape[0]):
        for y in range(angles.shape[1]):
            # handles the potential divide by zero error if a particular x_gradient is 0
            if grad_x[x][y] == 0:
                angles[x][y] = 0 if grad_y[x][y] == 0 else 90
            else:
                angles[x][y] = np.rad2deg(np.arctan(grad_y[x][y] / grad_x[x][y]))

    return angles


# custom sobel operator. can perform thresholding or high pass filtering or neither.
@jit
def sobel(image_array, threshold=None, filter_mode=False):
    mask_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    mask_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    # compute gradient x and gradient y.
    gradient_x = convolution(image_array, mask_x)
    gradient_y = convolution(image_array, mask_y)

    # compute the edge orientations 
    angles = get_edge_orientations(gradient_y, gradient_x)

    # compute the gradient magnitude
    gradient = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # normalize the gradient magnitude
    gradient *= 255 / np.max(gradient)

    
    if threshold is not None:
        # high pass filter the gradient magnitude
        gradient[gradient < threshold] = 0
        
        if not filter_mode:
            # threshold the gradient magnitude
            gradient[gradient >= threshold] = 255

    return gradient, angles


# calculate the sector of an angle for non maxima supression
@jit
def _quant(angle):
    ang = angle % 180

    if 22.5 <= ang < 67.5:
        return 1
    elif 67.5 <= ang < 112.5:
        return 2
    elif 112.5 <= ang < 157.5:
        return 3
    else:
        return 0


# assign a sector to every single pixel in the edge
@jit
def quantize(angles):
    quants = np.zeros(shape=angles.shape, dtype=np.uint8)
    for x in range(quants.shape[0]):
        for y in range(quants.shape[1]):
            quants[x][y] = _quant(angles[x][y])

    return quants


# performs non maxima supression
@jit
def nms(grads, angles):
    quantized_angles = quantize(angles)
    supressed_gradients = grads.copy()
    
    # sets the boundary pixels to zero to avoid calculation out of bounds of image.
    supressed_gradients[:, 0] = 0
    supressed_gradients[:, -1] = 0
    supressed_gradients[0, :] = 0
    supressed_gradients[-1, :] = 0

    for x in range(1, supressed_gradients.shape[0] - 1):
        for y in range(1, supressed_gradients.shape[1] - 1):
            condition = [0, 0]
            if quantized_angles[x][y] == 0:
                condition[0] = grads[x][y - 1]
                condition[1] = grads[x][y + 1]

            elif quantized_angles[x][y] == 1:
                condition[0] = grads[x - 1][y + 1]
                condition[1] = grads[x + 1][y - 1]

            elif quantized_angles[x][y] == 2:
                condition[0] = grads[x + 1][y]
                condition[1] = grads[x - 1][y]

            else:
                condition[0] = grads[x + 1][y - 1]
                condition[1] = grads[x - 1][y + 1]

            if not condition[0] < grads[x][y] > condition[1]:
                supressed_gradients[x][y] = 0

    return supressed_gradients


# method to pad the boundary of an image with zeros. 
# used to restore an image to its original size after multiple convolution operations
@jit
def restore_size(image_array, target_shape):
    pad_x_size, pad_y_size = target_shape[0] - image_array.shape[0], target_shape[1] - image_array.shape[1]

    return np.pad(image_array, (pad_x_size // 2, pad_y_size // 2), 'constant', constant_values=0)


# performs hough transform and returns the accumulator array, rho_values and theta_values. 
@jit
def hough(img_array, angle_step=1):
    height, width = img_array.shape

    max_distance = np.int(np.ceil(np.sqrt(np.square(height) + np.square(width))))

    # compute all possible values of rho and theta for the given precision
    all_rhos = np.arange(-max_distance, max_distance)
    all_thetas = np.radians(np.arange(-90, 90, angle_step))

    # Initialize the accumulator array
    accumulator = np.zeros((all_rhos.size, all_thetas.size), dtype=np.uint64)

    # gives the indices of points with value > 0, that is, the edge points
    edges = np.where(img_array > 0)

    for x, y in zip(edges[0], edges[1]):
        for index, theta in enumerate(all_thetas):
            rho = np.int(np.round(y * np.cos(theta) + x * np.sin(theta))) + max_distance
            accumulator[rho][index] += 1

    return accumulator, all_rhos, all_thetas


# finds the cartesian product of a set with itself. 
# used to generate the indices of the neighborhood pixels
@jit
def _cartesian_product(x):
    return np.dstack(np.meshgrid(x, x)).reshape(-1, 2)


# finds the peaks present in the accumulator array.
# Can change the size of the comparison neighborhood for higher accuracy.
@jit
def hough_peaks(input_array, neighborhoodsize=1):
    output = input_array.copy()
    # find the indices of all points with value > 0.
    peaks = np.where(input_array > 0)
    # compute neighbours indicies for the given neighborhood size.
    neighbours = _cartesian_product(np.arange(-neighborhoodsize, neighborhoodsize + 1))

    for x, y in zip(peaks[0], peaks[1]):
        # array used to store the value of the accumulator at each point of the neighborhood
        max_pixel = np.zeros(neighbours.shape[0])
        for index, xy in enumerate(neighbours):
            if x + xy[0] < input_array.shape[0] and y + xy[1] < input_array.shape[1]:
                max_pixel[index] = input_array[x + xy[0]][y + xy[1]]

        # checks if value of current cell of accumulator array is equivalent
        # to the maximal value in the current neighborhood
        if input_array[x][y] != np.max(max_pixel):
            output[x][y] = 0

    return output


# method used to draw a line on an image array. 
# can accept either a line object or two points.
# variable thickness
def draw_line(src_img, line=None, pt1=None, pt2=None, thickness=2):
    img = src_img.copy()
    if line: 
        cv2.line(img, (line.x1, line.y1), (line.x2, line.y2), (255,0,0), thickness)
    else:
        cv2.line(img, pt1, pt2, (255,0,0), thickness)
    return img


# method used to compute the co-ordinates of the end points of the lines detected by hough transform.
@jit
def retrieve_lines(accumulator_array, rhos, thetas):
    peaks_locations = np.where(accumulator_array > 0)

    x_peaks, y_peaks = peaks_locations

    lines = []
    for index, xy in enumerate(zip(x_peaks, y_peaks)):
        x, y = xy
        a = np.cos(thetas[y])
        b = np.sin(thetas[y])
        x0 = a * rhos[x]
        y0 = b * rhos[x]
        x1 = int(x0 - 1000 * b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 + 1000 * b)
        y2 = int(y0 - 1000 * a)
    
        lines.append(Line(x1, y1, x2, y2, rhos[x], thetas[y]))
    
    return lines


# method used to divide the input lines into mutually parallel groups
def make_lines_pairs(input_lines, leniency=5, merge=False ,_flag=False):
    main = []
    if merge and _flag:
        angles = np.abs(np.array([np.degrees(each.theta) for each in input_lines]))
    else:
        angles = np.array([np.degrees(each.theta) for each in input_lines])
        
    for index in range(angles.size):
        T = np.abs(angles - angles[index])
        indicies = np.where(T <= leniency)
        if indicies[0].size > 1:
            main.append(indicies[0])
    
    # Merge groups in list main containing common items
    iset = set([frozenset(s) for s in main])
    result = []
    while(iset):
        nset = set(iset.pop())
        check = len(iset)
        while check:
            check = False
            for s in iset.copy():
                if nset.intersection(s):
                    check = True
                    iset.remove(s)
                    nset.update(s)
        result.append(sorted(list(nset)))
    
    result.sort(key= lambda x:x[0])
    
    # Convert Indices stored in list main into line objects
    main = []
    for each_group in result:
        temp = []
        for each_line_index in each_group:
            temp.append(input_lines[each_line_index])
        main.append(temp)
  
    # merge positive and negative angles, within leniency range, into groups
    if merge and not _flag:
        temp = []
        for each in main:
            temp += each
    
        main = make_lines_pairs(temp, leniency, True, True)
        
    return main


# main method of the program. 
# Takes a dictionay object an input containing all the information like file names and threshold values.
def detect_parallelograms(PARAMETERS, DEBUG=False):
    # open the image.
    test_img = Image.open(PARAMETERS['name'])
    # convert image into an array
    test_img_array = np.asarray(test_img)
    # convert color image array into a greyscale image array
    test_img_greyscale_array = rgb2grey(test_img_array)

    # determine the order of the filters to be applied
    if PARAMETERS['filter_order'] == 'gaussian then median':
        # perform gaussian filtering first
        filtered_img_array = gaussian_filter(test_img_greyscale_array, PARAMETERS['gaussian_filter_size'])
        # perform median filtering second
        filtered_img_array = median_filter(filtered_img_array, PARAMETERS['median_filter_size'])
    
    else:
        # perform median filtering first
        filtered_img_array = median_filter(test_img_greyscale_array, PARAMETERS['median_filter_size'])
        # perform gaussian filtering second
        filtered_img_array = gaussian_filter(filtered_img_array, PARAMETERS['gaussian_filter_size'])

    # compute the edge gradient magnitudes and edge gradient angles by sobel operator.
    test_img_edges, test_img_edge_angles = sobel(filtered_img_array, PARAMETERS['sobel_threshold'], filter_mode=True)
    # supress the thick edges obtained by soble's operator
    test_img_edges_suppressed = nms(test_img_edges, test_img_edge_angles)
    # pad the boundary of the suppressed edges to match the original size of the input image
    test_img_restored = restore_size(test_img_edges_suppressed, test_img_greyscale_array.shape)

    # perform hough transform.
    accumulator, rho_values, theta_values = hough(test_img_restored)
    # threshold the hough accumulator array
    accumulator[accumulator < PARAMETERS['hough_threshold']] = 0
    # find the peaks present in the accumulator array
    accumulator_peaks = hough_peaks(accumulator, PARAMETERS['hough_peaks_neighborhood_size'])

    # find all lines detected by hough transform. Is a list containing several Line objects.
    all_lines = retrieve_lines(accumulator_peaks, rho_values, theta_values)
    # divide all the lines into smaller groups. ideally 2 groups
    parallel_groups = make_lines_pairs(all_lines, leniency=PARAMETERS['parallel_lines_leniency'], merge=True)

    group_1 = parallel_groups[0]
    group_2 = parallel_groups[1]
    # find every possbile combination of picking two lines in each group
    all_pairs_group_1 = list(combinations(group_1, 2))
    all_pairs_group_2 = list(combinations(group_2, 2))

    # finding every possbile parallelogram that can be formed by by picking 2 lines from each group
    all_possible_parallelograms = []
    for each_group_1_pair in all_pairs_group_1:
        for each_group_2_pair in all_pairs_group_2:
            all_possible_parallelograms.append(parallelogram(each_group_1_pair, each_group_2_pair))

    weighted_parallelograms = []
    # copy of unsuppressed edge gradient
    raw_edge_map = restore_size(test_img_edges, test_img_greyscale_array.shape)

    for each_parallelogram in all_possible_parallelograms:
        # generate a blank image array of the same size as the input image
        blank_img = np.zeros(shape=test_img_restored.shape)

        # draw the parallelogram on the blank image array
        for each_line in each_parallelogram.extract():
            blank_img = draw_line(blank_img, each_line, thickness=1)

        # count the number of pixels / cells in the blank image array that is nonzero
        parallelogram_perimeter = np.count_nonzero(blank_img)
        # perform logical and of the blank image array and the unsupressed edge gradient
        coincidence_value = np.count_nonzero(np.logical_and(blank_img, raw_edge_map))

        # avoid a divide by zero error
        if parallelogram_perimeter == 0:
            continue

        # compute the percentage of pixels of the parallelogram that lies on the detected edges of the image.
        coincidence_percentage = coincidence_value / parallelogram_perimeter * 100

        # threshold the parallelogram on the basis of the length of the perimeter and the coincidence percentage
        if parallelogram_perimeter >= PARAMETERS['parallelogram_perimeter_threshold'] and coincidence_percentage > PARAMETERS['coincidence_percentage_threshold']:
            weighted_parallelograms.append((np.round(coincidence_percentage), each_parallelogram, parallelogram_perimeter))

    # sort the parallelograms that passed though the previous steps. 
    # first by coincidence percentage then by lenght of perimeter.
    weighted_parallelograms.sort(key= lambda x: (x[0], x[2]), reverse=True)

    # remove any duplicate parallelograms that may have been detected.
    non_duplicates = []
    pllgm_list = []
    for each in weighted_parallelograms:
        if each[2] not in non_duplicates:
            non_duplicates.append(each[2])
            pllgm_list.append(each)
    weighted_parallelograms = pllgm_list

    # draw all parallelograms that passed through the previous step, on top of the original image. 
    output = test_img_array.copy()
    for index in range(len(weighted_parallelograms)):
        for each_line in weighted_parallelograms[index][1].extract():
            output = draw_line(output, each_line)

    # save the image with the drawn parallelograms in the same directory if debug mode is not active.
    if not DEBUG:
        Image.fromarray(output).save(PARAMETERS['name'][:-4]+"_detected_parallelograms.jpg")
    
    # debug mode is used to save all the images that are obtained at every stage of the program.
    if DEBUG:
        from os import mkdir, chdir
        try:
            mkdir(PARAMETERS['name'][:-4] + "_files")
        except FileExistsError:
            pass
        chdir(PARAMETERS['name'][:-4] + "_files")
        
        Image.fromarray(test_img_array).save("1_" + PARAMETERS['name'][:-4] + "_original.jpg")
        Image.fromarray(test_img_greyscale_array).convert('RGB').save("2_" + PARAMETERS['name'][:-4] + "_greyscale.jpg")
        Image.fromarray(filtered_img_array).convert('RGB').save("3_" + PARAMETERS['name'][:-4] + "_filtered.jpg")
        tresholded_edges, _ = sobel(filtered_img_array, PARAMETERS['sobel_threshold'])
        Image.fromarray(tresholded_edges).convert('RGB').save("4_" + PARAMETERS['name'][:-4] + "_edge_magnitude.jpg")
        temp = test_img_edges_suppressed.copy()
        temp[temp > 0] = 255
        Image.fromarray(temp).convert('RGB').save("5_" + PARAMETERS['name'][:-4] + "_edge_magnitude_suppressed.jpg")
        temp = test_img_array.copy()
        for each_line in all_lines:
            temp = draw_line(temp, each_line)
        Image.fromarray(temp).save("6_" + PARAMETERS['name'][:-4] + "_all_detected_lines.jpg")
        Image.fromarray(output).save("7_" + PARAMETERS['name'][:-4]+"_detected_parallelograms.jpg")
        with open(PARAMETERS['name'][:-4] + "_parallelogram_co-ordinates.txt", 'w') as file:
            for index, each_parallelogram in enumerate(weighted_parallelograms):
                file.write("Parallelogram #{} Co-ordinates: {}\n".format(index + 1, each_parallelogram[1].co_ordinates))
        chdir("..")


if __name__ == '__main__':
	test_image_1_parameters = {
	    'name': 'TestImage1c.jpg',
	    'filter_order': 'median then gaussian',
	    'median_filter_size': 11,
	    'gaussian_filter_size': 11,
	    'sobel_threshold': 160,
	    'hough_threshold': 90,
	    'hough_peaks_neighborhood_size': 11,
	    'parallel_lines_leniency': 5,
	    'parallelogram_perimeter_threshold': 203,
	    'coincidence_percentage_threshold': 88.0
	}

	test_image_2_parameters = {
	    'name': 'TestImage2c.jpg',
	    'filter_order': 'gaussian then median',
	    'median_filter_size': 15,
	    'gaussian_filter_size': 9,
	    'sobel_threshold': 18,
	    'hough_threshold': 40,
	    'hough_peaks_neighborhood_size': 39,
	    'parallel_lines_leniency': 15,
	    'parallelogram_perimeter_threshold': 203,
	    'coincidence_percentage_threshold': 73.0
	}

	test_image_3_parameters = {
	    'name': 'TestImage3.jpg',
	    'filter_order': 'gaussian then median',
	    'median_filter_size': 5,
	    'gaussian_filter_size': 5,
	    'sobel_threshold': 20,
	    'hough_threshold': 63,
	    'hough_peaks_neighborhood_size': 9,
	    'parallel_lines_leniency': 5,
	    'parallelogram_perimeter_threshold': 203,
	    'coincidence_percentage_threshold': 93.25
	}


	detect_parallelograms(test_image_1_parameters)
	detect_parallelograms(test_image_2_parameters)
	detect_parallelograms(test_image_3_parameters)

	# detect_parallelograms(test_image_1_parameters, DEBUG=True)
	# detect_parallelograms(test_image_2_parameters, DEBUG=True)
	# detect_parallelograms(test_image_3_parameters, DEBUG=True)