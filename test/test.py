from phylanx import Phylanx
from cv2 import imread, IMREAD_COLOR, IMREAD_GRAYSCALE
from numpy import eye

@Phylanx(debug=True)
def eye_test():
    A = eye((10,10))
    print(A)

def imload_cv(img_path_str, color):
    A = imread(img_path_str, color)
    print(A)

@Phylanx(debug=True)
def imload_physl(img_path_str, color):
    A = imread(img_path_str, color)
    print(A)

if __name__ == "__main__":
    eye_test()

    import sys
    fstr = sys.argv[1]
    print(fstr)

    imload_cv(fstr, IMREAD_COLOR)
    imload_cv(fstr, IMREAD_GRAYSCALE)
    imload_physl(fstr, IMREAD_COLOR)
    imload_physl(fstr, IMREAD_GRAYSCALE)
