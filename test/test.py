from phylanx import Phylanx
from cv2 import imread, IMREAD_COLOR, IMREAD_GRAYSCALE

def imload_cv(img_path_str):
    A = imread(img_path_str, IMREAD_COLOR)
    B = imread(img_path_str, IMREAD_GRAYSCALE)
    print('color')
    print(A)
    print('grayscale')
    print(B)

@Phylanx
def imload_physl(img_path_str):
    A = imread(img_path_str, IMREAD_COLOR)
    B = imread(img_path_str, IMREAD_GRAYSCALE)
    print('color')
    print(A)
    print('grayscale')
    print(B)

if __name__ == "__main__":
    import sys
    fstr = sys.argv[1]
    print(fstr)
    imload_cv(fstr)
    imload_physl(fstr)
