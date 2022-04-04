"""
Exercise Set 5: Neural networks

Ossi Koski
"""

from PIL import Image

def main():
    get_data()

def get_data():
    im = Image.open('./GTSRB_subset_2/class1/000.jpg', 'r')
    pix_val = list(im.getdata())
    #print(pix_val.shape)
    print(type(pix_val[0]))

if __name__ == '__main__':
    main()
