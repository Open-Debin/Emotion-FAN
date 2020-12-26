#coding=utf-8
import os,sys
import dlib
import cv2
import numpy as np
import pdb

if len(sys.argv) != 6:
    print(
        "Call this program like this:\n"
        "   ./face_alignment.py shape_predictor_5_face_landmarks.dat image_root_folder save_root_folder cnn_detector gpuID\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n")
    exit()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[5]

predictor_path = sys.argv[1]
image_root_path = sys.argv[2]
save_root_path = sys.argv[3]

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.argv[4])
sp = dlib.shape_predictor(predictor_path)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.gif',
]

def main():
    cnt = 0
    for root, dirs, files in os.walk(image_root_path,followlinks=False):
        for name in files:
            if is_image_file(name):
                if cnt%1000==0: print("{}".format(cnt) + "completed... ")
                cnt += 1
                img_path = os.path.join(root, name)
                save_path = img_path.replace(image_root_path, save_root_path)
                face = face_alignment(img_path)
                if face is not None:
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    cv2.imwrite(save_path, face)

    print("{}".format(cnt) + "completed...")
    
def face_alignment(face_file_path):
    # Load the image using OpenCV
    bgr_img = cv2.imread(face_file_path)
    if bgr_img is None:
        print("Sorry, we could not load '{}' as an image".format(face_file_path))
        exit()

    # Convert to RGB since dlib uses RGB images
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    ''' traditional method '''
    dets = detector(img, 1)
    if len(dets) == 0:
        # first use cnn detector
        dets = apply_cnn_detection(img)
        if len(dets) == 0:
            ''' Linear '''
            img = LinearEqual(img)
            dets = apply_cnn_detection(img)
            if len(dets) == 0:
                ''' clahe '''
                img = claheColor(img)
                dets = apply_cnn_detection(img)
                if len(dets) == 0:
                    #''' Histogram_equalization '''
                    img = hisEqulColor(img)
                    dets = apply_cnn_detection(img)
                    if len(dets) == 0:
                        return None

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()

    for detection in dets:
        faces.append(sp(img, detection))
    image = dlib.get_face_chip(img, faces[0], size=224, padding=0.25)
    cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return cv_bgr_img

def claheColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    channels = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    clahe.apply(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def LinearEqual(image):
    lut = np.zeros(256, dtype = image.dtype )
    hist= cv2.calcHist([image], [0], None, [256], [0.0,255.0])
    minBinNo, maxBinNo = 0, 255

    for binNo, binValue in enumerate(hist):
        if binValue != 0:
            minBinNo = binNo
            break
    for binNo, binValue in enumerate(reversed(hist)):
        if binValue != 0:
            maxBinNo = 255-binNo
            break
    for i,v in enumerate(lut):
        #print(i)
        if i < minBinNo:
            lut[i] = 0
        elif i > maxBinNo:
            lut[i] = 255
        else:
            lut[i] = int(255.0*(i-minBinNo)/(maxBinNo-minBinNo)+0.5)   # why plus 0.5
    return cv2.LUT(image, lut)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def apply_cnn_detection(img):
    cnn_dets = cnn_face_detector(img, 1)
    dets = dlib.rectangles()
    dets.extend([d.rect for d in cnn_dets])
    return dets

if __name__ == '__main__':
    main()
