'''
    * Copyright (C) 2011-2020 Doubango Telecom <https://www.doubango.org>
    * File author: Mamadou DIOP (Doubango Telecom, France).
    * License: For non commercial use only.
    * Source code: https://github.com/DoubangoTelecom/ultimateALPR-SDK
    * WebSite: https://www.doubango.org/webapps/alpr/


    https://github.com/DoubangoTelecom/ultimateALPR/blob/master/SDK_dist/samples/c++/recognizer/README.md
	Usage: 
		recognizer.py \
			--image <path-to-image-with-plate-to-recognize> \
			[--assets <path-to-assets-folder>] \
            [--charset <recognition-charset:latin/korean/chinese>] \
			[--tokenfile <path-to-license-token-file>] \
			[--tokendata <base64-license-token-data>]
	Example:
		recognizer.py \
			--image C:/Projects/GitHub/ultimate/ultimateALPR/SDK_dist/assets/images/lic_us_1280x720.jpg \
            --charset "latin" \
			--assets C:/Projects/GitHub/ultimate/ultimateALPR/SDK_dist/assets \
			--tokenfile C:/Projects/GitHub/ultimate/ultimateALPR/SDK_dev/tokens/windows-iMac.lic
'''
import time
toc = time.time()

import ultimateAlprSdk
import sys
import argparse
import json
import platform
import os.path
from PIL import Image, ExifTags
import numpy as np
import math
from matplotlib import pyplot as plt 

import json
import cv2 as cv

# import call method from subprocess module
from subprocess import call
  
# import sleep to show output for some time period
from time import sleep

#from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

import matplotlib.pyplot as plt
import argparse


def getAreaOfBox(x1, y1, x2, y2):
    a = x2-x1
    b = y2-y1
    area = a*b
    return area

def getSegSizeValues(areasList):
    mean = np.mean(areasList)
    limit1 = mean - mean/4
    limit2 = mean + mean/4
    return limit1, limit2

def getSegBySize(area, limit1, limit2):
    if area <=limit1:
        return "Small"
    elif area>limit1 and area<limit2:
        return "Medium"
    else:
        return "Large" 

def getSegByState(numberplate):
    state_code = numberplate[:2]
    
    code2name = {
      "AN":"Andaman and Nicobar",
      "AP":"Andhra Pradesh",
      "AR":"Arunachal Pradesh",
      "AS":"Assam",
      "BR":"Bihar",
      "CG":"Chhattisgarh",
      "CH":"Chandigarh",
      "DD":"Dadra and Nagar Haveli and Daman and Diu",
      "DL":"Delhi",
      "GA":"Goa",
      "GJ":"Gujarat",
      "HP":"Himachal Pradesh",
      "HR":"Haryana",
      "JH":"Jharkhand",
      "JK":"Jammu and Kashmir",
      "KA":"Karnataka",
      "KL":"Kerala",
      "LA":"Ladakh",
      "LD":"Lakshadweep",
      "MH":"Maharashtra",
      "ML":"Meghalaya",
      "MN":"Manipur",
      "MP":"Madhya Pradesh",
      "MZ":"Mizoram",
      "NL":"Nagaland",
      "OD":"Odisha",
      "PB":"Punjab",
      "PY":"Puducherry",
      "RJ":"Rajasthan",
      "SK":"Sikkim",
      "TN":"Tamil Nadu",
      "TR":"Tripura",
      "TS":"Telangana",
      "UK":"Uttarakhand",
      "UP":"Uttar Pradesh",
      "WB":"West Bengal"
    }
    
    return code2name[state_code]

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_rgb(image):
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA) # reduces the time needed to extract the colors
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3) # reshape the input to two dimensions for KMeans
    clf = KMeans(n_clusters = 1) # We only want dominant color, so n_clusters = 1
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    
    plt.figure(figsize = (8, 6))
    plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

    return rgb_colors[0]



def EuclideanDistance(rgb1, rgb2):
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2
    return np.sqrt((r2-r1)**2 + (g2-g1)**2 + (b2-b1)**2)

def get_nearest_color(rgb):
    color2rgb = {
    "White":(255,255,255),
    "Black":(0,0,0),
    "Gray":(128,128,128),
    "Silver":(192,192,192),
    "Red":(255,0,0),    
    "Blue":(0,0,255),
    "Brown":(165,42,42),
    "Green":(0,255,0),
    "Beige":(245,245,245),
    "Orange":(255,165,0),
    "Gold":(255,215,0),
    "Purple":(128,0,128)
    }
    
    nearest_color_dist = 442 #Maximum Distance in RGB space is 441.6729559300637
    nearest_color = "Others"
    for color in list(color2rgb.keys()):
        dist = EuclideanDistance(rgb, color2rgb[color])
        if dist < nearest_color_dist:
            nearest_color = color
            nearest_color_dist = dist

    return nearest_color


import cv2 as cv

# EXIF orientation TAG
ORIENTATION_TAG = [orient for orient in ExifTags.TAGS.keys() if ExifTags.TAGS[orient] == 'Orientation']

# Defines the default JSON configuration. More information at https://www.doubango.org/SDKs/anpr/docs/Configuration_options.html
JSON_CONFIG = {
    "debug_level": "info",
    "debug_write_input_image_enabled": False,
    "debug_internal_data_path": ".",
    
    "num_threads": -1,
    "gpgpu_enabled": True,
    "max_latency": -1,

    "klass_vcr_gamma": 1.5,
    
    "detect_roi": [0, 0, 0, 0],
    "detect_minscore": 0.1,

    "car_noplate_detect_min_score": 0.8,
    
    "pyramidal_search_enabled": True,
    "pyramidal_search_sensitivity": 0.28,
    "pyramidal_search_minscore": 0.3,
    "pyramidal_search_min_image_size_inpixels": 800,
    
    "recogn_minscore": 0.3,
    "recogn_score_type": "min"
}

TAG = "[PythonRecognizer] "

# Check result
def checkResult(operation, result):
    if not result.isOK():
        print(TAG + operation + ": failed -> " + result.phrase())
        assert False
    else:
        print(TAG + operation + ": OK -> " + result.json())

def IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    This is the recognizer sample using python language
    """)

    parser.add_argument("--image", required=True, help="Path to the image with ALPR data to recognize")
    parser.add_argument("--assets", required=False, default="../../../assets", help="Path to the assets folder")
    parser.add_argument("--charset", required=False, default="latin", help="Defines the recognition charset (a.k.a alphabet) value (latin, korean, chinese...)")
    parser.add_argument("--car_noplate_detect_enabled", required=False, default=False, help="Whether to detect and return cars with no plate")
    parser.add_argument("--ienv_enabled", required=False, default=platform.processor()=='i386', help="Whether to enable Image Enhancement for Night-Vision (IENV). More info about IENV at https://www.doubango.org/SDKs/anpr/docs/Features.html#image-enhancement-for-night-vision-ienv. Default: true for x86-64 and false for ARM.")
    parser.add_argument("--openvino_enabled", required=False, default=True, help="Whether to enable OpenVINO. Tensorflow will be used when OpenVINO is disabled")
    parser.add_argument("--openvino_device", required=False, default="CPU", help="Defines the OpenVINO device to use (CPU, GPU, FPGA...). More info at https://www.doubango.org/SDKs/anpr/docs/Configuration_options.html#openvino-device")
    parser.add_argument("--klass_lpci_enabled", required=False, default=False, help="Whether to enable License Plate Country Identification (LPCI). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#license-plate-country-identification-lpci")
    parser.add_argument("--klass_vcr_enabled", required=False, default=False, help="Whether to enable Vehicle Color Recognition (VCR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-color-recognition-vcr")
    parser.add_argument("--klass_vmmr_enabled", required=False, default=False, help="Whether to enable Vehicle Make Model Recognition (VMMR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-make-model-recognition-vmmr")
    parser.add_argument("--klass_vbsr_enabled", required=False, default=False, help="Whether to enable Vehicle Body Style Recognition (VBSR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-body-style-recognition-vbsr")
    parser.add_argument("--tokenfile", required=False, default="", help="Path to license token file")
    parser.add_argument("--tokendata", required=False, default="", help="Base64 license token data")

    args = parser.parse_args()
    
    # Update JSON options using values from the command args
    JSON_CONFIG["assets_folder"] = args.assets
    JSON_CONFIG["charset"] = args.charset
    JSON_CONFIG["car_noplate_detect_enabled"] = (args.car_noplate_detect_enabled == "True")
    JSON_CONFIG["ienv_enabled"] = (args.ienv_enabled == "True")
    JSON_CONFIG["openvino_enabled"] = (args.openvino_enabled == "True")
    JSON_CONFIG["openvino_device"] = args.openvino_device
    JSON_CONFIG["klass_lpci_enabled"] = (args.klass_lpci_enabled == "True")
    JSON_CONFIG["klass_vcr_enabled"] = (args.klass_vcr_enabled == "True")
    JSON_CONFIG["klass_vmmr_enabled"] = (args.klass_vmmr_enabled == "True")
    JSON_CONFIG["klass_vbsr_enabled"] = (args.klass_vbsr_enabled == "True")
    JSON_CONFIG["license_token_file"] = args.tokenfile
    JSON_CONFIG["license_token_data"] = args.tokendata

    # Initialize the engine
    checkResult("Init", 
                ultimateAlprSdk.UltAlprSdkEngine_init(json.dumps(JSON_CONFIG))
            )

    video_address = args.image
    cap = cv.VideoCapture(video_address)
    font = cv.FONT_HERSHEY_SIMPLEX
    fps = int(cap.get(cv.CAP_PROP_FPS))
    z = 0
    y = 0
    flag = 0
    width= int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer= cv.VideoWriter('../../../assets/images/out_video.mp4', cv.VideoWriter_fourcc('m','p','4','v'), fps, (width,height))
    impath = str(video_address)
    impath = impath[:-4]+"_out.jpg"

    while cap.isOpened():
        try:
            ret, image = cap.read()

            cv.imwrite(impath, image)
            image = Image.open(impath)
            width, height = image.size
            if image.mode == "RGB":
                format = ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGB24
            elif image.mode == "RGBA":
                format = ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGBA32
            elif image.mode == "L":
                format = ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_Y
            else:
                print(TAG + "Invalid mode: %s" % image.mode)
                assert False

            # Read the EXIF orientation value
            exif = image._getexif()
            exifOrientation = exif[ORIENTATION_TAG[0]] if len(ORIENTATION_TAG) == 1 and exif != None else 1
            break
        except:
            print("\nEXIF END\n")
            break
        break

    cap.release()
    cap = cv.VideoCapture(video_address)

    clr = [(0,255,0), (0,255,255), (0,0,255), (255,255,255)]
    car_plates_curr = {}
    car_plates_final = {}
    car_plates_prev = {}
    boxA = [0, 0]
    boxB = [0, 0]
    in_count = 0
    out_count = 0
    while cap.isOpened():
        try:            
            ret, image = cap.read()

            imageSize = image.shape
            checkBoxout = (0.35,0.4)
            checkBoxin = (0.35,0.4)

            # IN BOX
            x_1 = 0
            x_2 = int(imageSize[1]/2)-1
            y_1 = int(checkBoxout[0]*imageSize[0])
            y_2 = int(checkBoxout[1]*imageSize[0])-1
            shape = np.zeros_like(image)
            shape = cv.rectangle(shape,(x_1,y_1),(x_2,y_2),(0,0,255),cv.FILLED)
            
            # OUT BOX
            x__1 = int(imageSize[1]/2)
            x__2 = imageSize[1]-1
            y__1 = int(checkBoxin[0]*imageSize[0])
            y__2 = int(checkBoxin[1]*imageSize[0])-1
            shape = cv.rectangle(shape,(x__1,y__1),(x__2,y__2),(0,0,255),cv.FILLED)
            
            alpha = 0.5
            mask = shape.astype(bool)
            image[mask] = cv.addWeighted(image, alpha, shape, 1 - alpha, 0)[mask]

            z = z+1
            if z%fps==0 or z%fps==int(fps/2):
                y = y + 1
                flag = 1
            if flag == 1:
                result =   ultimateAlprSdk.UltAlprSdkEngine_process(
                                format,
                                image.tobytes(), # type(x) == bytes
                                width,
                                height,
                                0, # stride
                                exifOrientation
                                )
                                
                checkResult("Process",
                            result
                    )

                result = json.loads(result.json())

                print(f"\Second {y}\n")

                try:
                    for j in range(len(result["plates"])):
                        if result["plates"][j]["text"] not in car_plates_curr:
                            car_plates_curr[result["plates"][j]["text"]] = [len(car_plates_curr)+1]
                            k = [result["plates"][j]["car"]["warpedBox"]] + [result["plates"][j]["warpedBox"]]
                            label = [str(car_plates_curr[result["plates"][j]["text"]][0]), result["plates"][j]["text"]]
                            for i in range(3):
                                if i<2:
                                    k[i] = list(map(int, k[i][:3]+[k[i][-1]]))
                                    image = cv.rectangle(image,(k[i][0],k[i][1]),(k[i][2],k[i][3]),clr[i],3)
                                    x1, y1 = k[i][0], k[i][1]
                                    x2, y2 = k[i][2], k[i][3]
                                    if i == 0:
                                        car_plates_curr[result["plates"][j]["text"]].append([x1, y1, x2, y2])
                                        car_plates_curr[result["plates"][j]["text"]].append(result["plates"][j]["text"])

                                        boxA = [x1, y1, x2, y2]
                                        for p in car_plates_prev:
                                            boxB = car_plates_prev[p][1]
                                            ar = IOU(boxA, boxB)
                                            
                                            if ar >= 0.75 or car_plates_prev[p][2] == result["plates"][j]["text"]:
                                                label = [str(car_plates_prev[p][0]), result["plates"][j]["text"]]
                                                break
                                            else:
                                                car_plates_final[result["plates"][j]["text"]] = [len(car_plates_final)+1, [x1, y1, x2, y2], result["plates"][j]["text"]]
                                                label = [str(car_plates_final[result["plates"][j]["text"]][0]), car_plates_final[result["plates"][j]["text"]][2]]

                                        if boxA[1] <= y_1 and boxA[0]>=x_1:
                                            in_count +=1
                                        
                                        if boxA[1] >= y__1 and boxA[0]<=x__2:
                                            out_count +=1
                                else:
                                    pass
                                    # x1, y1 = (10, 50)
                                    # label.append("Number Plates found: "+str(len(car_plates_final)))
                                if i !=0:
                                    (w, h), _ = cv.getTextSize(label[i], cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                    image = cv.rectangle(image, (x1, y1 - 20), (x1 + w, y1), clr[2], -1)
                                    image = cv.putText(image, label[i], (x1, y1-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, clr[3], 2)
                                car_plates_prev = car_plates_curr
                except:
                    pass
        
            # x1, y1 = (10, 50)
            # label = []
            # i=0
            # label.append("Number Plates found: "+str(len(car_plates_final)))
            # (w, h), _ = cv.getTextSize(label[i], cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # image = cv.rectangle(image, (x1, y1 - 20), (x1 + w, y1), clr[2], -1)
            # image = cv.putText(image, label[i], (x1, y1-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, clr[3], 2)

            x1, y1 = (10, 70)
            label = []
            i=0
            label.append("IN: "+str(in_count))
            (w, h), _ = cv.getTextSize(label[i], cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            image = cv.rectangle(image, (x1, y1 - 20), (x1 + w, y1), clr[2], -1)
            image = cv.putText(image, label[i], (x1, y1-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, clr[3], 2)

            x1, y1 = (10, 90)
            label = []
            i=0
            label.append("OUT: "+str(out_count))
            (w, h), _ = cv.getTextSize(label[i], cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            image = cv.rectangle(image, (x1, y1 - 20), (x1 + w, y1), clr[2], -1)
            image = cv.putText(image, label[i], (x1, y1-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, clr[3], 2)

            writer.write(image)
            cv.imshow('image',image)
            # os.remove(impath)
            flag = 0
            k = cv.waitKey(30) & 0xff # press ESC to exit
            if k == 27 or cv.getWindowProperty('image', 0)<0:
                break
        except Exception as e:
        # else:
            print("\nEND\n")
            print(e)
            k = cv.waitKey(30) & 0xff # press ESC to exit
            if k == 27 or cv.getWindowProperty('image', 0)<0:
                break
            break

    cv.destroyAllWindows()
    cap.release()
    writer.release()

    # DeInit the engine
    checkResult("DeInit", 
                ultimateAlprSdk.UltAlprSdkEngine_deInit()
            )
    tic = time.time()
    time1 = tic - toc
    print("Time taken : " + str(time1) + " seconds")

    cap = cv.VideoCapture("D:/App_Files/Codes/VSCode_Programs/ES/ultimateALPR-SDK/assets/images/out_video.mp4")
    fps = cap.get(cv.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    print('fps = ' + str(fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    minutes = int(duration/60)
    seconds = duration%60
    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    cap.release()

    cap = cv.VideoCapture("D:/App_Files/Codes/VSCode_Programs/ES/ultimateALPR-SDK/assets/images/vid.mp4")
    fps = cap.get(cv.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    cap.release()

    areas = []
    for k in car_plates_final:
       box_ar = (car_plates_final[k][1][2] - car_plates_final[k][1][0]) * (car_plates_final[k][1][3] - car_plates_final[k][1][1])
       areas.append(box_ar)
    areas = np.array(areas)


    print(f"Analytics:\n    Mean: {np.mean(areas)}\n    Median: {np.median(areas)}\n")

    print('    fps = ' + str(fps))
    print('    number of frames = ' + str(frame_count))
    print('    duration (S) = ' + str(duration))
    minutes = int(duration/60)
    seconds = duration%60
    print('    duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    # Creating plot
    #fig = plt.figure(figsize =(10, 7))
    #
    #plt.hist(areas, bins = 4) 
    #
    #plt.title("Numpy Histogram") 
    #
    # show plot
    #plt.show()

    # Code here from  this indent
    code2name = {
      "AN":"Andaman and Nicobar",
      "AP":"Andhra Pradesh",
      "AR":"Arunachal Pradesh",
      "AS":"Assam",
      "BR":"Bihar",
      "CG":"Chhattisgarh",
      "CH":"Chandigarh",
      "DD":"Dadra and Nagar Haveli and Daman and Diu",
      "DL":"Delhi",
      "GA":"Goa",
      "GJ":"Gujarat",
      "HP":"Himachal Pradesh",
      "HR":"Haryana",
      "JH":"Jharkhand",
      "JK":"Jammu and Kashmir",
      "KA":"Karnataka",
      "KL":"Kerala",
      "LA":"Ladakh",
      "LD":"Lakshadweep",
      "MH":"Maharashtra",
      "ML":"Meghalaya",
      "MN":"Manipur",
      "MP":"Madhya Pradesh",
      "MZ":"Mizoram",
      "NL":"Nagaland",
      "OD":"Odisha",
      "PB":"Punjab",
      "PY":"Puducherry",
      "RJ":"Rajasthan",
      "SK":"Sikkim",
      "TN":"Tamil Nadu",
      "TR":"Tripura",
      "TS":"Telangana",
      "UK":"Uttarakhand",
      "UP":"Uttar Pradesh",
      "WB":"West Bengal"
    }
    states_count = {}
    for number_plate in list(car_plates_final.keys()):
        state_code = number_plate[:2]
        if state_code in states_count.keys():
            states_count[state_code]+=1
        else:
            states_count[state_code]=1

    print('    Statewise Distribution')
    others_count = 0
    name_count={}
    for state_code in list(states_count.keys()):
        try:
            print('        '+code2name[state_code]+': '+str(states_count[state_code]))
            name_count[code2name[state_code]] = states_count[state_code]
        except:
            others_count+=1
    print('        Others: '+str(others_count))
    name_count['Others'] = others_count

    fig1 = plt.figure(figsize = (10, 7))
 
    # creating the bar plot
    plt.bar(list(name_count.keys()), list(name_count.values()), width = 0.4)
 
    plt.xlabel("State")
    plt.ylabel("Count")
    plt.title("Statewise Distribution")
    plt.show()
    

    size_count = {"Small":0, "Medium":0, "Large":0}
    limit1, limit2 = getSegSizeValues(areas)
    for area in list(areas):
        size = getSegBySize(area, limit1, limit2)
        size_count[size]+=1

    print('    Sizewise Distribution')
    print('        ',end="")
    print(size_count)
    
    fig2 = plt.figure(figsize = (10, 7))
 
    # creating the bar plot
    plt.bar(list(size_count.keys()), list(size_count.values()), width = 0.4)
 
    plt.xlabel("Size of car")
    plt.ylabel("Count")
    plt.title("Sizewise Distribution")
    plt.show()
    

    # keep this code block at last only
    a = 1
    while (a == 1):
        try:
            a = int(input("\nPress 1 to execute python\n"))
            if a == 1:
                ip = input("\nEnter Python\n")
                eval(ip)
        except:
            a = 1
            print("Error somewhere")

