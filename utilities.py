from os import path
from dash import dcc
import copy
from dash import dcc
from dash import html
import os
import base64
from io import BytesIO as _BytesIO
from PIL import Image
import numpy as np
import cv2
import json
import time
from pkg_resources import SOURCE_DIST
import processing_function as pfunc

from app import APP_PATH, DIV_DICT

# Image utility functions
def pil_to_b64(im, enc_format="png", verbose=False, **kwargs):
    """
    Converts a PIL Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :param verbose: Allow for debugging tools
    :return: base64 encoding
    """
    t_start = time.time()

    buff = _BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    t_end = time.time()
    if verbose:
        print(f"PIL converted to b64 in {t_end - t_start:.3f} sec")

    return encoded
def b64_to_pil(string):
    decoded = base64.b64decode(string)
    buffer = _BytesIO(decoded)
    im = Image.open(buffer)
    return im
def save_to_local(content,session_id):
    # Parse the string and convert to pil
    string = content.split(";base64,")[-1]
    im_np = b64_to_numpy(string,False)
    #存储上传图片并创建输出图片
    cv2.imwrite(os.path.join(
                            os.path.join(APP_PATH,"data"),
                            str(session_id)+"-original-img"+".png"
                            ),
                im_np
                )
    cv2.imwrite(os.path.join(
                            os.path.join(APP_PATH,"data"),
                            str(session_id)+"-lastest-img"+".png"
                            ),
                im_np
                )
    print(session_id+"upload succeed")
    return True

def b64_to_numpy(string, to_scalar=True):
    im = b64_to_pil(string)
    np_array = np.asarray(im)

    if to_scalar:
        np_array = np_array / 255.0
    
    #arrange the depth of image
    image_shape=np.shape(np_array)
    
    if  len(image_shape) == 2:#bmp格式
        r=np.copy(np_array)
        b=np.copy(np_array)
        g=np.copy(np_array)
        full_mask=255*np.ones(np.shape(np_array))
        np_array=np.stack((r,g,b,full_mask),axis=2)#重复通道
    elif len(image_shape) == 3 :
        if image_shape[2] == 3 :#jpg格式
            b=np.copy(np_array[:,:,0])
            g=np.copy(np_array[:,:,1])
            r=np.copy(np_array[:,:,2])
            full_mask=255*np.ones(np.shape(np_array[:,:,0]))
            np_array=np.stack((r,g,b,full_mask),axis=2)#添加通道,注意通道的顺序
        elif image_shape[2] == 4 :#png格式
            b=np.copy(np_array[:,:,0])
            g=np.copy(np_array[:,:,1])
            r=np.copy(np_array[:,:,2])
            full_mask=np.copy(np_array[:,:,3])
            np_array=np.stack((r,g,b,full_mask),axis=2)#添加通道,注意通道的顺序
        else:
            return None #格式不正确 
    else:
        return None #格式不正确

    return np_array

def get_address_lastest_img(session_id,isRelative=False,isFilename=False):
    if(isRelative):
        return "/data"+str(session_id)+"-lastest-img"+".png"
    if(isFilename):
        return str(session_id)+"-lastest-img"+".png"
    return os.path.join(
                            os.path.join(APP_PATH,"data"),
                            str(session_id)+"-lastest-img"+".png"
                            )
def get_address_original_img(session_id,isRelative=False,isFilename=False):
    if(isRelative):
        return "/data"+str(session_id)+"-original-img"+".png"
    if(isFilename):
        return str(session_id)+"-original-img"+".png"
    return os.path.join(
                            os.path.join(APP_PATH,"data"),
                            str(session_id)+"-original-img"+".png"
                            )

def image_io_and_op(session_id,img,operations):
    if(operations is not None):
        for operation in operations:    
            if(operation not in DIV_DICT):
                return False
            else:
                ##demo procession
                img = eval("pfunc."+operation)(img)###初版，eval is evil
                ##
        cv2.imwrite(get_address_lastest_img(session_id),img)
        return True
    cv2.imwrite(get_address_lastest_img(session_id),img)#没有操作，则替换
    return False
def run_op(session_id,storage,operation):
    
    print(storage)
    if( operation in DIV_DICT):
        ##save the action
        storage["action_stack"].append(operation)
        storage["repealed_action_stack"]=[]
        img = cv2.imread(get_address_lastest_img(session_id),-1)
        image_io_and_op(session_id,img,[operation])
        
        print(operation + "successfully")
    
    return get_address_lastest_img(session_id),storage

def undo_why_undo(session_id,storage,direction):
    
    
    print(storage)
    if(direction == "backward"  and (storage["action_stack"] is not None) ):
        undo_op = storage["action_stack"].pop()
        img = cv2.imread(get_address_original_img(session_id),-1)
        image_io_and_op(session_id,img,storage["action_stack"])
        storage["repealed_action_stack"].append(undo_op)
        print("backward")
    if(direction == "forward" and (storage["repealed_action_stack"] is not None) ):
        repealed_op = storage["repealed_action_stack"].pop()
        storage["action_stack"].append(repealed_op)
        img = cv2.imread(get_address_lastest_img(session_id),-1)
        image_io_and_op(session_id,img,[repealed_op])
        print("forward")
    
    return get_address_lastest_img(session_id),storage

def get_transform_data(session_id , transform_op):
    source = get_address_lastest_img(session_id)
    img = cv2.imread(source,-1)
    if(transform_op == DIV_DICT["histogram"]):
        hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
        hist = np.stack((hist_r.T,hist_g.T,hist_b.T),axis=0)
        print(np.shape(hist))
        return hist
    elif(transform_op == DIV_DICT["dft_one"]):
        return pfunc.dft_one(img)
    elif(transform_op == DIV_DICT["dct_one"]):
        return pfunc.dct_one(img)
    return None