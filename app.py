#调用模块 
import json
from logging import disable
import os
import time
from typing import Dict
import uuid
from copy import copy, deepcopy
import csv
import sys
import pathlib
from cv2 import FlannBasedMatcher

import dash
from dash.html.Div import Div
from dash.html.S import S
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from PIL import Image
import plotly.graph_objs as go
import plotly.express as px
from flask import send_from_directory
import numpy as np
#import matplotlib.pyplot  as plt
import base64
from io import BytesIO
#自定义模块
import utilities
from processing_function import *

#初始定义
app = dash.Dash(__name__)
app.title = "综合图像处理系统"
server = app.server

APP_PATH = str(pathlib.Path(__file__).parent.resolve()) 


# 运行前清空data中存储的文件
data_folder = os.path.join(APP_PATH, "data")
for the_file in os.listdir(data_folder):
    file_path = os.path.join(data_folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)


######################Web界面的定义与响应定义####################
def server_layout():
    ###定义web界面的布局
    #生成一次会话的uid
    session_id=str(uuid.uuid4())
    
    #存储图片（创建图片存储单元：原始图片，最新图片）
    return app_root(session_id=session_id)


# Display utility functions
def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}

###########布局##############

###变量定义
DIV_DICT=dict() #html单元id字典
INTIAL_IMAGE_SOURCE=os.path.join(os.path.join(APP_PATH,"images"),"Initial.png")

SESSION_STORAGE=json.dumps(#存储于html中的数据，用于记录每个会话的设置和操作
    ##repealed_action_stack:被撤销的操作,每次有新的操作,栈将清空
    ##action_stack:由初始到最后的操作
    {
        "repealed_action_stack":[],
        "action_stack":[],
    }
)

###root###
###总布局###
##将该布局中的id存入字典
DIV_DICT["root"]="root"
DIV_DICT["session-id"]="session-id"
def app_root(session_id):
    utilities.save_to_local(utilities.pil_to_b64(Image.open(INTIAL_IMAGE_SOURCE)),session_id)
    return html.Div(
        id=DIV_DICT["root"],
        children=[
            html.Div(children=session_id,id=DIV_DICT["session-id"]),
            app_container(),
            app_sidebar(),
        ]
    )

###container###
DIV_DICT["app-container"]="app-container"
def app_container():
    return html.Div(
                id=DIV_DICT["app-container"],
                children=[
                    app_banner(),
                    app_image(),
                ],
            )
#banner#
DIV_DICT["banner"]="banner"
DIV_DICT["logo"]="logo"
DIV_DICT["title"]="title"
def app_banner():
    # Banner display
    return  html.Div(
                id=DIV_DICT["banner"],
                children=[
                    html.Img(
                        id=DIV_DICT["logo"], 
                        src=Image.open(os.path.join(os.path.join(APP_PATH,"images"),"nudt_removeBackground.png"))
                    ),
                    html.H2("综合图像处理系统", id=DIV_DICT["title"]),
                ],
            )


#image#
DIV_DICT["image"]="image"
DIV_DICT["div-interactive-image"]="div-interactive-image"
DIV_DICT["interactive-image"]="interactive-image"
DIV_DICT["div-storage"]="div-storage"
DIV_DICT["time-mark"]="time-mark"
def app_image():
    return html.Div(
        id=DIV_DICT["image"],
        children=[
            html.Div(
                #children=refresh_interactive_image(INTIAL_IMAGE_SOURCE),
                id=DIV_DICT["div-interactive-image"],    
            ),
            html.Div(
                id=DIV_DICT["div-storage"],
                children=SESSION_STORAGE,
            ),
           
        ]
    )


def refresh_interactive_image(image_source):
    source=Image.open(image_source)
    width, height = source.size
    return [
        html.Div(
                id=DIV_DICT["time-mark"],
                children=str(time.asctime( time.localtime(time.time()) ))#记录时间，标记图片
            ),
        dcc.Graph(#显示图像设置
                    id=DIV_DICT["interactive-image"],
                    figure={
                        "data": [],
                        "layout": {
                            "autosize": True,
                            "paper_bgcolor": "#31343a",
                            "plot_bgcolor": "#31343a", 
                            #"margin": go.layout.Magin(l=40, b=40, t=26, r=10),
                            "xaxis": {
                                "range": (0, width),
                                "scaleanchor": "y",
                                "scaleratio": 1,
                                "color": "white",
                                "gridcolor": "#43454a",
                                "tickwidth": 1,
                            },
                            "yaxis": {
                                "range": (0, height),
                                "color": "white",
                                "gridcolor": "#43454a",
                                "tickwidth": 1,
                            },
                            "images": [
                                {
                                    "xref": "x",
                                    "yref": "y",
                                    "x": 0,
                                    "y": 0,
                                    "yanchor": "bottom",
                                    "sizing": "fix",
                                    "sizex": width,
                                    "sizey": height,
                                    "layer": "above",
                                    "source": source,
                                    "paper_bgcolor": "#31343a",
                                    "plot_bgcolor": "#31343a"
                                }
                            ],
                            "dragmode": "pan",
                        },
                    },
                )
    ]

def refresh_image(img_source , storage):
    return [
            html.Div(
                children=refresh_interactive_image(img_source),
                id=DIV_DICT["div-interactive-image"],    
            ),
            html.Div(
                id=DIV_DICT["div-storage"],
                children=storage,
            ),           
        ]
###sidebar###
DIV_DICT["sidebar"]="sidebar"
DIV_DICT["under_tabs"]="under_tabs"
def app_sidebar():
    return html.Div(
                id=DIV_DICT["sidebar"],
                children=[
                    #app_sidebar_tabs(),
                    cards_image_spatial(),
                    cards_image_transformed()
                    #html.Div(id=DIV_DICT["under_tabs"]),

#+++++++++++++++++++++++++++++++++++++++++++++


                    ]
            )

#tabs and tab#
DIV_DICT["sidebar_tabs"]="sidebar_tabs"
DIV_DICT["tab_image_spatial"]="tab_image_spatial"
DIV_DICT["tab_image_transformed"]="tab_image_transformed"

def app_sidebar_tabs():
    return dcc.Tabs(
        id=DIV_DICT["sidebar_tabs"],
        value=DIV_DICT["tab_image_spatial"],
        parent_className='custom-tabs',
        className='custom-tabs-container',
        children=[
            dcc.Tab(
                label='空间域', 
                value=DIV_DICT["tab_image_spatial"],
                className="tab_style", 
                selected_className="tab_selected_style",
            ),
            dcc.Tab(
                label='变换域', 
                value=DIV_DICT["tab_image_transformed"],
                className="tab_style", 
                selected_className="tab_selected_style",
            ),
            
        ]

    )


#Card#
def Card(children, **kwargs):
    return html.Section( #section means the content in the sidebar
        children,
        style=_merge(
            {
                "padding": 12,
                "margin": 5,
                # Remove possibility to select the text for better UX
                "user-select": "none",
                "-moz-user-select": "none",
                "-webkit-user-select": "none",
                "-ms-user-select": "none",
            },
            kwargs.get("style", {}),
        ),
        **_omit(["style"], kwargs),
    )

#CustomDropdown
def CustomDropdown(**kwargs):
    return html.Div(
        dcc.Dropdown(**kwargs, style={"margin-top": "1px", "margin-bottom": "1px"})
    )

#cards-image-spatial#
DIV_DICT["cards_image_spatial"]="cards_image_spatial"
def cards_image_spatial():
    return html.Div(
        id=DIV_DICT["cards_image_spatial"],
        children=[
            card_upload(),
            #card_download(),
            card_filters(),
            button_group(),
            card_select_plot(),
        ],
    )

#card_upload#
DIV_DICT["upload"]="upload"
def card_upload():
    return Card(
        dcc.Upload(
                id="upload",
                #contents=Image.open(INTIAL_IMAGE_SOURCE),
                children=[
                    "拖拽插入 或者  ",
                    html.A(children="选择插入"),
                ],
                # No CSS alternative here
                style={
                    "color": "white",
                    "font-size": "22px",
                    "width": "100%",
                    "height": "50px",
                    "lineHeight": "50px",
                    "borderWidth": "1.5px",
                    "borderStyle": "dashed",
                    "borderRadius": "20px",
                    "borderColor": "snow",
                    "textAlign": "center",
                    "padding": "2rem 0",
                    "margin-bottom": "0rem",
                    "margin-top": "1.2rem",
                    "background-color":'#4e505a'
                },
                accept=".jpg,.png,.bmp,.jpeg",
            ),
    )
'''
#card_download
DIV_DICT["div-download-link"]="div-download-link"
DIV_DICT["download"]="download"
def card_download():
    #return Card(
    #    children=[
    #            html.Button(
    #                    id=DIV_DICT["button_download"],
    #                    children="下载图片"
    #                    ),
    #            dcc.Download(
    #                id=DIV_DICT["download"],
    #            ),
    #    ]
    #)
    return html.Div(
                id=DIV_DICT["div-download-link"],
                children=gen_download_url("none"),
                )
'''
def gen_download_url(file):
    return (
                html.A(children="下载", href=f'/download/{file}', target="_blank", style={"color":"white","text-decoration":"none"})
            )


#card_filters
DIV_DICT["card_filters"]="card_filters"
DIV_DICT["rgb2gray"]="rgb2gray"
DIV_DICT["myPseudoColor"]="myPseudoColor"
DIV_DICT["myBlur"]="myBlur"
DIV_DICT["myMedianBlur"]="myMedianBlur"
DIV_DICT["myGaussianBlur"]="myGaussianBlur"
DIV_DICT["myBilateralFilter"]="myBilateralFilter"
DIV_DICT["myNlMeans"]="myNlMeans"
DIV_DICT["mySobel"]="mySobel"
DIV_DICT["myScharr"]="myScharr"
DIV_DICT["myLaplacian"]="myLaplacian"
DIV_DICT["myHistEqual"]="myHistEqual"
DIV_DICT["myDilate"]="myDilate"
DIV_DICT["myErode"]="myErode"
DIV_DICT["myGradiant"]="myGradiant"
DIV_DICT["myCanny"]="myCanny"
DIV_DICT["myHoughLines"]="myHoughLines"
DIV_DICT["myHoughCircles"]="myHoughCircles"
DIV_DICT["myWaterShed"]="myWaterShed"
DIV_DICT["myGrabCut"]="myGrabCut"
DIV_DICT["myOSTU"]="myOSTU"
DIV_DICT["myAdaptiveThreshold"]="myAdaptiveThreshold"
DIV_DICT["myInpaint"]="myInpaint"

def card_filters():
    return Card(
                CustomDropdown(
                    id=DIV_DICT["card_filters"],
                    options=[
                        #彩色变换
                        {"label": "灰度化", "value": DIV_DICT["rgb2gray"]},
                        {"label": "伪彩色", "value": DIV_DICT["myPseudoColor"]},
                        #空间滤波
                        {"label": "均值滤波(k=5)", "value": DIV_DICT["myBlur"]},
                        {"label": "中值滤波(k=5)", "value": DIV_DICT["myMedianBlur"]},
                        {"label": "高斯滤波(k=5)", "value": DIV_DICT["myGaussianBlur"]},
                        {"label": "双边滤波(d=30,σc=σs=50)", "value": DIV_DICT["myBilateralFilter"]},
                        {"label": "非局部均值(h=15,hc=15)", "value": DIV_DICT["myNlMeans"]},
                        #锐化处理
                        {"label": "Sobel锐化(k=3,dx=1)", "value": DIV_DICT["mySobel"]},
                        {"label": "Scharr锐化(k=3,dx=1)", "value": DIV_DICT["myScharr"]},
                        {"label": "Laplacian锐化(k=3)", "value": DIV_DICT["myLaplacian"]},
                        #直方图均衡化
                        {"label": "直方图均衡化", "value": DIV_DICT["myHistEqual"]},
                        #形态学
                        {"label": "膨胀操作 (k=5)", "value": DIV_DICT["myDilate"]},
                        {"label": "腐蚀操作 (k=5)", "value": DIV_DICT["myErode"]},
                        {"label": "形态学梯度", "value": DIV_DICT["myGradiant"]},
                        #图像分割
                        {"label": "Canny边缘检测", "value": DIV_DICT["myCanny"]},
                        {"label": "霍夫线变换", "value": DIV_DICT["myHoughLines"]},
                        {"label": "霍夫圆变换", "value": DIV_DICT["myHoughCircles"]},
                        {"label": "分水岭算法", "value": DIV_DICT["myWaterShed"]},
                        {"label": "GrabCut算法", "value": DIV_DICT["myGrabCut"]},
                        #阈值处理
                        {"label": "OSTU阈值化", "value": DIV_DICT["myOSTU"]},
                        {"label": "自适应阈值化", "value": DIV_DICT["myAdaptiveThreshold"]},
                        #去水印
                        {"label": "去水印", "value": DIV_DICT["myInpaint"]},
                    ],
                    searchable=True,
                    placeholder="空间域",
                    multi=False,
                )
            )

#button_group#
DIV_DICT["button_group"]="button_group"
DIV_DICT["button_run_operation"]="button-run-operation"
DIV_DICT["button_undo"]="button-undo"
DIV_DICT["download"]="download"
DIV_DICT["forward"]="forward"
def button_group():
    return Card(
                id=DIV_DICT["button_group"],
                children=[
                    html.Button(
                        "执行", 
                        id=DIV_DICT["button_run_operation"],
                        n_clicks=0
                        ),
                    html.Button(
                        "撤回", 
                        id=DIV_DICT["button_undo"],
                        n_clicks=0
                        ),
                    html.Button(
                        "前进", 
                        id=DIV_DICT["forward"],
                        n_clicks=0
                        ),
                    html.Button(
                        #'下载',
                        disabled = "DISABLED",
                        children=gen_download_url(None),
                        id=DIV_DICT["download"],
                    ),
                ],
            )


#cards-image-transformed#
DIV_DICT["cards_image_transformed"]="cards_image_transformed"
def cards_image_transformed():
    return html.Div(
        id=DIV_DICT["cards_image_transformed"],
        children=[
            
            plot_graph(),
        ],
    )

#card_select_plot#
DIV_DICT["card_select_plot"]="card_select_plot"
DIV_DICT["dropdown_select_plot"]="dropdown_select_plot"
DIV_DICT["histogram"] = "histogram"
DIV_DICT["dft_one"]="dft_one"
DIV_DICT["dct_one"]="dct_one"
def card_select_plot():
    return Card(
        #id=DIV_DICT["card_select_plot"],
        #children=[
            CustomDropdown(
                    id=DIV_DICT["dropdown_select_plot"],
                    options=[
                        {"label": "直方图", "value": DIV_DICT["histogram"]},
                        {"label": "DFT", "value": DIV_DICT["dft_one"]},
                        {"label": "DCT", "value": DIV_DICT["dct_one"]},
                    ],
                    value='histogram',
                    searchable=True,
                    placeholder="变换域",
                    multi=False,
                ),
        #],
        )

#plot_graph#
DIV_DICT["plot_graph"]="plot_graph"
def plot_graph():
    return dcc.Graph(
                    id=DIV_DICT["plot_graph"],
                    figure={
                        "layout": {
                            "paper_bgcolor": "#272a31",
                            "plot_bgcolor": "#272a31",
                        }
                    },
                    config={"displayModeBar": False},
                )

def show_histogram(hg):
    def hg_trace(name, color, hg):
        line = go.Scatter(
            x=list(range(0, 256)),
            y=hg,
            name=name,
            line=dict(color=(color)),
            mode="lines",
            showlegend=False,
        )
        fill = go.Scatter(
            x=list(range(0, 256)),
            y=hg,
            mode="lines",
            name=name,
            line=dict(color=(color)),
            fill="tozeroy",
            hoverinfo="none",
        )

        return line, fill
    rhg = hg[0][0]
    ghg = hg[1][0]
    bhg = hg[2][0]
    data = [
        *hg_trace("Red", "#FF4136", rhg),
        *hg_trace("Green", "#2ECC40", ghg),
        *hg_trace("Blue", "#0074D9", bhg),
    ]
    title = "RGB 直方图"
    layout = go.Layout(
        autosize=True,
        #height=1000,
        #width=3400,
        title=title,
        margin=go.Margin(l=50, r=30),
        legend=dict(x=0, y=1.15, orientation="h"),
        paper_bgcolor="#272a31",
        plot_bgcolor="#272a31",
        font=dict(color="darkgray"),
        xaxis=dict(gridcolor="#43454a"),
        yaxis=dict(gridcolor="#43454a"),

    )
    return dcc.Graph(figure=go.Figure(data=data,layout= layout))

def show_gray_img(img):
    figure = px.imshow(img)
    figure["layout"]["paper_bgcolor"]= "#272a31"
    figure["layout"]["plot_bgcolor"]= "#272a31"
    return dcc.Graph(figure=figure)

##########响应#################
@app.callback(
    Output(DIV_DICT["div-interactive-image"], "children"),
    Output(DIV_DICT["upload"],"contents"),
    Output(DIV_DICT["div-storage"],"children"),
    [   
        Input(DIV_DICT["upload"], "contents")],
    [
        State(DIV_DICT["div-storage"], "children"),
        State(DIV_DICT["session-id"], "children"),
    ],
)
def upload_image(
    content,
    storage,
    session_id,
):  
    
    storage = json.loads(storage)
    if( not content == None):
        #storage = json.loads(storage)
        if(not utilities.save_to_local(content,session_id)):
            #fail
            None

        else:
            #success
            storage["action_stack"]=[]
            storage["repealed_stack"]=[]
            None
    storage = json.dumps(storage)
    return refresh_interactive_image(utilities.get_address_lastest_img(session_id)),None,storage


@app.server.route('/download/<file>')
def download(file):
    return send_from_directory("data", file)




@app.callback(
    Output(DIV_DICT["under_tabs"], 'children'),
    Input(DIV_DICT["sidebar_tabs"], 'value')
)
def tab_change(tab):
    '''
    if tab == DIV_DICT["tab_image_spatial"]:
        return cards_image_spatial()
    elif tab == DIV_DICT["tab_image_transformed"]:
        return cards_image_transformed()
    else:
        print("error")
    '''
    return html.Div(childern=[cards_image_spatial(), cards_image_transformed()])




@app.callback(
    Output(DIV_DICT["button_run_operation"],"n_clicks"),
    Output(DIV_DICT["button_undo"],"n_clicks"),
    Output(DIV_DICT["forward"],"n_clicks"),
    Output(DIV_DICT["image"],"children"),
    [
        Input(DIV_DICT["button_run_operation"],"n_clicks"),
        Input(DIV_DICT["button_undo"],"n_clicks"),
        Input(DIV_DICT["forward"],"n_clicks"),
        #State(DIV_DICT["upload"],"loading_state"),
        State(DIV_DICT["div-storage"],"children"),
        State(DIV_DICT["card_filters"],"value"),
        State(DIV_DICT["session-id"],"children")
    ]
)
def run_operation(
    click_run,
    click_backward,
    click_forward,
    #loading_state,
    storage,
    operation_name,
    session_id
):      
    print(click_run,click_backward,click_forward)
    source = utilities.get_address_lastest_img(session_id)
    
    storage = json.loads(storage)
    if(click_run >=1 ):
        print("run")
        source,storage=utilities.run_op(session_id,storage,operation_name)
    elif(click_backward >=1 ):
        print("backward")
        source,storage=utilities.undo_why_undo(session_id,storage,"backward")
    elif(click_forward >=1 ):
        print("forward")
        source,storage=utilities.undo_why_undo(session_id,storage,"forward")
    else:
        None
    storage= json.dumps(storage)
    return 0,0,0,refresh_image(source, storage)




@app.callback(
    Output(DIV_DICT["download"],"children"),
    [Input(DIV_DICT["time-mark"],"children"),
    State(DIV_DICT["session-id"],"children"),
    ]
)
def download_refresh(time,session_id):
    return gen_download_url(utilities.get_address_lastest_img(session_id,isFilename=True))

@app.callback(
    Output(DIV_DICT["cards_image_transformed"],"children"),
    Input(DIV_DICT["dropdown_select_plot"],"value"),
    Input(DIV_DICT["time-mark"],"children"),
    State(DIV_DICT["session-id"],"children")
    
)
def show_plot(plot_name,time,session_id):
    
    pre_data = utilities.get_transform_data(session_id,plot_name)
    print(np.shape(pre_data))
    if( plot_name == DIV_DICT["histogram"]):
        print(plot_name)
        figure = show_histogram(pre_data)
        return figure
    if( plot_name == DIV_DICT["dft_one"]):
        print(plot_name)
        figure = show_gray_img(pre_data)
        return figure
    if( plot_name == DIV_DICT["dct_one"]):
        print(plot_name)
        figure = show_gray_img(pre_data)
        return figure
    return None
############################
# Running the server
if __name__ == "__main__":
    app.layout=server_layout()
    app.run_server(host="0.0.0.0",port="19000",debug=False)