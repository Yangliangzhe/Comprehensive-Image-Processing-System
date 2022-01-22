from typing import Dict
import dash
from dash.html.Div import Div
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import cv2
import os
import plotly.graph_objs as go
import json
from PIL import Image

from app import APP_PATH,app
import utilities

# Display utility functions
def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}

###########布局##############

###变量定义
DIV_DICT=dict() #html单元id字典


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
def app_image():
    return html.Div(
        id=DIV_DICT["image"],
        children=[
            html.Div(
                children=refresh_interactive_image(Image.open(os.path.join(os.path.join(APP_PATH,"images"),"Genshin.jpg"))),
                id=DIV_DICT["div-interactive-image"],    
            ),
            html.Div(
                id=DIV_DICT["div-storage"],
                children=SESSION_STORAGE,
            )
        ]
    )

def refresh_interactive_image(image_source):
    return dcc.Graph(#显示图像设置
                    id="interactive-image",
                    figure={
                        "data": [],
                        "layout": {
                            "autosize": True,
                            "paper_bgcolor": "#31343a",
                            "plot_bgcolor": "#31343a", 
                            "margin": go.Margin(l=40, b=40, t=26, r=10),
                            "xaxis": {
                                "range": (0, 1980),
                                "scaleanchor": "y",
                                "scaleratio": 1,
                                "color": "white",
                                "gridcolor": "#43454a",
                                "tickwidth": 1,
                            },
                            "yaxis": {
                                "range": (0, 1080),
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
                                    "sizing": "stretch",
                                    "sizex": 1980,
                                    "sizey": 1080,
                                    "layer": "above",
                                    "source": image_source,
                                    "paper_bgcolor": "#31343a",
                                    "plot_bgcolor": "#31343a"
                                }
                            ],
                            "dragmode": "pan",
                        },
                    },
                )


###sidebar###
DIV_DICT["sidebar"]="sidebar"
DIV_DICT["under_tabs"]="under_tabs"
def app_sidebar():
    return html.Div(
                id=DIV_DICT["sidebar"],
                children=[
                    app_sidebar_tabs(),
                    html.Div(id=DIV_DICT["under_tabs"]),
                    ]
            )

#tabs and tab#
DIV_DICT["sidebar_tabs"]="sidebar_tabs"
DIV_DICT["tab_image_operation"]="tab_image_operation"
DIV_DICT["tab_image_analysis"]="tab_image_analysis"

def app_sidebar_tabs():
    return dcc.Tabs(
        id=DIV_DICT["sidebar_tabs"],
        value=DIV_DICT["tab_image_operation"],
        children=[
            dcc.Tab(
                label='图像操作', 
                value=DIV_DICT["tab_image_operation"],
                className="tab_style", 
                selected_className="tab_selected_style",
            ),
            dcc.Tab(
                label='图像分析', 
                value=DIV_DICT["tab_image_analysis"],
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
                "padding": 20,
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
        dcc.Dropdown(**kwargs), style={"margin-top": "5px", "margin-bottom": "5px"}
    )

#cards-image-operation#
DIV_DICT["cards_image_operation"]="cards_image_operation"
def cards_image_operation():
    return html.Div(
        id=DIV_DICT["cards_image_operation"],
        children=[
            card_upload(),
            card_filters(),
            button_group(),
        ],
    )

#card_upload#
DIV_DICT["upload"]="upload"
def card_upload():
    return Card(
        dcc.Upload(
                id="upload-image",
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
                    "margin-bottom": "2rem",
                    "margin-top": "1.2rem",
                    "background-color":'#4e505a'
                },
                accept="image/*",
            ),
    )

#card_filters
DIV_DICT["card_filters"]="card_filters"
def card_filters():
    return Card(
                CustomDropdown(
                    id=DIV_DICT["card_filters"],
                    options=[
                        {"label": "模糊化", "value": "blur"},
                        {"label": "轮廓提取", "value": "contour"},
                        #{"label": "Detail", "value": "detail"},
                        {"label": "边缘增强", "value": "edge_enhance"},
                        #{
                        #    "label": "Enhance Edge (More)",
                        #    "value": "edge_enhance_more",
                        #},
                        #{"label": "Emboss", "value": "emboss"},
                        {"label": "边缘提取", "value": "find_edges"},
                        {"label": "锐化处理", "value": "sharpen"},
                        {"label": "平滑处理", "value": "smooth"},
                        #{"label": "Smooth (More)", "value": "smooth_more"},
                    ],
                    searchable=True,
                    placeholder="图像操作",
                    multi=False,
                )
            )

#button_group#
DIV_DICT["button_group"]="button_group"
DIV_DICT["button_run_operation"]="button_run_operation"
DIV_DICT["button_undo"]="button_undo"
def button_group():
    return html.Div(
                id=DIV_DICT["button_group"],
                children=[
                    html.Button(
                        "执行操作", 
                        id=DIV_DICT["button_run_operation"]
                        ),
                    html.Button(
                        "撤回", 
                        id=DIV_DICT["button_undo"]
                        ),
                ],
            )


#cards-image-analysis#
DIV_DICT["cards_image_anlaysis"]="cards_image_anlaysis"
def cards_image_analysis():
    return html.Div(
        id=DIV_DICT["cards_image_anlaysis"],
        children=[
            card_select_plot(),
            plot_graph(),
        ],
    )

#card_select_plot#
DIV_DICT["card_select_plot"]="card_select_plot"
DIV_DICT["dropdown_select_plot"]="dropdown_select_plot"
def card_select_plot():
    return Card(
        id=DIV_DICT["card_select_plot"],
        children=[
            CustomDropdown(
                        id=DIV_DICT["dropdown_select_plot"],
                        options=[
                            {"label": "直方图", "value": "histogram"},
                            {"label": "DFT", "value": "DFT"},
                            {"label": "DCT", "value": "DCT"},
                            {"label": "DWT", "value": "DWT"},
                        ],
                        value='histogram',
                        searchable=True,
                        placeholder="图像分析",
                    ),
        ],
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










