import streamlit as st
import pandas as pd
import urllib.request
import imghdr
import os

#超像素图像处理库
import cv2
import numpy as np
from skimage import segmentation, color
from skimage.future import graph
from skimage.filters import sobel
from skimage.color import rgb2gray

SIDEBAR_OPTIONS = ["项目信息", "上传图片", "使用预置图片"]
IMAGE_DIR = 'Image'

def get_file_content_as_string(path):
    # url = 'https://gitee.com/wu_jia_sheng/graduation_program/blob/master/' + path
    url = 'https://raw.githubusercontent.com/Alex0Stephen/Superpixel_Project/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

@st.cache()
def Ncut_Process(img):
    labels_slic0 = segmentation.slic(img, slic_zero=True)
    g = graph.rag_mean_color(img, labels_slic0, mode='similarity')
    ncuts_labels_slic0 = graph.cut_normalized(labels_slic0, g)
    out = color.label2rgb(ncuts_labels_slic0, img, kind='avg')
    return out

@st.cache()
def Ncut_Mark_Process(img):
    labels_slic0 = segmentation.slic(img, slic_zero=True)
    g = graph.rag_mean_color(img, labels_slic0, mode='similarity')
    ncuts_labels_slic0 = graph.cut_normalized(labels_slic0, g)
    out = segmentation.mark_boundaries(img, ncuts_labels_slic0)
    return out

@st.cache()
def Watershed_Process(img, Markers, Compactness):
    gradient = sobel(rgb2gray(img))
    segments_watershed = segmentation.watershed(gradient, markers = Markers, compactness = Compactness)
    out = color.label2rgb(segments_watershed, img, kind='avg')
    return out

@st.cache()
def Watershed_Mark_Process(img, Markers, Compactness):
    gradient = sobel(rgb2gray(img))
    segments_watershed = segmentation.watershed(gradient, markers = Markers, compactness = Compactness)
    out = segmentation.mark_boundaries(img, segments_watershed)
    return out

@st.cache()
def Quickshift_Process(img, Kernel_size, Max_dist, Ratio):
    segments_quick = segmentation.quickshift(img, kernel_size = Kernel_size, max_dist = Max_dist, ratio = Ratio)
    out = color.label2rgb(segments_quick, img, kind='avg')
    return out

@st.cache()
def Quickshift_Mark_Process(img, Kernel_size, Max_dist, Ratio):
    segments_quick = segmentation.quickshift(img, kernel_size = Kernel_size, max_dist = Max_dist, ratio = Ratio)
    out = segmentation.mark_boundaries(img, segments_quick)
    return out

@st.cache()
def SLIC_Process(img, N_segments, Compactness):
    labels_slic0 = segmentation.slic(img, slic_zero=True, n_segments = N_segments, compactness = Compactness)     
    out = color.label2rgb(labels_slic0, img, kind = 'avg')
    return out

@st.cache()
def SLIC_Mark_Process(img, N_segments, Compactness):
    labels_slic0 = segmentation.slic(img, slic_zero=True, n_segments = N_segments, compactness = Compactness)     
    out = segmentation.mark_boundaries(img, labels_slic0)
    return out

def Display_result(out, img_boundaries):
    st.title("超像素图像分割结果：")
    left_column, right_column = st.columns(2)
    left_column.image(out, caption = "分割结果图")
    right_column.image(img_boundaries,  caption = "分割边框图")

    

if __name__ == '__main__':
    st.set_page_config(page_title="Welcome To MY Project",page_icon=":rainbow:")

    if 'first_visit' not in st.session_state:
        st.session_state.first_visit=True
    else:
        st.session_state.first_visit=False

    if st.session_state.first_visit:
       st.balloons()

    st.sidebar.warning('请上传图片')
    st.sidebar.write(" ------ ")
    st.sidebar.title("让我们来一起探索吧")

    app_mode = st.sidebar.selectbox("请从下列选项中选择您想要的功能", SIDEBAR_OPTIONS)

    st.title('Welcome To MY Project')
    st.write(" ------ ")

    if app_mode == "项目信息":
        st.sidebar.write(" ------ ")
        st.sidebar.success("项目信息请往右看!")
        st.write(get_file_content_as_string("Project-Info.md"))

    elif app_mode == "上传图片":
        st.sidebar.write(" ------ ")
        file = st.file_uploader('上传图片')
        if file:
           file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
           image = cv2.imdecode(file_bytes, 1)
           img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

           st.title("所选图片展示：")
           img = cv2.resize(img,(300,300))
           st.image(img)

           dtype_algorithm = {
              'N-Cut' : 'ncut',
              'Watershed' : 'watershed',
              'Quick shift' : 'quickshift',
              'SLIC' : 'slic',
           }
           algorithm_split_names = list(dtype_algorithm.keys())

           algorithm_type = st.sidebar.selectbox("算法类别：", algorithm_split_names)
           algorithms_type = dtype_algorithm[algorithm_type]
           
           if algorithms_type == 'watershed':
                markers = st.sidebar.number_input('标记数：', min_value=1, max_value=1000, value=150)
                compactness = st.sidebar.number_input('紧凑度：', min_value=0.0, max_value=1.0, value=0.01)
           elif algorithms_type == 'quickshift':
                kernel_size = st.sidebar.number_input('核函数宽度（用于平滑样本宽度）：', min_value=1, max_value=10000, value=10)
                max_dist = st.sidebar.number_input('最大距离（用于控制聚类数量）', min_value=1, max_value=100, value=10)
                ratio = st.sidebar.number_input('比率（用于平衡色彩空间和图像空间的相似性）：', min_value=0.0, max_value=1.0, value=1.0)
           elif algorithms_type == 'slic':
                n_segments = st.sidebar.number_input('超像素数量：', min_value=1, max_value=10000, value=100)
                compactness = st.sidebar.number_input('超像素紧凑性：', min_value=0, max_value=100, value=10)

            
           pressed = st.sidebar.button('确 定')

           if pressed:
                st.empty()
                st.sidebar.write('请稍等! 你知道的，这通常需要一点时间。')
                
                if algorithms_type == 'ncut':
                    out = Ncut_Process(img)
                    img_boundaries = Ncut_Mark_Process(img)
                    Display_result(out, img_boundaries)
                elif algorithms_type == 'watershed':
                    out = Watershed_Process(img, markers, compactness)
                    img_boundaries = Watershed_Mark_Process(img, markers, compactness)
                    Display_result(out, img_boundaries)
                elif algorithms_type == 'quickshift':
                    out = Quickshift_Process(img, kernel_size, max_dist, ratio)
                    img_boundaries = Quickshift_Mark_Process(img, kernel_size, max_dist, ratio)
                    Display_result(out, img_boundaries)
                elif algorithms_type == 'slic':
                    out = SLIC_Process(img, n_segments, compactness)
                    img_boundaries = SLIC_Mark_Process(img, n_segments, compactness)
                    Display_result(out, img_boundaries)

        # else:
        #     st.warning("上传图片失败!")

    elif app_mode == "使用预置图片":
        directory = os.path.join(IMAGE_DIR)

        photos = []
        for file in os.listdir(directory):
            filepath = os.path.join(directory, file)

            # Find all valid images
            if imghdr.what(filepath) is not None:
                photos.append(file)

        photos.sort()

        st.sidebar.write(" ------ ")
        option = st.sidebar.selectbox('请选择一张预置的图片，然后点击按钮', photos)

        dtype_algorithm = {
              'N-Cut' : 'ncut',
              'Watershed' : 'watershed',
              'Quick shift' : 'quickshift',
              'SLIC' : 'slic',
            }
        algorithm_split_names = list(dtype_algorithm.keys())

        algorithm_type = st.sidebar.selectbox("算法类别：", algorithm_split_names)
        algorithms_type = dtype_algorithm[algorithm_type]

        if algorithms_type == 'watershed':
                markers = st.sidebar.number_input('标记数：', min_value=1, max_value=1000, value=150)
                compactness = st.sidebar.number_input('紧凑度：', min_value=0.0, max_value=1.0, value=0.01)
        elif algorithms_type == 'quickshift':
                kernel_size = st.sidebar.number_input('核函数宽度（用于平滑样本宽度）：', min_value=1, max_value=10000, value=10)
                max_dist = st.sidebar.number_input('最大距离（用于控制聚类数量）', min_value=1, max_value=100, value=10)
                ratio = st.sidebar.number_input('比率（用于平衡色彩空间和图像空间的相似性）：', min_value=0.0, max_value=1.0, value=1.0)
        elif algorithms_type == 'slic':
                n_segments = st.sidebar.number_input('超像素数量：', min_value=1, max_value=10000, value=100)
                compactness = st.sidebar.number_input('超像素紧凑性：', min_value=0, max_value=100, value=10)

        pressed = st.sidebar.button('确 定')

        if pressed:
            st.empty()
            st.sidebar.write('请稍等! 你知道的，这通常需要一点时间。')
            pic = os.path.join(directory, option)

            image = cv2.imread(pic, cv2.IMREAD_COLOR)
            img = image[:,:,[2,1,0]]

            st.title("所选图片展示：")
            img = cv2.resize(img,(300,300))
            st.image(img)

            if algorithms_type == 'ncut':
                out = Ncut_Process(img)
                img_boundaries = Ncut_Mark_Process(img)
                Display_result(out, img_boundaries)
            elif algorithms_type == 'watershed':
                out = Watershed_Process(img, markers, compactness)
                img_boundaries = Watershed_Mark_Process(img, markers, compactness)
                Display_result(out, img_boundaries)
            elif algorithms_type == 'quickshift':
                out = Quickshift_Process(img, kernel_size, max_dist, ratio)
                img_boundaries = Quickshift_Mark_Process(img, kernel_size, max_dist, ratio)
                Display_result(out, img_boundaries)
            elif algorithms_type == 'slic':
                out = SLIC_Process(img, n_segments, compactness)
                img_boundaries = SLIC_Mark_Process(img, n_segments, compactness)
                Display_result(out, img_boundaries)


