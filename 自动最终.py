import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops
import joblib
import time
import warnings

warnings.filterwarnings('ignore')
class PicCut(object):
    def __init__(self, ori_file_path, processed_file_path):
        self.ori_file_path = ori_file_path
        self.processed_file_path = processed_file_path

    def pic_spilt(self, img):
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (500, 622, 1920, 1080)  # 根据图片尺寸调整
        mask, bgdModel, fgdModel = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        img_spilt = img * mask2[:, :, np.newaxis]
        return img_spilt

    def pic_cut_for_single(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            return
        img_spilt = self.pic_spilt(img)
        filename = os.path.basename(img_path)
        save_path = os.path.join(self.processed_file_path, filename)
        cv2.imwrite(save_path, img_spilt)
        print(f"图片 {filename} 处理并保存成功")


class ExtractFeature(object):
    def __init__(self, image_path):
        self.image_path = image_path

    def yel_proportion(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([11, 43, 46]), np.array([26, 255, 255]))
        yellow_pixCnt = np.sum(mask == 255)
        roi_pixCnt = np.sum(mask >= 0)
        proportion = round(np.true_divide(yellow_pixCnt, roi_pixCnt), 4)
        return proportion

    def process_single_image(self):
        img = cv2.imread(self.image_path)
        if img is None:
            print("图片无法打开，请检查路径是否正确：" + self.image_path)
            return
        proportion = self.yel_proportion(img)
        data = {'Image_Path': [self.image_path], 'yel_per': [proportion]}
        df = pd.DataFrame(data)
        try:
            existing_df = pd.read_excel('file.xlsx')
            if not existing_df.empty:
                df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass
        df.to_excel('file.xlsx', index=False)
        print("黄化特征值已保存到Excel文件中：", proportion)


def calculate_color_space_means(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法找到或加载图像：{img_path}")

        # 转换到Lab颜色空间
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a_mean = np.mean(lab_img[:, :, 1])
    b_mean = np.mean(lab_img[:, :, 2])

    # 转换到YCbCr颜色空间
    ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cb_mean = np.mean(ycbcr_img[:, :, 1])
    cr_mean = np.mean(ycbcr_img[:, :, 2])

    # 计算绿色通道的均值（MeanG）
    mean_g = np.mean(img[:, :, 1])

    data = {
        'a': a_mean,
        'b': b_mean,
        'cb': cb_mean,
        'cr': cr_mean,
        'MeanG': mean_g
    }

    return data


def compute_glcm(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法找到或加载图像：{img_path}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray[200:750, 600:1600]
    glcm = graycomatrix(img_gray, [1], [0], levels=256, symmetric=True, normed=True)
    glcm_features = {prop: graycoprops(glcm, prop).flatten()[0] for prop in
                     ['contrast', 'dissimilarity', 'correlation']}
    return glcm_features

def calculate_max_vertical_length(img_path):
    # 新增函数，计算最大垂直长度
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    max_vertical_length = 0
    height, width = binary_image.shape
    for col in range(width):
        current_length = 0
        for row in range(height):
            if binary_image[row, col] == 255:
                current_length += 1
                max_vertical_length = max(max_vertical_length, current_length)
            else:
                current_length = 0
    return {'max_vertical_length': max_vertical_length}


def main(image_path, excel_path, model_path):
    start_time = time.time()
    processed_file_path = "processed_images"
    os.makedirs(processed_file_path, exist_ok=True)
    pc = PicCut(None, processed_file_path)
    pc.pic_cut_for_single(image_path)
    processed_image_path = os.path.join(processed_file_path, os.path.basename(image_path))

    df = pd.read_excel(excel_path)

    environment_features = {
        'dry_up': df['dry_up'].iloc[0],
        'wet_up': df['wet_up'].iloc[0],
        'baketime': df['baketime'].iloc[0],
    }
    water_features = {
        'driage': ((12.2604 - df['重量'].iloc[0]) / 12.2604),
        'waterloss_rate': ((12.2604 - df['重量'].iloc[0]) / df['baketime'].iloc[0])
    }

    water_features_values = np.array(list(water_features.values()))
    environment_features_values = np.array(list(environment_features.values()))

    # 加载第一层模型并预测
    status_2_water = joblib.load(r"E:\01\2tihuan\model\bayes_2_water.pkl")
    status_2_environment = joblib.load(r"E:\01\2tihuan\model\bayes_2_environment.pkl")

    water_pred_proba = status_2_water.predict_proba(water_features_values.reshape(1, -1))
    environment_pred_proba = status_2_environment.predict_proba(environment_features_values.reshape(1, -1))
    first_layer_pred_proba = water_pred_proba * 0.5 + environment_pred_proba * 0.5
    first_layer_pred = np.argmax(first_layer_pred_proba, axis=1) + 1

    # 根据第一层预测结果决定后续处理
    if first_layer_pred in [1, 2, 3, 4]:
        # 计算颜色特征
        color_features = calculate_color_space_means(processed_image_path)
        ef = ExtractFeature(processed_image_path)
        ef_features = {'yel_per': ef.yel_proportion(cv2.imread(processed_image_path))}
        color_features.update(ef_features)
        color_features_array = np.array(list(color_features.values()))

        # 加载颜色模型并预测
        status_2_color = joblib.load(r"E:\01\2tihuan\model\bayes_2_color.pkl")
        color_pred_proba = status_2_color.predict_proba(color_features_array.reshape(1, -1))

        # 结合第一层预测结果
        weights = [6, 4]
        total = (color_pred_proba * weights[0] + first_layer_pred_proba * weights[1]) / (weights[0] + weights[1])
    elif first_layer_pred in [5, 6, 7]:
        # 计算纹理特征
        glcm_features = compute_glcm(processed_image_path)
        glcm_features_values = np.array(list(glcm_features.values()))

        # 加载纹理模型并预测
        status_2_texture = joblib.load(r"E:\01\2tihuan\model\bayes_2_texture.pkl")
        texture_pred_proba = status_2_texture.predict_proba(glcm_features_values.reshape(1, -1))

        # 结合第一层预测结果
        weights = [6, 4]
        total = (texture_pred_proba * weights[0] + first_layer_pred_proba * weights[1]) / (weights[0] + weights[1])
    elif first_layer_pred in [8, 9]:
        # 计算长度特征
        max_vertical_length_dict = calculate_max_vertical_length(processed_image_path)
        max_vertical_length = max_vertical_length_dict['max_vertical_length']
        original_length = 1810
        length_features = {
            'length': (original_length - max_vertical_length) / original_length,
            'length_rate': (12.2604 - max_vertical_length) / df['baketime'].iloc[0]
        }
        length_features_values = np.array(list(length_features.values()))

        # 加载长度模型并预测
        status_2_length = joblib.load(r"E:\01\2tihuan\model\bayes_2_length.pkl")
        length_pred_proba = status_2_length.predict_proba(length_features_values.reshape(1, -1))

        # 权值3:3:4
        weights = [6, 4]
        total = (length_pred_proba * weights[0] + first_layer_pred_proba * weights[1]) / (weights[0] + weights[1])
    else:
        # 计算所有特征
        color_features = calculate_color_space_means(processed_image_path)
        ef = ExtractFeature(processed_image_path)
        ef_features = {'yel_per': ef.yel_proportion(cv2.imread(processed_image_path))}
        color_features.update(ef_features)
        color_features_array = np.array(list(color_features.values()))

        glcm_features = compute_glcm(processed_image_path)
        glcm_features_values = np.array(list(glcm_features.values()))

        max_vertical_length_dict = calculate_max_vertical_length(processed_image_path)
        max_vertical_length = max_vertical_length_dict['max_vertical_length']
        original_length = 1810
        length_features = {
            'length': (original_length - max_vertical_length) / original_length,
            'length_rate': (12.2604 - max_vertical_length) / df['baketime'].iloc[0]
        }
        length_features_values = np.array(list(length_features.values()))

        # 加载所有模型并预测
        status_2_color = joblib.load(r"E:\01\2tihuan\model\bayes_2_color.pkl")
        status_2_texture = joblib.load(r"E:\01\2tihuan\model\bayes_2_texture.pkl")
        status_2_length = joblib.load(r"E:\01\2tihuan\model\bayes_2_length.pkl")

        color_pred_proba = status_2_color.predict_proba(color_features_array.reshape(1, -1))
        texture_pred_proba = status_2_texture.predict_proba(glcm_features_values.reshape(1, -1))
        length_pred_proba = status_2_length.predict_proba(length_features_values.reshape(1, -1))

        # 权值1:1:2:2:2 (总权重8)
        # 结合第一层预测结果
        weights = [3, 3, 3, 1]
        total = (color_pred_proba * weights[0] +
                 texture_pred_proba * weights[1] +
                 length_pred_proba * weights[2] +
                 first_layer_pred_proba * weights[3]) / sum(weights)

    predicted_labels = np.argmax(total, axis=1) + 1
    end_time = time.time()

    # 输出最终预测结果
    print(f"预测的烘烤阶段为: {predicted_labels}")
    # 输出运行时间和性能指标
    print(f"Model run time: {end_time - start_time} seconds")


# 替换为你的图片路径、Excel路径和模型路径
image_path = r"E:\01\tobacco\224901.jpg"
excel_path = r"E:\01\tobacco\123.xlsx"
model_path = r"E:\01\tobacco\model"

main(image_path, excel_path, model_path)