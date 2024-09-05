import cv2
import json
import os

def splitImgandJson(IPath_json, IPath_img, OPath_json, OPath_img):
    events = os.listdir(IPath_json)

    # 循环处理每一张图像和标注
    for onevent in events:
        name = onevent.split('.json')[0]
        img_name = name + '.jpg'

        with open(os.path.join(IPath_json, onevent), 'r') as json_file:
            large_json_data = json.load(json_file)

        # 读取大图的JPEG图像
        large_image = cv2.imread(os.path.join(IPath_img, img_name))

        # 定义每个小图的大小
        crop_width, crop_height = 320, 320  # 可以根据需要调整大小

        # 循环裁剪图像并生成新的JSON文件
        L_H = large_image.shape[0]
        L_W = large_image.shape[1]
        for i in range(0, L_H, crop_height):
            for j in range(0, L_W, crop_width):
                # 裁剪图像
                small_image = large_image[i:i + crop_height, j:j + crop_width]

                # 创建新的JSON数据，基于大图的JSON数据进行调整
                small_json_data = {"shapes": [], "imageWidth": crop_width, "imageHeight": crop_height}

                # 遍历大图标注看是否存在在小图中
                for item in large_json_data["shapes"]:
                    points = item["points"]
                    new_points = []

                    # 检查多边形的每个点是否在小图的范围内
                    for x, y in points:
                        if (j <= x < j + crop_width) and (i <= y < i + crop_height):
                            new_x = x - j
                            new_y = y - i
                            new_points.append([new_x, new_y])

                    # 如果小图中有该多边形的点，则添加到小图的JSON数据中
                    if new_points:
                        new_item = {
                            "label": item["label"],
                            "points": new_points,
                            "shape_type": item["shape_type"]
                        }
                        small_json_data["shapes"].append(new_item)

                # 保存小图的JSON文件
                small_json_filename = f'{name}_{i}_{j}.json'
                with open(os.path.join(OPath_json, small_json_filename), 'w') as small_json_file:
                    json.dump(small_json_data, small_json_file, indent=4)

                # 保存小图的JPEG图像
                small_image_filename = f'{name}_{i}_{j}.jpg'
                cv2.imwrite(os.path.join(OPath_img, small_image_filename), small_image)

if __name__ == '__main__':
    IPath_json = 'E:\\projects\\ultralytics-8.2.0\\Datasets\\stone_imgs\\json'
    IPath_img = 'E:\\projects\\ultralytics-8.2.0\\Datasets\\stone_imgs\\imgs'
    OPath_img = r'E:\projects\ultralytics-8.2.0\Datasets\stone_imgs_s320\imgs_s320'
    OPath_json = r'E:\projects\ultralytics-8.2.0\Datasets\stone_imgs_s320\json_s320'

    if not os.path.exists(OPath_json):
        os.mkdir(OPath_json)
    if not os.path.exists(OPath_img):
        os.mkdir(OPath_img)

    splitImgandJson(IPath_json, IPath_img, OPath_json, OPath_img)