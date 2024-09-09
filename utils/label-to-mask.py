import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils

'''
制作自己的语义分割数据集需要注意以下几点：
1、我使用的labelme版本是3.16.7，建议使用该版本的labelme，有些版本的labelme会发生错误，
   具体错误为：Too many dimensions: 3 > 2
   安装方式为命令行pip install labelme==3.16.7
2、此处生成的标签图是8位彩色图，与视频中看起来的数据集格式不太一样。
   虽然看起来是彩图，但事实上只有8位，此时每个像素点的值就是这个像素点所属的种类。
   所以其实和视频中VOC数据集的格式一样。因此这样制作出来的数据集是可以正常使用的。也是正常的。
'''
if __name__ == '__main__':
    jpgs_path = r"E:\projects\ultralytics-8.2.0\Datasets\stone_imgs\imgs"   # 原图片位置
    pngs_path = r"C:\Users\15983\Desktop\2_gai"   # 保存mask图片位置
    json_path = r"E:\projects\ultralytics-8.2.0\Datasets\stone_imgs\json"   # labelme文件
    # classes     = ["_background_","person", "car", "motorbike", "dustbin","chair","fire_hydrant","tricycle","bicycle","stone"]
    classes = ["_background_", "inorganic matter", "organic pore", "organic matter"]

    count = os.listdir(json_path)
    # 遍历数据集
    for i in range(0, len(count)):
        path = os.path.join(json_path, count[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))

            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))
            try:
                lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
                # 返回标签图、标签名称
                # 取标签图
                lbl = lbl[0]
            except AssertionError as e:
                # 如果发生 AssertionError，打印错误消息并跳过当前迭代
                print(f"发生错误：{e}")
                print(f"跳过文件：{count[i]}")
                continue  # 跳过当前文件，继续处理下一个文件

            # PIL.Image.fromarray(img).save(osp.join(pngs_path, count[i].split(".")[0] + '.png'))

            new = np.zeros([np.shape(img)[0], np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all * (np.array(lbl) == index_json)

            # utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)
            # print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')

            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0] + '.png'), new)
            print('Saved ' + count[i].split(".")[0] + '.png')
