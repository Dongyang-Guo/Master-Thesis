
# version1

# from pycocotools.coco import COCO
#
# dataDir = './data/coco'
# dataType = 'val'
# # dataType='train2017'
# annFile = '{}/annotations/instance_{}.json'.format(dataDir, dataType)
#
# # initialize COCO api for instance annotations
# coco = COCO(annFile)
#
# # display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# cat_nms = [cat['name'] for cat in cats]
# print('number of categories: ', len(cat_nms))
# print('COCO categories: \n', cat_nms)
#
# # 统计各类的图片数量和标注框数量
# for cat_name in cat_nms:
#     catId = coco.getCatIds(catNms=cat_name)  # 1~90
#     imgId = coco.getImgIds(catIds=catId)  # 图片的id
#     annId = coco.getAnnIds(catIds=catId)  # 标注框的id
#
#     print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))

# version2

import json
import os

def cal_categories_num(img_path,json_path):
    label_list = []
    img_list = os.listdir(img_path)
    for img in img_list:
        json_info = json.load(open(os.path.join(json_path,img.split('.')[0] + '.json'),encoding='gbk'))
        labels = json_info['shapes']
        for label in labels:
            category = label['label']
            label_list.append(category)
    return label_list



if __name__ == '__main__':
    img_path = 'data/all/img'
    json_path = 'data/all/json'
    label_list = cal_categories_num(img_path,json_path)

    set=set(label_list)
    print(set)
    dict={}
    for item in set:
        dict.update({item:label_list.count(item)})
    print(dict)

