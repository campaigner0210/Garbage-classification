import torch
from PIL import Image
import os
import glob
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms
from PIL import ImageFile
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


# 读取 train.txt 文件，加载数据。
# 数据预处理,将图像resize 到 160 * 160 的尺寸（符合resnet101输入）。

class Garbage_Loader(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        # print(self.imgs_info)
        self.train_flag = train_flag

        self.train_tf = transforms.Compose([
            transforms.Resize(160),
            # 添加了水平翻转和垂直翻转的随机操作，进行数据增强
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),

        ])
        self.val_tf = transforms.Compose([
            transforms.Resize(160),
            transforms.ToTensor(),
        ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info)) # lambda函数用于指定对imgs_info列表中每一个元素的共同操作
        return imgs_info

    def padding_black(self, img):

        w, h = img.size

        scale = 160. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]]) # 等比压缩

        size_fg = img_fg.size
        size_bg = 160

        img_bg = Image.new("RGB", (size_bg, size_bg))

        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img

    # 当实例对象做P[key]运算时，就会调用类中的__getitem__()方法。
    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB') # 去除透明通道
        img = self.padding_black(img)
        print(img.size)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.imgs_info)


if __name__ == "__main__":
    train_dataset = Garbage_Loader("train.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(label)
