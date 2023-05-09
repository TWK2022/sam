## 快速部署图像分割模型
>基于segment_anything官方项目改编：https://github.com/facebookresearch/segment-anything  
>官方示例：https://segment-anything.com/  
>  
### SAM介绍
>2023年facebook发布的图像分割模型(pytorch)，能对图片进行分割，可用于抠图等  
>有vit_b(358M)、vit_l(1.16G)、vit_h(2.39G)三个型号，要单独下载模型  
### 项目介绍
>本项目将部署和运行SAM模型  
### 1，下载模型到本地
>vit_b()：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth  
>vit_l()：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth  
>vit_h()：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  
### 2，test_pth.py
>运行模型
>```
>python test_pth.py --image_path demo.jpg --checkpoint vit_l.pth --model_type vit_l --device cuda
>```
### 3，export_onnx.py
>导出模型为onnx
>```
>python export_onnx.py --checkpoint vit_l.pth --model_type vit_l --device cuda
>```
### 其他
>github链接：https://github.com/TWK2022/  
>学习笔记：https://github.com/TWK2022/notebook  
>邮箱：1024565378@qq.com  