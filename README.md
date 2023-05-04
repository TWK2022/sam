## 快速建立图片特征数据库并用文本描述进行搜索
>基于segment_anything官方项目改编:https://github.com/facebookresearch/segment-anything  
>官方示例:https://segment-anything.com/  
>采用国内配套中文模型：https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese  
>首次运行代码时会自动下载模型文件到本地clip和huggingface文件夹中，大小在1G左右  
### SAM介绍
>2023年facebook发布的图像分割模型(pytorch)，能对图片进行分割，可用于抠图等  
>有vit_b(358M)、vit_l(1.16G)、vit_h(2.39G)三个型号，要单独下载模型  
### 项目介绍
>本项目将建立图片特征数据库，然后用文本描述(75字以内，长的会被截断)去搜索符合描述的图片  
### 1，database_prepare.py
>将数据库图片放入文件夹image_database中  
>运行database_prepare.py即可生成特征数据库feature_database.csv  
### 2，predict.py
>在english_text、chinese_text中输入英文、中文文本，运行程序后可以搜到数据库中符合文本描述的图片  
>args中english_score_threshold、chinese_score_threshold为匹配的得分筛选阈值，17、12为基准，可适当调整  
### 其他
>github链接:https://github.com/TWK2022/clip  
>学习笔记:https://github.com/TWK2022/notebook  
>邮箱:1024565378@qq.com  