import cv2
import argparse
import numpy as np
import segment_anything
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='|pth模型推理|')
parser.add_argument('--image_path', default='demo.jpg', type=str, help='|图片位置|')
parser.add_argument('--checkpoint', default='vit_l.pth', type=str, help='|模型位置|')
parser.add_argument('--model_type', default='vit_l', type=str, help='|型号|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
parser.add_argument('--segment_all', default=False, type=bool, help='|True时分割整张图，False时根据提示点分割|')
parser.add_argument('--multimask', default=False, type=bool, help='|根据提示点分割，True时输出3个得分最高的掩码，False时输出1个|')
parser.add_argument('--input_point', default=[[260, 200]], type=list, help='|根据提示点所在图层分割图片，如[[260, 200],...]|')
parser.add_argument('--input_label', default=[1], type=list, help='|表示input_point是前景点(1)还是背景点(0)，如[1,...]|')
args = parser.parse_args()


def draw_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def draw_point(coords, labels, ax, marker_size=375):  # 用五角星显示提示点在图片中的位置
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def draw_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# 模型
if __name__ == '__main__':
    # 模型
    model = segment_anything.sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    # 数据
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 分割
    if args.segment_all:  # 分割全图
        print('| 分割全图 |')
        # 推理
        predictor = segment_anything.SamAutomaticMaskGenerator(model)
        masks = predictor.generate(image)
        print(f'| len(masks):{len(masks)} |')
        print(f'| masks[0].keys():{masks[0].keys()} |')
    else:  # 分割提示点
        print(f'| 分割提示点:{args.input_point} 标签为:{args.input_label} |')
        # 提示点
        input_point = np.array(args.input_point)
        input_label = np.array(args.input_label)
        plt.imshow(image)
        draw_point(input_point, input_label, plt.gca())
        plt.savefig('input_point.jpg')
        print(f'| 保存图片:input_point.jpg |')
        # 推理(multimask_output=True输出3个，False输出1个)
        predictor = segment_anything.SamPredictor(model)
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label,
                                             multimask_output=args.multimask)
        # print(f'| masks.shape:{masks.shape},{masks.dtype} |')  # masks:(1/3,h,w)
        # print(f'| scores.shape:{scores.shape},{scores.dtype} |')  # scores:(1/3,)
        # 画图
        for i, (mask, score) in enumerate(zip(masks, scores)):
            name = f"Mask_{i + 1}__Score_{score:.3f}"
            plt.title(name)
            plt.imshow(image)
            draw_mask(mask, plt.gca())
            draw_point(input_point, input_label, plt.gca())
            plt.savefig(f"{name}.jpg")
            print(f'| 保存图片:{name} |')
