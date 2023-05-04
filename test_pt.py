# python export_onnx.py --checkpoint vit_l.pth --model-type vit_l --output sam.onnx
import cv2
import argparse
import numpy as np
import segment_anything
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')
parser.add_argument('--image_path', default='demo.jpg', type=str, help='|图片位置|')
parser.add_argument('--weight', default='vit_l.pth', type=str, help='|模型位置|')
parser.add_argument('--model_type', default='vit_l', type=str, help='|型号|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
args = parser.parse_args()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):  # 用五角星显示提示点在图片中的位置
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# 模型
if __name__ == '__main__':
    model = segment_anything.sam_model_registry[args.model_type](checkpoint=args.weight).to(args.device)
    predictor = segment_anything.SamPredictor(model)
    # 数据
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    # 提示点
    input_point = np.array([[260, 200]])
    input_label = np.array([1])
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.savefig('show_points.jpg')
    # 预测(multimask_output=True输出3个，False输出1个)
    masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
    print(f'| masks.shape:{masks.shape},{masks.dtype} |')
    print(f'| scores.shape:{scores.shape},{scores.dtype} |')
    print(f'| logits.shape:{logits.shape},{logits.dtype} |')
    # 画图
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask_{i + 1}__Score_{score:.3f}")
        plt.savefig(f"Mask_{i + 1}__Score_{score:.3f}.jpg")
