import cv2
import argparse
import numpy as np
import onnxruntime
import segment_anything
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='|onnx模型推理|')
parser.add_argument('--image_path', default='demo.jpg', type=str, help='|图片位置|')
parser.add_argument('--checkpoint', default='vit_h.pth', type=str, help='|模型位置|')
parser.add_argument('--model_type', default='vit_h', type=str, help='|型号|')
parser.add_argument('--onnx_model_path', default='sam_part.onnx', type=str, help='|模型位置|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
parser.add_argument('--input_point', default=[[260, 200]], type=list, help='|根据提示点所在图层分割图片，如[[260, 200],...]|')
parser.add_argument('--input_label', default=[1], type=list, help='|表示input_point是要分割的点还是抑制的点，1表示分割，0表示抑制，如[1,...]|')
args = parser.parse_args()


def draw_mask(mask, ax):
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
    # onnx部分
    provider = 'CUDAExecutionProvider' if args.device.lower() in ['gpu', 'cuda'] else 'CPUExecutionProvider'
    onnx_model = onnxruntime.InferenceSession(args.onnx_model_path, providers=[provider])  # 加载模型和框架
    # 数据
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 提示点
    input_point = np.array(args.input_point)
    input_label = np.array(args.input_label)
    # 分割
    predictor = segment_anything.SamPredictor(model)
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }
    masks, scores, _ = onnx_model.run(None, ort_inputs)
    # print(f'| masks:{masks.shape},{masks.dtype} scores:{scores.shape},{scores.dtype} |')
    # 画图
    for mask_batch, score_batch in zip(masks, scores):
        for i, (mask, score) in enumerate(zip(mask_batch, score_batch)):
            name = f"Mask_{i + 1}__Score_{score:.3f}"
            plt.title(name)
            plt.imshow(image)
            draw_mask(mask, plt.gca())
            draw_point(input_point, input_label, plt.gca())
            plt.savefig(f"{name}.jpg")
            print(f'| 保存图片:{name} |')
