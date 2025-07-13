from PIL import Image
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# COCO类别定义
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# 数据处理管道
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载模型
print("Loading model...")
model, postprocessor = torch.hub.load(
    'facebookresearch/detr',
    'detr_resnet101_panoptic',
    pretrained=True,
    return_postprocessor=True,
    num_classes=250
)
model = model.to(device).eval()
print("Model loaded successfully.")

# 加载图像
im_path = r"C:\03_code\code_transformer\detr\demo\images\000000000285.jpg"
im = Image.open(im_path)
original_size = im.size

# 处理图像
print("Processing image...")
img = transform(im).unsqueeze(0).to(device)

# 推理
print("Running inference...")
out = model(img)

# 筛选高置信度结果
scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
keep = scores > 0.85
n_keep = keep.sum().item()
print(f"Found {n_keep} segments with confidence > 0.85")

# 创建最终的分割结果图
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(im)
ax.axis('off')

# 获取掩码数据
if n_keep > 0:
    masks = out["pred_masks"][keep].cpu().numpy()

    # 为每个掩码创建半透明覆盖层
    overlay = np.zeros((*im.size[::-1], 4), dtype=np.float32)

    # 为每个类别分配随机颜色
    colors = plt.cm.tab20(np.linspace(0, 1, n_keep))

    for i in range(n_keep):
        # 获取掩码并缩放到原始图像尺寸
        mask = masks[i]
        mask_img = Image.fromarray(mask).resize(im.size, Image.BILINEAR)
        mask_array = np.array(mask_img)

        # 获取类别信息
        class_idx = out["pred_logits"][keep][i].softmax(-1)[:-1].argmax().item()
        class_name = CLASSES[class_idx] if class_idx < len(CLASSES) else f"Class {class_idx}"

        # 计算掩码的中心点用于放置标签
        y_indices, x_indices = np.where(mask_array > 0.5)
        if len(y_indices) > 0 and len(x_indices) > 0:
            center_x = int(np.mean(x_indices))
            center_y = int(np.mean(y_indices))

            # 在中心点添加标签
            ax.text(
                center_x, center_y, class_name,
                fontsize=10, color='white',
                bbox=dict(facecolor='black', alpha=0.7, pad=1, edgecolor='none'),
                ha='center', va='center'
            )

        # 添加半透明颜色覆盖
        color = colors[i]
        for c in range(3):  # RGB通道
            overlay[..., c] += mask_array * color[c]
        overlay[..., 3] += mask_array * 0.5  # Alpha通道

    # 限制alpha值在0-1之间
    overlay[..., 3] = np.clip(overlay[..., 3], 0, 1)

    # 显示分割覆盖层
    ax.imshow(overlay, alpha=0.5)

# 添加标题
ax.set_title(f"Panoptic Segmentation Results ({n_keep} segments)", fontsize=14)

# 保存和显示结果
plt.tight_layout()
output_path = im_path.replace("images", "outputs").replace(".jpg", "_segmentation.jpg")
plt.savefig(output_path, dpi=120, bbox_inches='tight')
print(f"Saved segmentation result to: {output_path}")
plt.show()