import os
from ultralytics import YOLO

"""
简单的 YOLOv8 人体检测微调脚本示例。

使用方法（示例）：
1. 准备数据集目录结构，例如：

   datasets/person_dataset/
     images/
       train/
         xxx1.jpg
         xxx2.jpg
         ...
       val/
         yyy1.jpg
         ...
     labels/
       train/
         xxx1.txt
         xxx2.txt
       val/
         yyy1.txt

   其中 labels 里的 txt 为 YOLO 格式：
       <class_id> <cx> <cy> <w> <h>
   class_id 对于单类 person 通常为 0。

2. 写一个数据集配置 YAML 文件（例如 person_dataset.yaml）：

   path: datasets/person_dataset
   train: images/train
   val: images/val
   names:
     0: person

3. 在本目录下运行：

   python3 train_person_yolo.py --data person_dataset.yaml --epochs 50

训练完成后，会在 runs/detect/exp*/weights/best.pt 生成你自己的权重，
然后可以在 person_detector.py 里把 'yolov8n-pose.pt' 换成该路径（或使用对应的 detect 模型）。
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 person detector on custom dataset")
    parser.add_argument("--data", type=str, required=True, help="数据集配置 YAML 路径，例如 person_dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="预训练检测模型，例如 yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--project", type=str, default="runs/detect", help="训练输出主目录")
    parser.add_argument("--name", type=str, default="person_custom", help="实验名称")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据集配置文件不存在: {args.data}")

    print("加载预训练 YOLOv8 模型:", args.model)
    model = YOLO(args.model)

    print("开始训练...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    print("训练完成。权重文件通常在:")
    print(f"  {os.path.join(args.project, args.name, 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
