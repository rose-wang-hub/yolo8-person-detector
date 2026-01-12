"""Kuavo4Pro 人形跟随示例

本文件在现有 RealSense + YOLO 检测基础上，提供：

1. 人相对机器人坐标与速度的简化接口：HumanStateEstimator
   - 使用 RealSense 彩色 + 深度
   - 使用 YOLOv8 检测最近的人
   - 输出 (x, z, vx, vz, yaw_err)
     * x: 机器人坐标系下的左右偏移（左为正，单位 m）
     * z: 前后距离，前方为正（单位 m）
     * vx, vz: 以上量的时间导数（m/s）
     * yaw_err: 朝向误差角（弧度），>0 表示目标在左侧

2. 简单的基于 PID 的跟随控制器：SimplePIDFollower
   - 目标：保持 1.5m 距离，机器人朝向对准目标
   - 输出: 线速度 v_cmd、旋转速度 yaw_rate_cmd

3. Kuavo4Pro 控制接口占位：KuavoController
   - 这里不直接调用实际 Kuavo SDK，只是
     * 计算参考 v_cmd, yaw_rate_cmd
     * 预留 send_command(v, yaw_rate) 方法，用户可对接 Kuavo 的
       MPC / RL 控制接口，将 (v, yaw_rate) 作为参考轨迹或高层指令。

依赖：pyrealsense2, opencv-python, numpy, ultralytics

运行方式（仅测试感知与控制输出，不驱动真实机器人）：

    python3 kuavo_follow_controller.py

按 q 退出。
"""

import time
import math
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from scipy.cluster.hierarchy import fclusterdata

# 可选的深度 ReID 提取器（与 person_detector.py 保持一致思路）
reid_extractor = None


YOLO_MODEL_PATH = "yolov8n.pt"   # 轻量级模型
YOLO_PERSON_CONF = 0.45           # 人体检测置信度（与 person_detector 保持一致）
REID_SIM_THRESHOLD = 0.55         # 自适应外观特征的 ReID 阈值
# 初始外观模板与颜色模板的相似度阈值：
# 实测发现之前设置得过于保守，导致你重新进入画面时，即便人已经很大、
# YOLO 能稳定检测到，但 ReID + 颜色双重条件一直不过，从而迟迟不能重获目标。
# 这里适当放宽阈值，让“像你的人”更容易被重新锁定，同时仍然保留颜色作为
# 辅助约束，避免明显误跟别人。
BASE_SIM_THRESHOLD = 0.55        # 初始外观模板的相似度阈值（适当放宽）
COLOR_SIM_THRESHOLD = 0.60       # 衣服颜色 HSV 直方图相似度阈值（适当放宽）
# 丢失后 ReID 重获时的“放宽条件”：当最佳候选远高于其他人时，允许稍微放宽阈值
RELAXED_SCORE_THRESHOLD = 0.50
RELAXED_BASE_DELTA = 0.10         # BASE_SIM_THRESHOLD 可放宽的幅度
RELAXED_COLOR_DELTA = 0.20        # COLOR_SIM_THRESHOLD 可放宽的幅度
RELAXED_SCORE_MARGIN = 0.05       # 最佳候选比分数第二高的人至少高出该 margin
MAX_LOST_FRAMES = 10              # YOLO 检测丢失帧数阈值（参考 person_detector）
TARGET_DISTANCE = 1.5             # 期望跟随距离 (m)
MAX_TRACK_LOST = 30               # 连续丢失帧数上限（用于状态速度估计）
IOU_MATCH_THRESH = 0.3            # 初始 ROI 与检测框匹配阈值
CLUSTER_EPS_M = 0.6               # DBSCAN 风格聚类半径（X-Z 平面，单位 m）
YOLO_DETECT_INTERVAL = 1          # YOLO 每帧检测，提升快速运动时的响应


@dataclass
class HumanState:
    x: float      # 左右偏移 (m)，左为正
    z: float      # 前方距离 (m)
    vx: float     # x 方向速度 (m/s)
    vz: float     # z 方向速度 (m/s)
    yaw_err: float  # 朝向误差 (rad)，>0 目标在左侧
    valid: bool   # 当前是否有可靠目标


def iou_xyxy(box1, box2):
    """计算两个 [x1, y1, x2, y2] 框的 IoU。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter) / float(union + 1e-6)


def compute_appearance_feature(image, box_xywh):
    """提取人体外观特征：优先使用深度 ReID 模型，失败时回退颜色直方图。

    box_xywh: (x, y, w, h)
    返回归一化后的特征向量，或在无效时返回 None。
    """
    x, y, w, h = [int(i) for i in box_xywh]
    h_img, w_img = image.shape[:2]
    if w <= 0 or h <= 0:
        return None
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    x2 = max(0, min(x + w, w_img))
    y2 = max(0, min(y + h, h_img))
    if x2 <= x or y2 <= y:
        return None

    crop = image[y:y2, x:x2]
    if crop.size == 0:
        return None

    global reid_extractor
    if reid_extractor is not None:
        try:
            import torch
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            feats = reid_extractor([crop_rgb])
            if feats is None:
                raise RuntimeError("ReID 提取返回 None")
            feat = feats[0]
            if hasattr(feat, "detach"):
                feat = feat.detach().cpu().numpy()
            feat = np.asarray(feat, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(feat)
            if norm < 1e-6:
                raise RuntimeError("ReID 特征范数过小")
            feat /= norm
            return feat
        except Exception as e:
            print("警告：深度 ReID 特征提取失败，将回退到颜色直方图：", e)
            reid_extractor = None

    # 回退方案：HSV 颜色直方图
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(hist)
    if norm < 1e-6:
        return None
    hist /= norm
    return hist


def compute_color_hist(image, box_xywh):
    """单独计算 HSV 颜色直方图，用来更强调衣服颜色特征。

    无论是否有深度 ReID，都可以用这个颜色模板作为附加约束。
    """
    x, y, w, h = [int(i) for i in box_xywh]
    h_img, w_img = image.shape[:2]
    if w <= 0 or h <= 0:
        return None
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    x2 = max(0, min(x + w, w_img))
    y2 = max(0, min(y + h, h_img))
    if x2 <= x or y2 <= y:
        return None

    crop = image[y:y2, x:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(hist)
    if norm < 1e-6:
        return None
    hist /= norm
    return hist


def cosine_similarity(f1, f2):
    """计算两个特征向量的余弦相似度。"""
    if f1 is None or f2 is None:
        return -1.0
    denom = (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-6)
    return float(np.dot(f1, f2) / denom)


def select_roi_with_mouse(pipeline):
    """使用鼠标在实时 RealSense 彩色画面中选择初始 ROI，返回 (x, y, w, h)。"""
    selecting = False
    selection_complete = False
    start_pt = (0, 0)
    end_pt = (0, 0)
    roi = None

    window_name = "Select target ROI (s/space/enter to confirm, q to quit)"
    cv2.namedWindow(window_name)

    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, selection_complete, start_pt, end_pt, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            selection_complete = False
            start_pt = (x, y)
            end_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            end_pt = (x, y)
            x0, y0 = start_pt
            x1, y1 = end_pt
            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])
            w = x_max - x_min
            h = y_max - y_min
            if w > 0 and h > 0:
                roi = (x_min, y_min, w, h)
                selection_complete = True
                print(f"当前选择 ROI: {roi}")

    cv2.setMouseCallback(window_name, mouse_callback)
    print("请在窗口中用鼠标拖动选择跟随目标，按 s/空格/回车确认，按 q 退出")

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())

        display_image = color_image.copy()
        if selecting:
            cv2.rectangle(display_image, start_pt, end_pt, (0, 255, 255), 2)
        elif selection_complete and roi is not None:
            x, y, w, h = roi
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow(window_name, display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("用户取消选择，退出程序")
            cv2.destroyWindow(window_name)
            return None

        if key in (ord("s"), 32, 13) and selection_complete and roi is not None:
            print(f"用户最终确认 ROI: {roi}")
            cv2.destroyWindow(window_name)
            return roi


class HumanStateEstimator:
    """基于 RealSense + YOLO + KCF + ReID + 卡尔曼 的人相对机器人状态估计。

    - 启动时用鼠标框选 ROI 初始化 KCF 与目标外观特征
    - 每帧先用 KCF 跟踪，再用 YOLO 检测 + ReID 修正 ROI、自适应缩放
    - 使用卡尔曼滤波对中心点做时序平滑与轻微提前量
    - 仅在 ReID 相似度足够高时才重获目标，避免跟错人
    """

    def __init__(self, yolo_model_path: str = YOLO_MODEL_PATH):
        # RealSense 初始化
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)

        # 获取彩色相机内参，用于像素 -> 相机坐标
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.fx = intr.fx
        self.fy = intr.fy
        self.cx = intr.ppx
        self.cy = intr.ppy

        # YOLO 模型
        self.model = YOLO(yolo_model_path)

        # 初始化深度 ReID 模型
        global reid_extractor
        if reid_extractor is None:
            try:
                import torch
                import torchvision.models as models
                import torchvision.transforms as T

                class ResNetReIDExtractor:
                    def __init__(self, device: str):
                        self.device = device
                        try:
                            weights = models.ResNet18_Weights.DEFAULT
                            backbone = models.resnet18(weights=weights)
                        except Exception:
                            backbone = models.resnet18(weights=None)
                        backbone.fc = torch.nn.Identity()
                        self.model = backbone.to(self.device)
                        self.model.eval()

                        self.transform = T.Compose([
                            T.ToTensor(),
                            T.Resize((256, 128)),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                        ])

                    def __call__(self, img_list):
                        tensors = []
                        for img in img_list:
                            t = self.transform(img).to(self.device)
                            tensors.append(t)
                        if not tensors:
                            return None
                        batch = torch.stack(tensors, dim=0)
                        with torch.no_grad():
                            feats = self.model(batch)
                        if feats.ndim > 2:
                            feats = feats.view(feats.size(0), -1)
                        return feats

                device = "cuda" if torch.cuda.is_available() else "cpu"
                print("初始化深度 ReID 模型 ResNet18，设备:", device)
                reid_extractor = ResNetReIDExtractor(device)
                print("深度 ReID 模型初始化成功 (ResNet18)")
            except Exception as e:
                reid_extractor = None
                print("警告：深度 ReID 模型初始化失败，将使用颜色直方图 ReID：", e)

        # 启动时让用户框选要跟随的目标区域，并初始化 KCF 和外观特征
        self.tracker = cv2.TrackerKCF_create()
        self.roi = None
        roi = select_roi_with_mouse(self.pipeline)
        if roi is None:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            raise SystemExit("未选择初始目标 ROI，程序退出")
        self.roi = roi

        # 先取一帧彩色 + 深度图像，用于初始化 tracker、外观特征以及参考深度
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            raise SystemExit("初始化时未获取到彩色/深度帧，程序退出")
        init_color = np.asanyarray(color_frame.get_data())
        init_depth = np.asanyarray(depth_frame.get_data())

        self.tracker.init(init_color, self.roi)

        # 为身份模板构造一个略大于用户框选区域的 ROI，尽量覆盖上半身，
        # 这样即便一开始只框了头部，站起来露出全身时 ReID 也更容易匹配。
        h_img, w_img = init_color.shape[:2]
        x0, y0, w0, h0 = [int(i) for i in self.roi]
        cx = x0 + w0 / 2.0
        cy = y0 + h0 / 2.0
        scale = 1.8  # 模板区域放大倍数
        tmpl_w = min(w0 * scale, w_img)
        tmpl_h = min(h0 * scale, h_img)
        tmpl_x = int(max(0, min(cx - tmpl_w / 2.0, w_img - tmpl_w)))
        tmpl_y = int(max(0, min(cy - tmpl_h / 2.0, h_img - tmpl_h)))
        template_box = (tmpl_x, tmpl_y, int(tmpl_w), int(tmpl_h))

        # 记录动态缩放的参考深度和初始 ROI 尺寸（基于用户框选的区域）
        self.ref_roi_w = w0
        self.ref_roi_h = h0
        if y0 >= 0 and y0 + h0 <= init_depth.shape[0] and x0 >= 0 and x0 + w0 <= init_depth.shape[1]:
            depth_roi = init_depth[y0:y0 + h0, x0:x0 + w0]
            valid_depth = depth_roi[depth_roi > 0]
            if len(valid_depth) > 0:
                d = valid_depth.astype(np.float32)
                low, high = np.percentile(d, (10, 90))
                d = d[(d >= low) & (d <= high)]
                if len(d) > 0:
                    self.ref_depth_m = float(np.mean(d)) / 1000.0
                    self.depth_smooth = self.ref_depth_m
                    self.last_depth_m = self.ref_depth_m
                    self.roi_scale = 1.0
                    print(f"初始化参考深度: {self.ref_depth_m:.2f} m, 参考ROI尺寸: {self.ref_roi_w}x{self.ref_roi_h}")

        # 自适应外观特征（会缓慢更新），模板使用放大后的区域
        self.target_feature = compute_appearance_feature(init_color, template_box)
        # 初始外观模板：永远不变，用来锚定“只认第一次选中的那个人”
        self.base_feature = None if self.target_feature is None else self.target_feature.copy()
        # 衣服颜色模板（HSV 直方图），同样只在初始化时记录一次
        self.base_color_hist = compute_color_hist(init_color, template_box)

        if self.target_feature is not None:
            print("已初始化目标外观特征，用于后续 ReID 辅助锁定同一人")
        else:
            print("警告：初始化目标外观特征失败，将仅使用位置跟踪")
        if self.base_color_hist is not None:
            print("已记录初始衣服颜色模板，将辅助身份判定")

        # 卡尔曼滤波器，状态 [x,y,vx,vy]
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        self.kalman_initialized = False
        self.lost_frames_yolo = 0  # YOLO 无人检测计数

        # 轨迹记录：实际测量中心点 & 卡尔曼预测中心点（约 0.5 秒）
        self.track_points = []
        self.predict_points = []
        self.MAX_TRACK_LEN = 15
        self.MAX_PREDICT_LEN = 15

        # 运动启动判定
        self.motion_started = False
        self.motion_ref_point = None
        self.MOTION_START_DIST = 30.0  # 像素距离阈值

        # 用于相对速度估计
        self.prev_time = None
        self.prev_x = None
        self.prev_z = None
        # 用于按距离动态调整 ROI 尺寸：参考深度 & 初始 ROI 尺寸
        self.ref_depth_m = None
        self.ref_roi_w = None
        self.ref_roi_h = None
        # 最近一段时间平滑后的深度和对应缩放因子，避免 ROI 尺寸突然跳变
        self.last_depth_m = None
        self.depth_smooth = None
        self.roi_scale = 1.0
        # 用于在可视化和深度估计时对 ROI 做额外的平滑，抑制单帧跳变
        self.prev_vis_roi = None
        # 帧计数器，用于控制 YOLO 检测频率，降低整体计算负载
        self.frame_idx = 0
        # 帧计数器，用于控制 YOLO 检测频率，降低整体计算负载
        self.frame_idx = 0

    def _pixel_to_cam(self, u: float, v: float, z: float):
        """像素坐标 (u,v) + 深度 z(m) -> 相机坐标系 (X,Y,Z)."""
        X = (u - self.cx) / self.fx * z
        Y = (v - self.cy) / self.fy * z
        Z = z
        return X, Y, Z

    def get_state(self) -> HumanState:
        """读取一帧，运行 YOLO+KCF+ReID+卡尔曼 跟踪，返回 HumanState。"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return HumanState(0.0, 0.0, 0.0, 0.0, 0.0, False)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 轻微高斯滤波，抑制噪声
        color_blur = cv2.GaussianBlur(color_image, (5, 5), 0)

        # YOLO 检测人体：为减轻计算负担，仅每 YOLO_DETECT_INTERVAL 帧执行一次
        self.frame_idx += 1
        run_yolo = (self.frame_idx % YOLO_DETECT_INTERVAL == 0)
        detections = []
        if run_yolo:
            results = self.model(color_blur, conf=YOLO_PERSON_CONF, classes=[0], verbose=False)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                detections = results[0].boxes.xyxy.cpu().numpy().astype(int)

        # tracker.update（仅在当前有有效 ROI 时才调用，避免未初始化/空矩阵异常）
        ok = False
        roi = None
        if self.roi is not None:
            try:
                ok, roi = self.tracker.update(color_blur)
            except Exception as e:
                print(f"tracker.update异常: {e}")
                ok = False
                roi = None

        # YOLO 丢失计数：只在真正执行 YOLO 检测的帧上更新
        if run_yolo:
            if len(detections) == 0:
                self.lost_frames_yolo += 1
            else:
                self.lost_frames_yolo = 0

        if self.lost_frames_yolo >= MAX_LOST_FRAMES:
            # 多帧无人，重置 KCF 和 ROI，但保留外观特征，等待 ReID 重获
            self.tracker = cv2.TrackerKCF_create()
            self.roi = None
            ok = False

        # 输出和可视化初始化
        x_cam = 0.0
        z_cam = 0.0
        vx = 0.0
        vz = 0.0
        yaw_err = 0.0
        z_m = 0.0
        state_valid = False
        u = 0.0
        v = 0.0
        debug_image = color_image.copy()

        # --- KCF 成功：与 person_detector 一致的检测-跟踪融合与卡尔曼 ---
        used_roi = False
        if ok and roi is not None:
            x, y, w, h = [int(i) for i in roi]
            h_img, w_img = color_blur.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > w_img or y + h > h_img:
                # ROI 越界，视为丢失，交给 ReID 重获逻辑
                print(f"ROI 越界或无效，视为丢失: {roi}")
                self.roi = None
                ok = False
            else:
                # 检测-跟踪融合：优先用 YOLO+ReID+颜色模板 修正 ROI，
                # 若身份特征不可用或不稳定，则退化为基于 IoU 的自适应放大，
                # 确保当你从画面边缘慢慢走出更多身体时，ROI 能跟着“长大”。
                best_id_box = None
                best_id_area = 0.0
                best_iou_box = None
                best_iou = 0.0
                roi_xyxy = np.array([x, y, x + w, y + h])
                cur_area = float(max(0, w) * max(0, h))

                for b in detections:
                    x1, y1, x2, y2 = b
                    area = float(max(0, x2 - x1) * max(0, y2 - y1))

                    # 1) 记录与当前 ROI IoU 最大的检测框，供后面做几何放大兜底
                    iou_val = iou_xyxy(roi_xyxy, b)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_iou_box = b

                    # 2) 若有外观/颜色模板，则按“像初始化那个人 + 颜色相似”筛选，
                    #    并在这些候选中选择面积最大的一个，作为首选修正框。
                    if self.target_feature is None or self.base_feature is None:
                        continue

                    box_xywh = (x1, y1, x2 - x1, y2 - y1)
                    feat = compute_appearance_feature(color_blur, box_xywh)
                    color_feat = compute_color_hist(color_blur, box_xywh)
                    if feat is None or color_feat is None:
                        continue

                    sim_base = cosine_similarity(self.base_feature, feat)
                    sim_adapt = cosine_similarity(self.target_feature, feat)
                    sim_color = cosine_similarity(self.base_color_hist, color_feat) if self.base_color_hist is not None else 1.0

                    # 只在“像初始化的那个人 + 颜色也相似”时才认为候选是你
                    if sim_base >= BASE_SIM_THRESHOLD and sim_color >= COLOR_SIM_THRESHOLD:
                        if area > best_id_area:
                            best_id_area = area
                            best_id_box = b

                chosen_box = None
                # 优先使用通过身份判定且面积最大的检测框
                if best_id_box is not None:
                    chosen_box = best_id_box
                # 如果没有通过身份判定的候选，但存在与当前 ROI 重叠较大的检测框，
                # 则统一采用该检测框来纠正 ROI，无论它比当前 ROI 大还是小。
                # 之前只在“面积明显更大”时才更新，会导致当 KCF 漂移把框拉得很大
                # 时，YOLO 无法把 ROI 缩回到真实人体大小，从而出现类似“整幅画面
                # 都是蓝框”的情况。
                elif best_iou_box is not None and best_iou >= IOU_MATCH_THRESH:
                    chosen_box = best_iou_box

                if chosen_box is not None:
                    x1, y1, x2, y2 = chosen_box
                    self.roi = (x1, y1, x2 - x1, y2 - y1)
                    self.tracker = cv2.TrackerKCF_create()
                    self.tracker.init(color_blur, self.roi)
                    x, y, w, h = [int(i) for i in self.roi]
                else:
                    # YOLO 无法可靠确认身份，且几何上也没有显著更大的候选框，
                    # 则继续沿用当前 KCF ROI，避免突然跳到画面中另一个人身上。
                    self.roi = roi

                if self.roi is not None:
                    # 更新外观特征
                    x, y, w, h = [int(i) for i in self.roi]
                    curr_feat = compute_appearance_feature(color_blur, (x, y, w, h))
                    if curr_feat is not None:
                        if self.target_feature is None:
                            self.target_feature = curr_feat
                        else:
                            alpha = 0.9
                            self.target_feature = alpha * self.target_feature + (1 - alpha) * curr_feat
                            norm_tf = np.linalg.norm(self.target_feature)
                            if norm_tf > 1e-6:
                                self.target_feature = self.target_feature / norm_tf

                    # 卡尔曼滤波平滑中心点
                    center_x = x + w / 2.0
                    center_y = y + h / 2.0
                    if not self.kalman_initialized:
                        self.kalman.statePost = np.array([[center_x], [center_y], [0.0], [0.0]], np.float32)
                        self.kalman.statePre = self.kalman.statePost.copy()
                        self.kalman_initialized = True
                        self.motion_ref_point = (center_x, center_y)
                        self.track_points.clear()
                        self.predict_points.clear()
                        fused_x, fused_y = center_x, center_y
                    else:
                        measurement = np.array([[center_x], [center_y]], np.float32)
                        self.kalman.correct(measurement)
                        prediction = self.kalman.predict()
                        fused_x, fused_y = float(prediction[0]), float(prediction[1])

                    # 记录轨迹点
                    self.track_points.append((int(center_x), int(center_y)))
                    self.predict_points.append((int(fused_x), int(fused_y)))
                    if len(self.track_points) > self.MAX_TRACK_LEN:
                        self.track_points.pop(0)
                    if len(self.predict_points) > self.MAX_PREDICT_LEN:
                        self.predict_points.pop(0)

                    # 判断是否开始运动
                    if (not self.motion_started) and self.motion_ref_point is not None:
                        dx = center_x - self.motion_ref_point[0]
                        dy = center_y - self.motion_ref_point[1]
                        if (dx * dx + dy * dy) ** 0.5 >= self.MOTION_START_DIST:
                            self.motion_started = True

                    # 旧逻辑会根据深度对 ROI 进行较大幅度的放大，
                    # 实测容易把背景/地面也框进来，导致 ROI 看起来不够“贴身”。
                    # 这里取消按深度强制放大尺寸，仅保留后面基于历史 ROI 的
                    # 轻量平滑，让当前 ROI 更贴近 YOLO 实际检测到的人体框。

                    # 将 ROI 中心对齐到预测中心
                    new_x = int(fused_x - w / 2)
                    new_y = int(fused_y - h / 2)
                    new_x = max(0, min(new_x, depth_image.shape[1] - w))
                    new_y = max(0, min(new_y, depth_image.shape[0] - h))
                    x, y = new_x, new_y
                    self.roi = (x, y, w, h)
                    used_roi = True

        # --- KCF 失败：优先保证“不要跟错人”，但在画面里只有你一个人时要更果断地重获 ---
        if not ok:
            if len(detections) > 0 and self.target_feature is not None and self.base_feature is not None:
                chosen_idx = None
                best_score = -1.0
                second_best_score = -1.0
                best_base = -1.0
                best_color = -1.0

                # --- DBSCAN/3D 约束：在 X-Z 平面上对检测框中心聚类，优先选择既像目标又与上一次位置接近的簇 ---
                centers_3d = []  # (X, Z, det_idx)
                for idx, b in enumerate(detections):
                    x1, y1, x2, y2 = b
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    win = 4
                    ix = int(max(0, min(cx, depth_image.shape[1] - win)))
                    iy = int(max(0, min(cy, depth_image.shape[0] - win)))
                    d_roi = depth_image[iy:iy + win, ix:ix + win]
                    if d_roi.size == 0:
                        continue
                    valid = d_roi[d_roi > 0]
                    if len(valid) == 0:
                        continue
                    d_m = float(np.median(valid)) / 1000.0
                    if d_m <= 0.2:
                        continue
                    X3, Y3, Z3 = self._pixel_to_cam(cx, cy, d_m)
                    centers_3d.append((X3, Z3, idx))

                if centers_3d:
                    pts = np.array([[c[0], c[1]] for c in centers_3d], dtype=np.float32)
                    try:
                        labels = fclusterdata(pts, CLUSTER_EPS_M, criterion="distance")
                    except Exception:
                        labels = np.ones(len(pts), dtype=int)

                    clusters = {}
                    for (X3, Z3, det_idx), lab in zip(centers_3d, labels):
                        clusters.setdefault(lab, []).append((X3, Z3, det_idx))

                    for lab, items in clusters.items():
                        # 在每个簇内，先找到 ReID/颜色得分最高的检测框
                        cluster_best_idx = None
                        cluster_best_score = -1.0
                        cluster_best_base = -1.0
                        cluster_best_color = -1.0
                        rep_X, rep_Z = items[0][0], items[0][1]

                        for X3, Z3, det_idx in items:
                            x1, y1, x2, y2 = detections[det_idx]
                            box_xywh = (x1, y1, x2 - x1, y2 - y1)
                            feat = compute_appearance_feature(color_blur, box_xywh)
                            color_feat = compute_color_hist(color_blur, box_xywh)
                            if feat is None or color_feat is None:
                                continue
                            sim_base = cosine_similarity(self.base_feature, feat)
                            sim_adapt = cosine_similarity(self.target_feature, feat)
                            sim_color = cosine_similarity(self.base_color_hist, color_feat) if self.base_color_hist is not None else 1.0
                            score = 0.7 * sim_base + 0.3 * sim_adapt
                            if score > cluster_best_score:
                                cluster_best_score = score
                                cluster_best_idx = det_idx
                                cluster_best_base = sim_base
                                cluster_best_color = sim_color
                                rep_X, rep_Z = X3, Z3

                        if cluster_best_idx is None:
                            continue

                        # 将 3D 距离信息融入得分：更偏向于靠近上一帧目标位置的簇
                        score_spatial = cluster_best_score
                        if self.prev_x is not None and self.prev_z is not None:
                            dx = rep_X - self.prev_x
                            dz = rep_Z - self.prev_z
                            dist = math.hypot(dx, dz)
                            score_spatial = cluster_best_score - 0.3 * dist

                        if score_spatial > best_score:
                            second_best_score = best_score
                            best_score = score_spatial
                            chosen_idx = cluster_best_idx
                            best_base = cluster_best_base
                            best_color = cluster_best_color
                        elif score_spatial > second_best_score:
                            second_best_score = score_spatial
                else:
                    # 深度不可用时，退化为原有的逐检测 ReID/颜色打分逻辑
                    for idx, b in enumerate(detections):
                        x1, y1, x2, y2 = b
                        box_xywh = (x1, y1, x2 - x1, y2 - y1)
                        feat = compute_appearance_feature(color_blur, box_xywh)
                        color_feat = compute_color_hist(color_blur, box_xywh)
                        if feat is None or color_feat is None:
                            continue
                        sim_base = cosine_similarity(self.base_feature, feat)
                        sim_adapt = cosine_similarity(self.target_feature, feat)
                        sim_color = cosine_similarity(self.base_color_hist, color_feat) if self.base_color_hist is not None else 1.0
                        score = 0.7 * sim_base + 0.3 * sim_adapt
                        if score > best_score:
                            second_best_score = best_score
                            best_score = score
                            chosen_idx = idx
                            best_base = sim_base
                            best_color = sim_color
                        elif score > second_best_score:
                            second_best_score = score

                accept = False
                if chosen_idx is not None and best_score > 0:
                    # 场景 1：画面中只有一个人，且外观相似度明显较高 —— 更果断地重获，避免你
                    # 已经走进画面却迟迟不被识别的情况。
                    if len(detections) == 1 and best_base >= (BASE_SIM_THRESHOLD - 0.05):
                        accept = True
                    else:
                        # 场景 2：多人场景，仍然采用“严格 + 放宽”两级条件，避免跟错人。
                        # 1) 严格条件：外观和颜色都明显接近初始化模板
                        if best_base >= BASE_SIM_THRESHOLD and best_color >= COLOR_SIM_THRESHOLD:
                            accept = True
                        else:
                            # 2) 放宽条件：最佳候选远高于其他人，且相似度略低于严格阈值但仍然较高
                            if (best_score >= RELAXED_SCORE_THRESHOLD and
                                    (best_score - max(second_best_score, 0.0)) >= RELAXED_SCORE_MARGIN and
                                    best_base >= (BASE_SIM_THRESHOLD - RELAXED_BASE_DELTA) and
                                    best_color >= (COLOR_SIM_THRESHOLD - RELAXED_COLOR_DELTA)):
                                print("ReID 使用放宽阈值重获目标: score=%.2f base=%.2f color=%.2f" %
                                      (best_score, best_base, best_color))
                                accept = True

                if accept:
                    x1, y1, x2, y2 = detections[chosen_idx]
                    self.roi = (x1, y1, x2 - x1, y2 - y1)
                    self.tracker = cv2.TrackerKCF_create()
                    self.tracker.init(color_blur, self.roi)

                    # ReID 重获后也更新卡尔曼与轨迹，保持与 person_detector 一致
                    x, y, w, h = [int(i) for i in self.roi]
                    center_x = x + w / 2.0
                    center_y = y + h / 2.0
                    if not self.kalman_initialized:
                        self.kalman.statePost = np.array([[center_x], [center_y], [0.0], [0.0]], np.float32)
                        self.kalman.statePre = self.kalman.statePost.copy()
                        self.kalman_initialized = True
                        self.motion_ref_point = (center_x, center_y)
                        self.track_points.clear()
                        self.predict_points.clear()
                        fused_x, fused_y = center_x, center_y
                    else:
                        measurement = np.array([[center_x], [center_y]], np.float32)
                        self.kalman.correct(measurement)
                        prediction = self.kalman.predict()
                        fused_x, fused_y = float(prediction[0]), float(prediction[1])

                    self.track_points.append((int(center_x), int(center_y)))
                    self.predict_points.append((int(fused_x), int(fused_y)))
                    if len(self.track_points) > self.MAX_TRACK_LEN:
                        self.track_points.pop(0)
                    if len(self.predict_points) > self.MAX_PREDICT_LEN:
                        self.predict_points.pop(0)

                    if (not self.motion_started) and self.motion_ref_point is not None:
                        dx = center_x - self.motion_ref_point[0]
                        dy = center_y - self.motion_ref_point[1]
                        if (dx * dx + dy * dy) ** 0.5 >= self.MOTION_START_DIST:
                            self.motion_started = True

                    # 同理，这里也不再按深度对 ROI 做额外放大，
                    # 避免 ReID 重获后框的尺寸一下子变得很大，
                    # 直接使用 YOLO 检测框 + 后面的温和平滑即可。

                    used_roi = True
                else:
                    # 暂时丢失，不跟错人
                    self.roi = None
                    cv2.putText(debug_image, "目标暂时丢失，未找到外观/颜色匹配的人", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # 跟踪失败且没有可用检测
                self.roi = None
                cv2.putText(debug_image, "跟踪失败且当前帧未检测到人体", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 若此时没有有效 ROI，则认为丢失，但仍然刷新调试画面
        if self.roi is None or not used_roi:
            # 缓存一份调试图像供主窗口使用
            self.last_debug_image = debug_image.copy()
            cv2.imshow("Kuavo Person Tracking", debug_image)
            return HumanState(x_cam, z_cam, vx, vz, yaw_err, False)

        # 在用于姿态/深度之前，对 ROI 做一次温和的时序平滑，抑制单帧尺寸/位置的剧烈跳变
        x_raw, y_raw, w_raw, h_raw = [int(i) for i in self.roi]
        if self.prev_vis_roi is not None:
            px, py, pw, ph = self.prev_vis_roi

            # 1) 尺寸变化自适应：
            #    - 正常情况下做温和限幅，避免轻微抖动导致 ROI 忽大忽小；
            #    - 当面积变化非常明显（>40%）时，认为是你前后大幅移动，
            #      直接信任当前检测尺寸，快速跟上远近变化。
            w, h = w_raw, h_raw
            if pw > 0 and ph > 0:
                prev_area = float(pw) * float(ph)
                raw_area = float(max(1, w_raw) * max(1, h_raw))
                area_ratio = raw_area / max(1e-3, prev_area)
            else:
                area_ratio = 1.0

            if pw > 0 and ph > 0 and 0.7 <= area_ratio <= 1.4:
                # 面积变化不算剧烈时，启用平滑缩放，抑制抖动
                max_scale = 1.45
                min_scale = 0.65
                scale_w = float(w_raw) / float(pw)
                scale_h = float(h_raw) / float(ph)
                scale_w = max(min_scale, min(max_scale, scale_w))
                scale_h = max(min_scale, min(max_scale, scale_h))
                w = int(pw * scale_w)
                h = int(ph * scale_h)

            # 2) 中心位置跳变抑制：若中心位移过大，则只允许每帧最多移动一定像素
            cx_raw = x_raw + w_raw / 2.0
            cy_raw = y_raw + h_raw / 2.0
            cx_prev = px + pw / 2.0
            cy_prev = py + ph / 2.0
            dx = cx_raw - cx_prev
            dy = cy_raw - cy_prev
            # 允许更大的单帧中心位移，使快速移动时 ROI 追随更敏捷
            max_jump = 70.0  # 单帧允许的最大像素位移
            dist = math.hypot(dx, dy)
            if dist > max_jump and dist > 1e-3:
                ratio = max_jump / dist
                cx = cx_prev + dx * ratio
                cy = cy_prev + dy * ratio
            else:
                cx, cy = cx_raw, cy_raw

            x = int(cx - w / 2.0)
            y = int(cy - h / 2.0)
        else:
            x, y, w, h = x_raw, y_raw, w_raw, h_raw

        # 将平滑后的 ROI 限制在图像范围内
        h_img_full, w_img_full = depth_image.shape[:2]
        w = max(2, min(w, w_img_full))
        h = max(2, min(h, h_img_full))
        x = max(0, min(x, w_img_full - w))
        y = max(0, min(y, h_img_full - h))

        self.prev_vis_roi = (x, y, w, h)
        self.roi = (x, y, w, h)

        # 计算用于姿态/深度的“人体核心区域”中心像素
        # 为了避免单侧伸手、腿部等把中心点拉偏，这里只取 ROI 中央的一块区域
        # （例如宽度中间 60%、高度中间 60%），用来估计中心与深度。
        x, y, w, h = [int(i) for i in self.roi]
        inner_margin_x = int(w * 0.2)
        inner_margin_y = int(h * 0.2)
        inner_x = x + inner_margin_x
        inner_y = y + inner_margin_y
        inner_w = max(4, w - 2 * inner_margin_x)
        inner_h = max(4, h - 2 * inner_margin_y)

        u = inner_x + inner_w / 2.0
        v = inner_y + inner_h / 2.0

        # ROI 深度估计（10-90 百分位去噪 + 平均），同样只用核心区域，
        # 避免把身后墙面或桌面大量像素算进去导致距离偏移。
        if inner_y < 0 or inner_x < 0 or inner_y + inner_h > depth_image.shape[0] or inner_x + inner_w > depth_image.shape[1]:
            cv2.putText(debug_image, "ROI 超出深度范围，忽略本帧", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            self.last_debug_image = debug_image.copy()
            cv2.imshow("Kuavo Person Tracking", debug_image)
            return HumanState(x_cam, z_cam, vx, vz, yaw_err, False)

        depth_roi = depth_image[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w]
        if depth_roi.size > 0:
            valid_depth = depth_roi[depth_roi > 0]
            if len(valid_depth) > 0:
                d = valid_depth.astype(np.float32)
                low, high = np.percentile(d, (10, 90))
                d = d[(d >= low) & (d <= high)]
                if len(d) > 0:
                    z_m = float(np.mean(d)) / 1000.0

        if z_m <= 0.1:
            # 深度不可靠
            cv2.putText(debug_image, "深度不可靠，忽略本帧", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            self.last_debug_image = debug_image.copy()
            cv2.imshow("Kuavo Person Tracking", debug_image)
            return HumanState(x_cam, z_cam, vx, vz, yaw_err, False)

        # 记录最近一次可靠深度，并做指数平滑，用于后续按距离动态缩放 ROI
        if self.depth_smooth is None:
            self.depth_smooth = z_m
        else:
            alpha_d = 0.85  # 越接近 1 越平滑
            self.depth_smooth = alpha_d * self.depth_smooth + (1.0 - alpha_d) * z_m
        self.last_depth_m = self.depth_smooth

        # 像素 -> 相机坐标
        X, Y, Z = self._pixel_to_cam(u, v, z_m)
        x_cam = X
        z_cam = Z

        # 估计速度（相机坐标系下）
        now = time.time()
        if self.prev_time is None or self.prev_x is None or self.prev_z is None:
            vx = 0.0
            vz = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)
            vx = (x_cam - self.prev_x) / dt
            vz = (z_cam - self.prev_z) / dt

        self.prev_time = now
        self.prev_x = x_cam
        self.prev_z = z_cam

        yaw_err = math.atan2(x_cam, z_cam)
        state_valid = True

        # 调试可视化：绘制当前跟踪框、轨迹和位姿信息（与 person_detector 风格一致）
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(debug_image, (int(u), int(v)), 4, (0, 255, 0), -1)

        # 历史轨迹（测量点：黄色线，预测点：青色线，仅在 motion_started 后绘制预测）
        if len(self.track_points) > 1:
            for i in range(1, len(self.track_points)):
                cv2.line(debug_image, self.track_points[i - 1], self.track_points[i], (0, 255, 255), 2)
        if self.motion_started and len(self.predict_points) > 1:
            for i in range(1, len(self.predict_points)):
                cv2.line(debug_image, self.predict_points[i - 1], self.predict_points[i], (255, 255, 0), 2)

        text = f"x={x_cam:.2f}m z={z_cam:.2f}m yaw={yaw_err:.2f}rad"
        cv2.putText(debug_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_image, f"Depth: {z_m:.2f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Kuavo Person Tracking", debug_image)

        return HumanState(x=x_cam, z=z_cam, vx=vx, vz=vz, yaw_err=yaw_err, valid=state_valid)

    def stop(self):
        self.pipeline.stop()


class SimplePIDFollower:
    """基于 PID 的简单跟随控制：保持距离 TARGET_DISTANCE，朝向对准目标。"""

    def __init__(self,
                 dist_target: float = TARGET_DISTANCE,
                 kp_dist: float = 0.8, ki_dist: float = 0.0, kd_dist: float = 0.1,
                 kp_yaw: float = 1.2, ki_yaw: float = 0.0, kd_yaw: float = 0.1,
                 v_max: float = 0.8,
                 yaw_rate_max: float = 1.0):
        self.dist_target = dist_target
        self.kp_dist = kp_dist
        self.ki_dist = ki_dist
        self.kd_dist = kd_dist
        self.kp_yaw = kp_yaw
        self.ki_yaw = ki_yaw
        self.kd_yaw = kd_yaw
        self.v_max = v_max
        self.yaw_rate_max = yaw_rate_max

        self._int_dist = 0.0
        self._prev_dist_err = 0.0
        self._int_yaw = 0.0
        self._prev_yaw_err = 0.0
        self._last_time = None

    def compute(self, state: HumanState):
        """根据人相对位置，输出 (v_cmd, yaw_rate_cmd)。"""
        now = time.time()
        if self._last_time is None:
            dt = 0.01
        else:
            dt = max(1e-3, now - self._last_time)
        self._last_time = now

        if not state.valid:
            # 目标丢失时，缓慢停止
            return 0.0, 0.0

        # 距离误差：希望 z = dist_target
        dist_err = state.z - self.dist_target
        self._int_dist += dist_err * dt
        d_dist = (dist_err - self._prev_dist_err) / dt
        self._prev_dist_err = dist_err

        v_cmd = self.kp_dist * dist_err + self.ki_dist * self._int_dist + self.kd_dist * d_dist

        # 朝向误差：希望 yaw_err = 0
        yaw_err = state.yaw_err
        self._int_yaw += yaw_err * dt
        d_yaw = (yaw_err - self._prev_yaw_err) / dt
        self._prev_yaw_err = yaw_err

        yaw_rate_cmd = self.kp_yaw * yaw_err + self.ki_yaw * self._int_yaw + self.kd_yaw * d_yaw

        # 饱和限制
        v_cmd = max(-self.v_max, min(self.v_max, v_cmd))
        yaw_rate_cmd = max(-self.yaw_rate_max, min(self.yaw_rate_max, yaw_rate_cmd))

        return v_cmd, yaw_rate_cmd


class KuavoController:
    """Kuavo4Pro 高层跟随控制占位类。

    - 感知：调用 HumanStateEstimator 获取 (x, z, vx, vz, yaw_err)
    - 控制：可选 PID/MPC/RL 三种模式输出 (v, yaw_rate)
    - 下发：send_command(v, yaw_rate) 由用户对接实际 Kuavo SDK
    """

    def __init__(self, mode: str = "pid"):
        assert mode in ("pid", "mpc", "rl")
        self.mode = mode
        self.estimator = HumanStateEstimator()
        self.pid = SimplePIDFollower()

    # === 下发到 Kuavo 的接口（需用户根据实际 SDK 实现） ===
    def send_command(self, v: float, yaw_rate: float):
        """占位：将 (v, yaw_rate) 发送给 Kuavo4Pro。

        对接方式示例（伪代码）：

            cmd = KuavoCmd()
            cmd.set_linear_velocity(v)
            cmd.set_angular_velocity(yaw_rate)
            kuavo_api.send(cmd)
        """
        print(f"[KuavoCmd] v={v:.3f} m/s, yaw_rate={yaw_rate:.3f} rad/s")

    # === MPC 模式伪代码 ===
    def _mpc_control(self, state: HumanState):
        """MPC 模式示例：这里给出公式/结构，实际求解需接入 MPC 求解器。

        状态向量可定义为：
            s = [dist_err, yaw_err, v_rel]
        控制向量：
            u = [v_cmd, yaw_rate_cmd]

        目标：
            在预测时域 N 步内最小化代价：
                J = Σ (w_d * dist_err_k^2 + w_y * yaw_err_k^2 + w_u * ||u_k||^2)
        约束：
            v_min <= v_cmd <= v_max
            -yaw_rate_max <= yaw_rate_cmd <= yaw_rate_max

        这里为了示意，暂时返回与 PID 相同的输出，
        实际使用时请将此函数改为调用 Kuavo 自带或你实现的 MPC 控制器。
        """
        return self.pid.compute(state)

    # === RL 模式伪代码 ===
    def _rl_control(self, state: HumanState):
        """RL 模式示例：

        状态输入 s 可为：
            s = [dist_err, yaw_err, vx, vz]
        策略网络 π(s) 输出动作：
            a = [v_cmd, yaw_rate_cmd]

        这里为了示意，暂用简单线性“策略”：
            v_cmd = k1 * dist_err
            yaw_rate_cmd = k2 * yaw_err
        实际部署请替换为你训练好的 RL 策略网络。
        """
        if not state.valid:
            return 0.0, 0.0
        dist_err = state.z - TARGET_DISTANCE
        yaw_err = state.yaw_err
        k1, k2 = 0.6, 1.0
        v_cmd = k1 * dist_err
        yaw_rate_cmd = k2 * yaw_err
        v_cmd = max(-0.8, min(0.8, v_cmd))
        yaw_rate_cmd = max(-1.0, min(1.0, yaw_rate_cmd))
        return v_cmd, yaw_rate_cmd

    def step_once(self) -> bool:
        """执行一次感知+控制循环，返回 False 表示应退出。"""
        state = self.estimator.get_state()

        if self.mode == "pid":
            v_cmd, yaw_rate_cmd = self.pid.compute(state)
        elif self.mode == "mpc":
            v_cmd, yaw_rate_cmd = self._mpc_control(state)
        else:  # rl
            v_cmd, yaw_rate_cmd = self._rl_control(state)

        self.send_command(v_cmd, yaw_rate_cmd)
        return True

    def run_loop(self):
        print(f"KuavoController 运行中，模式={self.mode}，按 q 退出")
        try:
            while True:
                if not self.step_once():
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.estimator.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # 示例：以 PID 模式运行，仅打印 Kuavo 命令，不实际驱动机器人
    controller = KuavoController(mode="pid")
    controller.run_loop()
