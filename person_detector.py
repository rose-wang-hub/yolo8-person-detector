print("程序启动")
try:
    import pyrealsense2 as rs
    print("pyrealsense2 导入成功")
    import cv2
    print("cv2 导入成功")
    import numpy as np
    print("numpy 导入成功")
    from ultralytics import YOLO
    print("ultralytics 导入成功")
    from scipy.spatial import KDTree
    print("scipy 导入成功")
except Exception as e:
    print("依赖导入失败：", e)
    raise


# 全局深度 ReID 特征提取器（如可用）
reid_extractor = None


# ---- 可调参数配置 ----
YOLO_PERSON_CONF = 0.45       # YOLO 人体检测置信度阈值
KCF_DRIFT_IOU_THRESH = 0.3    # KCF 漂移判定的 IoU 阈值
REID_SIM_THRESHOLD = 0.55     # 深度 ReID 相似度阈值
MAX_LOST_FRAMES = 10          # 连续多少帧无人检测到就认为目标已离开


def iou_xyxy(box1, box2):
    """计算两个[x1,y1,x2,y2]框的IoU。"""
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

    # 1) 若已初始化深度 ReID 提取器，优先用深度特征
    global reid_extractor
    if reid_extractor is not None:
        try:
            import torch
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # torchreid 的 FeatureExtractor 接受 PIL/Image 或 ndarray
            feats = reid_extractor([crop_rgb])  # (1, D)
            if feats is None:
                raise RuntimeError("ReID 提取返回 None")
            feat = feats[0]
            if hasattr(feat, 'detach'):
                feat = feat.detach().cpu().numpy()
            feat = np.asarray(feat, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(feat)
            if norm < 1e-6:
                raise RuntimeError("ReID 特征范数过小")
            feat /= norm
            return feat
        except Exception as e:
            # 深度 ReID 失败时打印一次警告并回退到颜色直方图
            print("警告：深度 ReID 特征提取失败，将回退到颜色直方图：", e)
            reid_extractor = None

    # 2) 回退方案：转 HSV 做 3D 颜色直方图
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32)
    hist = hist.reshape(-1)
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


def main():
    print("初始化Realsense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    print("Realsense启动成功")

    print("加载YOLOv8模型...")
    model = YOLO('yolov8n-pose.pt')
    print("YOLOv8模型加载成功")

    # 初始化深度 ReID 模型（使用 torchvision ResNet18，如可用）
    global reid_extractor
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        class ResNetReIDExtractor:
            def __init__(self, device: str):
                self.device = device
                try:
                    weights = models.ResNet18_Weights.DEFAULT  # 新版 torchvision
                    backbone = models.resnet18(weights=weights)
                except Exception:
                    backbone = models.resnet18(weights=None)

                # 去掉分类头，直接用特征向量
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
                # img_list: list of HWC RGB np.ndarray
                tensors = []
                for img in img_list:
                    t = self.transform(img).to(self.device)
                    tensors.append(t)
                if not tensors:
                    return None
                batch = torch.stack(tensors, dim=0)
                with torch.no_grad():
                    feats = self.model(batch)  # (N, D, 1, 1) 或 (N, D)
                if feats.ndim > 2:
                    feats = feats.view(feats.size(0), -1)
                return feats

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("初始化深度 ReID 模型 ResNet18，设备:", device)
        reid_extractor = ResNetReIDExtractor(device)
        print("深度 ReID 模型初始化成功 (ResNet18)")
    except Exception as e:
        reid_extractor = None
        print("警告：深度 ReID 模型初始化失败，将使用颜色直方图 ReID：", e)

    # 目标人体的外观特征（用于 ReID）
    target_feature = None

    tracker = cv2.TrackerKCF_create()
    roi = None
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    try:
        # ---------- 实时视频下的ROI框选 ----------
        selecting = False
        selection_complete = False
        start_pt = (0, 0)
        end_pt = (0, 0)

        window_name = 'Select ROI'
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
                    print(f"当前选择ROI: {roi}")

        cv2.setMouseCallback(window_name, mouse_callback)

        print("请在窗口中用鼠标拖动选择目标，按 s/空格/回车确认，按 q 退出程序")

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("未获取到彩色帧")
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # 拷贝一份用于画框
            display_image = color_image.copy()
            if selecting:
                cv2.rectangle(display_image, start_pt, end_pt, (0, 255, 255), 2)
            elif selection_complete and roi is not None:
                x, y, w, h = roi
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("用户取消，退出程序")
                return

            # s、空格 或 回车 确认选择
            if key in (ord('s'), 32, 13) and selection_complete and roi is not None:
                tracker.init(color_image, roi)
                # 初始化目标外观特征
                target_feature = compute_appearance_feature(color_image, roi)
                if target_feature is not None:
                    print("已初始化目标外观特征，用于后续 ReID 辅助锁定同一人")
                else:
                    print("警告：初始化目标外观特征失败，将仅使用位置跟踪")
                print(f"用户最终确认ROI: {roi}")
                break

        cv2.destroyWindow(window_name)

        # 轨迹记录：实际测量中心点 & 卡尔曼预测中心点
        track_points = []           # 实际测量轨迹（约 0.5 秒）
        predict_points = []         # 预测轨迹（约 0.5 秒）
        MAX_TRACK_LEN = 15          # 真实轨迹点数上限（30FPS 下约 0.5 秒）
        MAX_PREDICT_LEN = 15        # 预测轨迹点数上限（与真实一致）

        # 卡尔曼初始化与运动检测
        kalman_initialized = False
        motion_started = False
        motion_ref_point = None     # 用于判断是否“开始运动”的参考点
        MOTION_START_DIST = 30.0    # 像素距离阈值，超过认为开始运动

        lost_frames = 0

        # 进入跟踪循环（KCF + YOLO纠偏与重获）
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 对彩色图像做轻微高斯滤波，抑制噪声与轻微抖动
            color_image = cv2.GaussianBlur(color_image, (5, 5), 0)

            # YOLO检测所有人体（person 类别为 0）
            results = model(color_image, conf=YOLO_PERSON_CONF, classes=[0])
            detections = []  # [N,4] xyxy
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                detections = results[0].boxes.xyxy.cpu().numpy().astype(int)


            # 检查roi有效性，防止tracker.update返回空或异常
            try:
                ok, roi = tracker.update(color_image)
            except Exception as e:
                print(f"tracker.update异常: {e}")
                ok = False
                roi = None
            print(f"跟踪器状态: {ok}, ROI: {roi}")

            # 若长时间检测不到任何人体，则认为目标已经离开，清空跟踪状态
            if len(detections) == 0:
                lost_frames += 1
            else:
                lost_frames = 0

            if lost_frames >= MAX_LOST_FRAMES:
                # 多帧都没有检测到人体，重置跟踪器和ROI，但保留外观特征，方便人再次出现时通过ReID重获
                tracker = cv2.TrackerKCF_create()
                roi = None
                ok = False
                print("连续多帧未检测到人体，重置跟踪器和ROI，等待目标再次出现")

            if ok and roi is not None:
                x, y, w, h = [int(i) for i in roi]
                h_img, w_img = color_image.shape[:2]
                # 检查ROI是否越界
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > w_img or y + h > h_img:
                    print(f"ROI 越界或无效，跳过本帧: {roi}")
                else:
                    # --- 检测-跟踪融合：用YOLO检测结果自适应修正ROI尺寸 ---
                    best_iou = 0.0
                    best_iou_box = None
                    best_sim = -1.0
                    best_sim_box = None
                    roi_xyxy = np.array([x, y, x + w, y + h])
                    for b in detections:
                        i = iou_xyxy(roi_xyxy, b)
                        if i > best_iou:
                            best_iou = i
                            best_iou_box = b
                        if target_feature is not None:
                            x1, y1, x2, y2 = b
                            box_xywh = (x1, y1, x2 - x1, y2 - y1)
                            f = compute_appearance_feature(color_image, box_xywh)
                            if f is not None:
                                sim = cosine_similarity(target_feature, f)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_sim_box = b

                    # 只要有高IoU或高ReID相似度的检测框，就用检测框修正ROI并重置KCF，实现自适应缩放
                    chosen_box = None
                    if best_sim_box is not None and best_sim >= REID_SIM_THRESHOLD:
                        chosen_box = best_sim_box
                        print(f"检测-跟踪融合：用ReID修正ROI，similarity={best_sim:.2f}, IoU={best_iou:.2f}")
                    elif best_iou_box is not None and best_iou > 0.2:  # IoU阈值可调
                        chosen_box = best_iou_box
                        print(f"检测-跟踪融合：用IoU修正ROI，IoU={best_iou:.2f}")

                    if chosen_box is not None:
                        x1, y1, x2, y2 = chosen_box
                        roi = (x1, y1, x2 - x1, y2 - y1)
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(color_image, roi)
                        x, y, w, h = [int(i) for i in roi]

                    # 更新目标外观特征（缓慢自适应光照/姿态变化）
                    curr_feat = compute_appearance_feature(color_image, (x, y, w, h))
                    if curr_feat is not None:
                        if target_feature is None:
                            target_feature = curr_feat
                        else:
                            alpha = 0.9
                            target_feature = alpha * target_feature + (1 - alpha) * curr_feat
                            norm_tf = np.linalg.norm(target_feature)
                            if norm_tf > 1e-6:
                                target_feature = target_feature / norm_tf

                    # 使用卡尔曼滤波对目标中心位置做时序平滑，并用预测中心微调ROI位置（带轻微提前量）
                    center_x, center_y = x + w/2, y + h/2

                    # 第一次有有效测量时初始化卡尔曼状态，避免从(0,0)拉长线
                    if not kalman_initialized:
                        kalman.statePost = np.array([[center_x], [center_y], [0.0], [0.0]], np.float32)
                        kalman.statePre = kalman.statePost.copy()
                        kalman_initialized = True
                        motion_ref_point = (center_x, center_y)
                        track_points.clear()
                        predict_points.clear()
                        fused_x, fused_y = center_x, center_y
                    else:
                        measurement = np.array([[center_x], [center_y]], np.float32)
                        kalman.correct(measurement)
                        prediction = kalman.predict()  # 状态: [x, y, vx, vy]
                        fused_x, fused_y = float(prediction[0]), float(prediction[1])

                    # 记录轨迹点
                    track_points.append((int(center_x), int(center_y)))
                    predict_points.append((int(fused_x), int(fused_y)))
                    if len(track_points) > MAX_TRACK_LEN:
                        track_points.pop(0)
                    if len(predict_points) > MAX_PREDICT_LEN:
                        predict_points.pop(0)

                    # 判断是否开始运动：与初始参考点的距离是否超过阈值
                    if not motion_started and motion_ref_point is not None:
                        dx = center_x - motion_ref_point[0]
                        dy = center_y - motion_ref_point[1]
                        if (dx * dx + dy * dy) ** 0.5 >= MOTION_START_DIST:
                            motion_started = True

                    # 将ROI中心对准预测位置，可认为是一点“提前量”
                    new_x = int(fused_x - w / 2)
                    new_y = int(fused_y - h / 2)
                    # 边界裁剪，防止越界
                    new_x = max(0, min(new_x, depth_image.shape[1] - w))
                    new_y = max(0, min(new_y, depth_image.shape[0] - h))
                    x, y = new_x, new_y
                    roi = (x, y, w, h)

                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(color_image, f"KF: ({int(fused_x)},{int(fused_y)})", (x, y+h+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 绘制历史轨迹（测量点：黄色线，预测点：青色线）
                    if len(track_points) > 1:
                        for i in range(1, len(track_points)):
                            cv2.line(color_image, track_points[i-1], track_points[i], (0, 255, 255), 2)
                    # 只有“开始运动”之后才绘制蓝色预测轨迹
                    if motion_started and len(predict_points) > 1:
                        for i in range(1, len(predict_points)):
                            cv2.line(color_image, predict_points[i-1], predict_points[i], (255, 255, 0), 2)

                    # 检查depth_roi是否越界和非空
                    if y >= 0 and y + h <= depth_image.shape[0] and x >= 0 and x + w <= depth_image.shape[1]:
                        depth_roi = depth_image[y:y+h, x:x+w]
                        if depth_roi.size > 0:
                            valid_depth = depth_roi[depth_roi > 0]
                            if len(valid_depth) > 0:
                                # 深度去噪：使用百分位裁剪去除异常值，再求平均
                                d = valid_depth.astype(np.float32)
                                low, high = np.percentile(d, (10, 90))
                                d = d[(d >= low) & (d <= high)]
                                if len(d) > 0:
                                    avg_depth = float(np.mean(d)) / 1000.0
                                    cv2.putText(color_image, f"Depth: {avg_depth:.2f}m", (x, y-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        print(f"depth_roi 越界或无效，跳过本帧: x={x}, y={y}, w={w}, h={h}")
            else:
                # KCF 失败时，尝试使用 YOLO 检测结果 + 外观特征重新初始化跟踪器
                if len(detections) > 0 and target_feature is not None:
                    chosen_idx = None
                    best_sim = -1.0

                    for idx, b in enumerate(detections):
                        x1, y1, x2, y2 = b
                        box_xywh = (x1, y1, x2 - x1, y2 - y1)
                        f = compute_appearance_feature(color_image, box_xywh)
                        if f is None:
                            continue
                        sim = cosine_similarity(target_feature, f)
                        if sim > best_sim:
                            best_sim = sim
                            chosen_idx = idx

                    # 只有当存在外观相似度足够高的人体时才重新初始化ROI，否则宁可暂时不跟踪也不跟错人
                    if chosen_idx is not None and best_sim >= REID_SIM_THRESHOLD:
                        x1, y1, x2, y2 = detections[chosen_idx]
                        print(f"跟踪失败，基于 ReID 重新锁定同一人，similarity={best_sim:.2f}")

                        roi = (x1, y1, x2 - x1, y2 - y1)
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(color_image, roi)
                        x, y, w, h = [int(i) for i in roi]
                        print("已通过 ReID 重新初始化 ROI:", roi)

                        # 使用卡尔曼滤波预测中心，并微调ROI位置
                        center_x, center_y = x + w/2, y + h/2

                        if not kalman_initialized:
                            kalman.statePost = np.array([[center_x], [center_y], [0.0], [0.0]], np.float32)
                            kalman.statePre = kalman.statePost.copy()
                            kalman_initialized = True
                            motion_ref_point = (center_x, center_y)
                            track_points.clear()
                            predict_points.clear()
                            fused_x, fused_y = center_x, center_y
                        else:
                            measurement = np.array([[center_x], [center_y]], np.float32)
                            kalman.correct(measurement)
                            prediction = kalman.predict()
                            fused_x, fused_y = float(prediction[0]), float(prediction[1])

                        # 记录轨迹点
                        track_points.append((int(center_x), int(center_y)))
                        predict_points.append((int(fused_x), int(fused_y)))
                        if len(track_points) > MAX_TRACK_LEN:
                            track_points.pop(0)
                        if len(predict_points) > MAX_PREDICT_LEN:
                            predict_points.pop(0)

                        if not motion_started and motion_ref_point is not None:
                            dx = center_x - motion_ref_point[0]
                            dy = center_y - motion_ref_point[1]
                            if (dx * dx + dy * dy) ** 0.5 >= MOTION_START_DIST:
                                motion_started = True

                        new_x = int(fused_x - w / 2)
                        new_y = int(fused_y - h / 2)
                        new_x = max(0, min(new_x, depth_image.shape[1] - w))
                        new_y = max(0, min(new_y, depth_image.shape[0] - h))
                        x, y = new_x, new_y
                        roi = (x, y, w, h)

                        cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(color_image, f"KF: ({int(fused_x)},{int(fused_y)})", (x, y+h+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # 绘制历史轨迹
                        if len(track_points) > 1:
                            for i in range(1, len(track_points)):
                                cv2.line(color_image, track_points[i-1], track_points[i], (0, 255, 255), 2)
                        if motion_started and len(predict_points) > 1:
                            for i in range(1, len(predict_points)):
                                cv2.line(color_image, predict_points[i-1], predict_points[i], (255, 255, 0), 2)

                        depth_roi = depth_image[y:y+h, x:x+w]
                        valid_depth = depth_roi[depth_roi > 0]
                        if len(valid_depth) > 0:
                            d = valid_depth.astype(np.float32)
                            low, high = np.percentile(d, (10, 90))
                            d = d[(d >= low) & (d <= high)]
                            if len(d) > 0:
                                avg_depth = float(np.mean(d)) / 1000.0
                                cv2.putText(color_image, f"Depth: {avg_depth:.2f}m", (x, y-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # 检测到人体，但没有与初始目标足够相似的人，宁可暂时丢失也不跟错人
                        cv2.putText(color_image, "目标暂时丢失，未找到外观相似的人体", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(color_image, "跟踪失败且未检测到人体", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Realsense Person Tracking', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户退出程序")
                break
    except Exception as e:
        print("主循环发生异常：", e)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("程序已关闭")

if __name__ == '__main__':
    main()
