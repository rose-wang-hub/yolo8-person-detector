#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""局部 A* 栅格规划 Demo（不依赖全局 SLAM）

目标：
- 在机器人局部 5~10 米范围内构建一个二维栅格地图
- 在该局部栅格上用 A* 搜索从起点到目标点的无碰撞路径
- 为后续与 RealSense / 雷达和 Kuavo 控制对接预留接口

当前版本：
- 仅包含栅格坐标系定义、A* 实现和简单的随机障碍测试 main()
- 后续可将障碍更新部分换成：来自 /scan、RealSense 深度、HumanState 目标点等
"""

from dataclasses import dataclass
import heapq
from typing import List, Tuple, Optional

import numpy as np
import cv2
import pyrealsense2 as rs


@dataclass
class GridConfig:
    """局部栅格参数配置。"""

    size_m: float = 10.0
    resolution: float = 0.1

    @property
    def cells(self) -> int:
        return int(self.size_m / self.resolution)


class LocalGrid:
    """以机器人为中心的局部栅格表示。"""

    def __init__(self, cfg: GridConfig):
        self.cfg = cfg
        n = cfg.cells
        self.grid = np.zeros((n, n), dtype=np.uint8)

    def world_to_grid(self, x: float, z: float) -> Optional[Tuple[int, int]]:
        half = self.cfg.size_m / 2.0
        if not (-half <= x <= half and 0.0 <= z <= self.cfg.size_m):
            return None
        gx = int((x + half) / self.cfg.resolution)
        gz = int(z / self.cfg.resolution)
        if 0 <= gz < self.grid.shape[0] and 0 <= gx < self.grid.shape[1]:
            return gz, gx
        return None

    def grid_to_world(self, gy: int, gx: int) -> Tuple[float, float]:
        half = self.cfg.size_m / 2.0
        x = gx * self.cfg.resolution - half + self.cfg.resolution * 0.5
        z = gy * self.cfg.resolution + self.cfg.resolution * 0.5
        return x, z

    def clear(self) -> None:
        self.grid.fill(0)

    def set_obstacle(self, gy: int, gx: int) -> None:
        if 0 <= gy < self.grid.shape[0] and 0 <= gx < self.grid.shape[1]:
            self.grid[gy, gx] = 1

    def is_free(self, gy: int, gx: int) -> bool:
        if 0 <= gy < self.grid.shape[0] and 0 <= gx < self.grid.shape[1]:
            return self.grid[gy, gx] == 0
        return False


def astar(grid: LocalGrid, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """在 LocalGrid 上执行 4 邻接 A* 搜索。"""

    sy, sx = start
    gy, gx = goal
    if not grid.is_free(sy, sx) or not grid.is_free(gy, gx):
        return None

    h = lambda y, x: abs(y - gy) + abs(x - gx)

    open_heap: List[Tuple[float, int, int]] = []
    heapq.heappush(open_heap, (h(sy, sx), sy, sx))

    g_cost = np.full_like(grid.grid, np.inf, dtype=float)
    g_cost[sy, sx] = 0.0

    came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = np.zeros_like(grid.grid, dtype=bool)

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_heap:
        f, y, x = heapq.heappop(open_heap)
        if visited[y, x]:
            continue
        visited[y, x] = True

        if (y, x) == (gy, gx):
            path: List[Tuple[int, int]] = [(y, x)]
            while (y, x) in came_from:
                y, x = came_from[(y, x)]
                path.append((y, x))
            path.reverse()
            return path

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if not grid.is_free(ny, nx) or visited[ny, nx]:
                continue
            tentative_g = g_cost[y, x] + 1.0
            if tentative_g < g_cost[ny, nx]:
                g_cost[ny, nx] = tentative_g
                came_from[(ny, nx)] = (y, x)
                f_new = tentative_g + h(ny, nx)
                heapq.heappush(open_heap, (f_new, ny, nx))

    return None


def demo_random_obstacles() -> None:
    """在随机障碍栅格上用 A* 规划，并用 OpenCV 可视化。"""

    cfg = GridConfig(size_m=10.0, resolution=0.1)
    lg = LocalGrid(cfg)

    n = cfg.cells
    rng = np.random.default_rng(0)
    density = 0.15
    mask = rng.random((n, n)) < density
    lg.grid[mask] = 1

    sy, sx = n - 2, n // 2
    gy, gx = n // 4, n // 2

    lg.grid[sy, sx] = 0
    lg.grid[gy, gx] = 0

    path = astar(lg, (sy, sx), (gy, gx))

    img = np.zeros((n, n, 3), dtype=np.uint8)
    img[lg.grid == 1] = (50, 50, 50)

    if path is not None:
        for py, px in path:
            img[py, px] = (0, 255, 0)

    img[sy, sx] = (0, 0, 255)
    img[gy, gx] = (255, 0, 0)

    scale = 4
    img_vis = cv2.resize(img, (n * scale, n * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Local A* demo", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_realsense_local_astar() -> None:
    """使用 RealSense 深度构建局部栅格，并实时在栅格上做 A* 规划。

    说明：
    - 以机器人为原点，z 轴向前，x 轴左右，构建 size_m × size_m 的局部栅格（默认 10m）。
    - 使用深度图中前方一定高度范围内的点，投影到 X-Z 平面，并在栅格中标记为障碍。
    - 起点固定在栅格底部中央（机器人脚下），目标点可以简单地取前方某个固定距离，
      后续可以替换为“鼠标点击”或基于 HumanState 的目标位置。
    - 每帧重新构建栅格并规划路径，在 OpenCV 窗口中显示：障碍、起点、终点和路径。
    """

    cfg = GridConfig(size_m=10.0, resolution=0.1)
    lg = LocalGrid(cfg)
    n = cfg.cells

    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(rs_cfg)

    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())

            lg.clear()

            intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

            h, w = depth.shape
            ys = np.arange(0, h, 4)
            xs = np.arange(0, w, 4)
            for v in ys:
                for u in xs:
                    d = depth[v, u] * depth_scale
                    if d <= 0.2 or d > cfg.size_m:
                        continue
                    X = (u - cx) / fx * d
                    Y = (v - cy) / fy * d
                    Z = d
                    if Y < -0.3 or Y > 1.5:
                        continue
                    cell = lg.world_to_grid(X, Z)
                    if cell is not None:
                        gy, gx = cell
                        lg.set_obstacle(gy, gx)

            sy, sx = n - 2, n // 2
            gy, gx = n // 3, n // 2
            lg.grid[sy, sx] = 0
            lg.grid[gy, gx] = 0

            path = astar(lg, (sy, sx), (gy, gx))

            img = np.zeros((n, n, 3), dtype=np.uint8)
            img[lg.grid == 1] = (50, 50, 50)
            if path is not None:
                for py, px in path:
                    img[py, px] = (0, 255, 0)
            img[sy, sx] = (0, 0, 255)
            img[gy, gx] = (255, 0, 0)

            scale = 4
            img_vis = cv2.resize(img, (n * scale, n * scale), interpolation=cv2.INTER_NEAREST)

            color_small = cv2.resize(color, (img_vis.shape[1], img_vis.shape[0]))
            stacked = np.hstack((color_small, img_vis))
            cv2.imshow("RealSense Local A* (left=color, right=grid)", stacked)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 默认运行随机障碍 demo，如需测试 RealSense 版本，请改为：
    demo_realsense_local_astar()
    #demo_random_obstacles()
