import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
import os


# 新增：加载角度与读数对照表
def load_angle_table(file_path):
    """加载角度与读数对照表"""
    angle_table = []
    if not os.path.exists(file_path):
        print(f"警告：角度对照表文件 {file_path} 不存在")
        return angle_table

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    reading = float(parts[0])
                    angle = float(parts[1])
                    # 转换为弧度存储
                    angle_table.append((reading, math.radians(angle)))
                except ValueError:
                    print(f"警告：无效的角度表数据格式 - {line}")

    # 按角度排序
    angle_table.sort(key=lambda x: x[1])
    return angle_table


# 新增：根据角度查找对应的读数
def get_reading_from_angle(angle, angle_table):
    """根据角度在对照表中查找对应的读数，进行线性插值"""
    if not angle_table:
        return 0.0

    # 确保角度在有效范围内
    min_angle = angle_table[0][1]
    max_angle = angle_table[-1][1]

    if angle <= min_angle:
        return angle_table[0][0]
    if angle >= max_angle:
        return angle_table[-1][0]

    # 查找角度所在的区间
    for i in range(len(angle_table) - 1):
        curr_reading, curr_angle = angle_table[i]
        next_reading, next_angle = angle_table[i + 1]

        if curr_angle <= angle <= next_angle:
            # 线性插值计算
            ratio = (angle - curr_angle) / (next_angle - curr_angle)
            return curr_reading + ratio * (next_reading - curr_reading)

    return 0.0


# 加载 YOLOv8 模型
model = YOLO(r"model/pointer_detect.onnx")

# 建立索引到类别的映射关系
index_to_class = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'center', 11: 'plate', 12: 'pointer'
}

# 置信度阈值设置
conf_thresholds = {
    '0': 0.6, '1': 0.6, '2': 0.6, '3': 0.3, '4': 0.6,
    '5': 0.3, '6': 0.6, '7': 0.3, '8': 0.6, '9': 0.3,
    'center': 0.3, 'plate': 0.5, 'pointer': 0.2
}

# 初始化摄像头
cap = cv2.VideoCapture(21)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 获取摄像头的实际帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"摄像头原始帧率: {fps} FPS")

# 设置检测间隔（每4帧检测一次模型）
detection_interval = 4
print(f"模型检测间隔: 每 {detection_interval} 帧一次")

# 新增：加载角度对照表并计算量程
angle_file_path = "angle.txt"  # 角度对照表文件路径
angle_table = load_angle_table(angle_file_path)
print(f"加载角度对照表完成，共 {len(angle_table)} 条记录")

# 从角度表中提取读数并确定量程（从小到大排序）
table_readings = [item[0] for item in angle_table]
if table_readings:
    table_readings.sort()
    global_min_reading = table_readings[0]
    global_max_reading = table_readings[-1]
    global_range = f"{global_min_reading:.1f}-{global_max_reading:.1f}"
    print(f"从角度表获取量程: {global_range}")
else:
    global_range = "0-0"
    print("角度表为空，无法确定量程")


# 辅助函数
def line_length(line):
    """计算直线长度"""
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1) **2 + (y2 - y1)** 2)


def distance_between_points(p1, p2):
    """计算两点之间的距离"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def extend_line(line, image_shape):
    """延展直线以确保它们能相交"""
    x1, y1, x2, y2 = line
    img_width, img_height = image_shape[1], image_shape[0]

    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
    else:
        return [x1, 0, x1, img_height]

    intercept = y1 - slope * x1
    x_start = 0
    y_start = int(slope * x_start + intercept)
    x_end = img_width
    y_end = int(slope * x_end + intercept)

    if y_start < 0:
        y_start = 0
        x_start = int((y_start - intercept) / slope)
    elif y_start >= img_height:
        y_start = img_height - 1
        x_start = int((y_start - intercept) / slope)

    if y_end < 0:
        y_end = 0
        x_end = int((y_end - intercept) / slope)
    elif y_end >= img_height:
        y_end = img_height - 1
        x_end = int((y_end - intercept) / slope)

    return [int(x_start), int(y_start), int(x_end), int(y_end)]


def line_intersection(line1, line2):
    """计算两条直线的交点"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return int(x), int(y)


def circle_line_intersection(circle_center, radius, line):
    """计算圆与直线的交点"""
    cx, cy = circle_center
    x1, y1, x2, y2 = line

    # 直线方程：ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    # 圆心到直线的距离
    distance = abs(a * cx + b * cy + c) / math.hypot(a, b) if (a != 0 or b != 0) else 0

    if distance > radius:
        return []  # 无交点
    elif distance == radius:
        # 相切，一个交点
        x0 = (b * (b * cx - a * cy) - a * c) / (a **2 + b** 2)
        y0 = (a * (-b * cx + a * cy) - b * c) / (a **2 + b** 2)
        return [(int(x0), int(y0))]
    else:
        # 相交，两个交点
        dx = x2 - x1
        dy = y2 - y1
        len_line = math.hypot(dx, dy)
        u = ((cx - x1) * dx + (cy - y1) * dy) / (len_line **2)
        x = x1 + u * dx
        y = y1 + u * dy
        d = math.sqrt(radius** 2 - distance **2)
        ax = x - (dy * d) / len_line
        ay = y + (dx * d) / len_line
        bx = x + (dy * d) / len_line
        by = y - (dx * d) / len_line
        return [(int(ax), int(ay)), (int(bx), int(by))]


def combine_digits(digit_detections):
    """改进的数字组合算法"""
    if not digit_detections:
        return [], []

    all_heights = [y2 - y1 for x1, y1, x2, y2, *_ in digit_detections]
    all_widths = [x2 - x1 for x1, y1, x2, y2, *_ in digit_detections]
    mean_height = sum(all_heights) / len(all_heights) if all_heights else 0
    mean_width = sum(all_widths) / len(all_widths) if all_widths else 0

    if mean_height <= 0 or mean_width <= 0:
        numbers = [int(d[4]) for d in digit_detections]
        centers = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2, *_ in digit_detections]
        return numbers, centers

    digits_info = []
    for det in digit_detections:
        x1, y1, x2, y2, cls_name, conf = det
        digits_info.append({
            'center_x': (x1 + x2) / 2,
            'center_y': (y1 + y2) / 2,
            'top': y1,
            'bottom': y2,
            'left': x1,
            'right': x2,
            'width': x2 - x1,
            'height': y2 - y1,
            'cls': cls_name,
            'conf': conf
        })

    horizontal_groups = []
    for digit in digits_info:
        placed = False
        for group in horizontal_groups:
            group_mean_top = sum(d['top'] for d in group) / len(group)
            group_mean_bottom = sum(d['bottom'] for d in group) / len(group)
            if (abs(digit['top'] - group_mean_top) < 0.5 * mean_height and
                    abs(digit['bottom'] - group_mean_bottom) < 0.5 * mean_height):
                group.append(digit)
                placed = True
                break
        if not placed:
            horizontal_groups.append([digit])

    combined_numbers = []
    combined_centers = []
    for group in horizontal_groups:
        group.sort(key=lambda d: d['center_x'])
        clusters = []
        max_gap = mean_width

        while max_gap < mean_width * 3.0:
            clusters = []
            current_cluster = []
            for i in range(len(group)):
                if not current_cluster:
                    current_cluster = [group[i]]
                else:
                    prev_digit = group[i - 1]
                    curr_digit = group[i]
                    gap = curr_digit['left'] - prev_digit['right']
                    if gap < max_gap:
                        current_cluster.append(curr_digit)
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [curr_digit]
            if current_cluster:
                clusters.append(current_cluster)
            num_clusters = len(clusters)
            if num_clusters <= 2:
                break
            else:
                max_gap += 0.2 * mean_width

        for cluster in clusters:
            cluster.sort(key=lambda d: d['center_x'])
            cluster_center_x = np.mean([d['center_x'] for d in cluster])
            cluster_center_y = np.mean([d['center_y'] for d in cluster])
            cluster_center = (int(cluster_center_x), int(cluster_center_y))
            cluster_nums = ''.join(d['cls'] for d in cluster)

            try:
                num_value = int(cluster_nums)
                combined_numbers.append(num_value)
                combined_centers.append(cluster_center)
            except ValueError:
                for d in cluster:
                    try:
                        num_val = int(d['cls'])
                        combined_numbers.append(num_val)
                        combined_centers.append((int(d['center_x']), int(d['center_y'])))
                    except:
                        pass

    unique_dict = {}
    for num, center in zip(combined_numbers, combined_centers):
        if num not in unique_dict:
            unique_dict[num] = center
    combined_numbers = sorted(unique_dict.keys())
    combined_centers = [unique_dict[num] for num in combined_numbers]

    return combined_numbers, combined_centers


def find_intersection_with_circle(center, radius, line_start, line_end):
    """找到直线与圆的交点"""
    cx, cy = center
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1
    a = dx **2 + dy** 2

    if a == 0:
        return None

    b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
    c = (x1 - cx) **2 + (y1 - cy)** 2 - radius **2
    discriminant = b** 2 - 4 * a * c

    if discriminant < 0:
        return None

    sqrt_d = np.sqrt(discriminant)
    t1 = (-b + sqrt_d) / (2 * a)
    t2 = (-b - sqrt_d) / (2 * a)

    point1 = (x1 + t1 * dx, y1 + t1 * dy)
    point2 = (x1 + t2 * dx, y1 + t2 * dy)

    if np.isnan(point1[0]) or np.isnan(point1[1]):
        point1 = None
    else:
        point1 = (int(point1[0]), int(point1[1]))

    if np.isnan(point2[0]) or np.isnan(point2[1]):
        point2 = None
    else:
        point2 = (int(point2[0]), int(point2[1]))

    if point1 is None and point2 is None:
        return None
    if point1 is None:
        return point2
    if point2 is None:
        return point1

    dist1 = np.sqrt((point1[0] - line_end[0]) **2 + (point1[1] - line_end[1])** 2)
    dist2 = np.sqrt((point2[0] - line_end[0]) **2 + (point2[1] - line_end[1])** 2)
    return point1 if dist1 < dist2 else point2


def detect_scale_lines(plate_roi, center, r, plate_box, digit_detections=None):
    """检测表盘上的刻度线"""
    cx, cy = center
    x1, y1, x2, y2 = plate_box

    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        ~blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )

    inner_radius = 0
    mean_diag = 0
    if digit_detections:
        diag_lengths = []
        for det in digit_detections:
            d_x1, d_y1, d_x2, d_y2 = det
            diag = math.sqrt((d_x2 - d_x1) **2 + (d_y2 - d_y1)** 2)
            diag_lengths.append(diag)
        if diag_lengths:
            mean_diag = sum(diag_lengths) / len(diag_lengths)

    if digit_detections:
        distances = []
        for det in digit_detections:
            d_x1, d_y1, d_x2, d_y2 = det
            d_center_x = (d_x1 + d_x2) / 2
            d_center_y = (d_y1 + d_y2) / 2
            dist = math.sqrt((d_center_x - cx) **2 + (d_center_y - cy)** 2)
            distances.append(dist)
        if distances:
            inner_radius = np.mean(distances) + mean_diag / 2
        else:
            inner_radius = r * 0.5
    else:
        inner_radius = r * 0.5

    mask = np.zeros_like(binary)
    cv2.circle(mask, (int(cx), int(cy)), int(inner_radius), 255, -1)
    binary = cv2.bitwise_and(binary, cv2.bitwise_not(mask))

    scale_lines = []
    lines = cv2.HoughLinesP(
        binary, 1, np.pi / 180, 25, minLineLength=10, maxLineGap=2
    )

    if lines is not None:
        lines = lines.squeeze()
        if lines.ndim == 1:
            lines = [lines]
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line
            mid_x = (x1_l + x2_l) / 2
            mid_y = (y1_l + y2_l) / 2
            distance = np.sqrt((mid_x - cx) **2 + (mid_y - cy)** 2)

            if not (inner_radius < distance < r):
                continue

            vec_to_center = np.array([cx - mid_x, cy - mid_y])
            vec_line = np.array([x2_l - x1_l, y2_l - y1_l])
            cos_angle = np.dot(vec_to_center, vec_line) / (
                    np.linalg.norm(vec_to_center) * np.linalg.norm(vec_line) + 1e-5
            )

            if abs(cos_angle) > 0.7:
                x1_pt = int(x1_l + x1)
                y1_pt = int(y1_l + y1)
                x2_pt = int(x2_l + x1)
                y2_pt = int(y2_l + y1)
                scale_lines.append((x1_pt, y1_pt, x2_pt, y2_pt))

    return scale_lines, binary, inner_radius


def point_on_line(point, line):
    """判断点是否在直线上（有一定容差）"""
    x, y = point
    x1, y1, x2, y2 = line

    # 计算点到直线的距离
    denominator = math.sqrt((y2 - y1) **2 + (x2 - x1)** 2)
    if denominator == 0:
        return False
    distance = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denominator
    return distance < 3  # 容差为3像素


def find_scale_intersection(center, scale_lines, max_radius, final_center):
    """
    改进的刻度点检测：以center为圆心扩张圆，与至少两条刻度线各有一交点后，
    选择距离final_center（最终圆圆心）更近的交点作为刻度点
    """
    if not scale_lines:
        return None, 0  # 无刻度线

    # 存储不同刻度线的交点（去重）
    line_intersections = {}  # key: 刻度线索引, value: 交点列表
    min_radius_with_two = None  # 首次出现两条刻度线都有交点的半径
    best_intersection = None
    best_distance = float('inf')

    # 逐步增大半径查找交点
    for r in range(1, max_radius, 2):
        current_intersections = {}  # 当前半径下各刻度线的交点

        # 检查每条刻度线与当前圆的交点
        for line_idx, line in enumerate(scale_lines):
            intersections = circle_line_intersection(center, r, line)
            if intersections:
                current_intersections[line_idx] = intersections

        # 更新全局交点记录
        for line_idx, points in current_intersections.items():
            if line_idx not in line_intersections:
                line_intersections[line_idx] = []
            # 添加新交点（去重）
            for p in points:
                if p not in line_intersections[line_idx]:
                    line_intersections[line_idx].append(p)

        # 检查是否有至少两条刻度线有交点
        valid_lines = [idx for idx, points in line_intersections.items() if len(points) > 0]
        if len(valid_lines) >= 2:
            # 记录首次满足条件的半径
            if min_radius_with_two is None:
                min_radius_with_two = r

            # 收集所有有效交点
            all_points = []
            for line_idx in valid_lines:
                all_points.extend(line_intersections[line_idx])

            # 计算各交点到最终圆圆心的距离，选择最近的
            for p in all_points:
                dist = distance_between_points(p, final_center)
                if dist < best_distance:
                    best_distance = dist
                    best_intersection = p

        # 半径超过首次满足条件半径的1.2倍后停止搜索（避免过大半径）
        if min_radius_with_two is not None and r > min_radius_with_two * 1.2:
            break

    # 如果找到有效交点
    if best_intersection:
        return best_intersection, min_radius_with_two if min_radius_with_two else max_radius
    else:
        # fallback：使用原始方法返回第一个交点
        for r in range(1, max_radius, 2):
            for angle in range(0, 360, 5):
                rad = math.radians(angle)
                x = int(center[0] + r * math.cos(rad))
                y = int(center[1] + r * math.sin(rad))
                for line in scale_lines:
                    if point_on_line((x, y), line):
                        return (x, y), r
        return None, 0


def line_angle_with_x_axis(line):
    """计算直线与X轴的夹角（弧度），范围[0, 2π)"""
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0:  # 垂直线
        return math.pi / 2 if dy > 0 else 3 * math.pi / 2

    angle = math.atan2(dy, dx)  # 范围(-π, π]
    return angle if angle >= 0 else angle + 2 * math.pi  # 转换为[0, 2π)


def angle_difference(angle1, angle2):
    """计算从angle1到angle2的顺时针角度差（范围[0, 2π)）"""
    diff = (angle2 - angle1) % (2 * math.pi)
    return diff


def get_valid_range_angle(min_angle, max_angle, pointer_angle):
    """
    计算包含指针的有效量程角度
    返回：(量程角度, 是否顺时针方向)
    """
    # 计算顺时针和逆时针两种可能的角度差
    clockwise_diff = angle_difference(min_angle, max_angle)
    counter_diff = 2 * math.pi - clockwise_diff

    # 判断哪种角度差包含指针
    if is_angle_between(pointer_angle, min_angle, max_angle, clockwise=True):
        return clockwise_diff, True
    else:
        return counter_diff, False


def is_angle_between(angle, angle1, angle2, clockwise=True):
    """
    判断角度是否在angle1和angle2之间（按指定方向）
    clockwise=True: 检查是否在angle1顺时针到angle2的范围内
    """
    angle = angle % (2 * math.pi)
    angle1 = angle1 % (2 * math.pi)
    angle2 = angle2 % (2 * math.pi)

    if clockwise:
        # 顺时针方向：angle1 -> angle2（可能angle2 < angle1，需跨0点）
        if angle1 <= angle2:
            return angle1 <= angle <= angle2
        else:
            return angle >= angle1 or angle <= angle2
    else:
        # 逆时针方向：angle1 -> angle2（可能angle2 > angle1，需跨0点）
        if angle1 >= angle2:
            return angle2 <= angle <= angle1
        else:
            return angle >= angle1 or angle <= angle2


def filter_outliers(readings, threshold=0.2):
    """过滤超出平均值20%的异常值"""
    if len(readings) < 2:
        return readings

    filtered = []
    for i in range(len(readings)):
        valid = True
        for j in range(len(readings)):
            if i == j:
                continue
            a, b = readings[i], readings[j]
            if a == 0 and b == 0:
                continue
            # 计算相对差异
            avg = (a + b) / 2
            diff = abs(a - b)
            if diff / avg > threshold:
                valid = False
                break
        if valid:
            filtered.append(readings[i])

    # 如果过滤后为空，返回原始数据
    return filtered if filtered else readings


# 新增：辅助函数用于实现新的指针误检测逻辑
def perpendicular_line(line, center):
    """计算在圆心处与给定直线垂直的直线"""
    x1, y1, x2, y2 = line
    cx, cy = center

    # 计算原直线的方向向量
    dx = x2 - x1
    dy = y2 - y1

    # 垂直方向向量（旋转90度）
    perp_dx = -dy
    perp_dy = dx

    # 延展垂线到图像边缘
    img_width = 1920  # 假设图像宽度
    img_height = 1080  # 假设图像高度

    # 计算垂线方程参数
    if perp_dx != 0:
        slope = perp_dy / perp_dx
        intercept = cy - slope * cx

        # 计算与图像边缘的交点
        x_left = 0
        y_left = int(slope * x_left + intercept)
        x_right = img_width
        y_right = int(slope * x_right + intercept)

        # 调整确保点在图像范围内
        if y_left < 0 or y_left >= img_height:
            y_left = 0 if y_left < 0 else img_height - 1
            x_left = int((y_left - intercept) / slope)
        if y_right < 0 or y_right >= img_height:
            y_right = 0 if y_right < 0 else img_height - 1
            x_right = int((y_right - intercept) / slope)
    else:
        # 垂直线情况
        x_left = cx
        y_left = 0
        x_right = cx
        y_right = img_height - 1

    return (x_left, y_left, x_right, y_right)


def point_region(point, line):
    """判断点在直线的哪一侧（区域0或1）"""
    x, y = point
    x1, y1, x2, y2 = line

    # 计算点到直线的位置（使用叉积）
    val = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    # 根据符号判断区域
    return 0 if val >= 0 else 1


def line_segment_in_region(line, region_line, region):
    """计算线段在指定区域内的长度"""
    x1, y1, x2, y2 = line
    segment_length = line_length(line)

    # 检查线段端点是否在目标区域
    p1_region = point_region((x1, y1), region_line)
    p2_region = point_region((x2, y2), region_line)

    if p1_region == region and p2_region == region:
        return segment_length  # 整条线段在区域内
    elif p1_region != region and p2_region != region:
        return 0  # 整条线段不在区域内
    else:
        # 线段穿过区域边界，计算交点
        intersection = line_intersection(line, region_line)
        if not intersection:
            return 0

        ix, iy = intersection

        # 计算在区域内的子线段长度
        if p1_region == region:
            return line_length((x1, y1, ix, iy))
        else:
            return line_length((ix, iy, x2, y2))


# 全局变量
frame_count = 0
detection_history = []  # 存储最近5次模型检测的类别
valid_classes = set()  # 最终验证有效的类别（5次中出现≥2次）
current_detections = []  # 存储最近一次模型检测的结果
last_valid_data = {}  # 存储每个表盘最后一次正确的检测数据（无任何误检测）
current_plate_states = []  # 当前绘制的表盘状态（始终为正确数据）
detection_count = 0  # 模型检测次数计数器

# 读数相关变量
temp_readings = []  # 临时读数存储
final_readings = {}  # 最终读数
reading_update_counter = 0  # 读数更新计数器

# 帧率计数器
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧")
        break

    frame_count += 1
    fps_frame_count += 1

    # 计算实时帧率（每10帧更新一次）
    if fps_frame_count % 10 == 0:
        current_fps = fps_frame_count / (time.time() - fps_start_time)
        fps_frame_count = 0
        fps_start_time = time.time()

    # 模型检测：每detection_interval帧执行一次
    if frame_count % detection_interval == 0:
        results = model(frame)
        temp_detections = []
        current_classes = set()

        # 处理模型检测结果
        for result in results:
            for box in result.boxes:
                cls_idx = int(box.cls.item())
                cls_name = index_to_class.get(cls_idx, "unknown")
                conf = box.conf.item()

                if conf >= conf_thresholds.get(cls_name, 0.5):
                    current_classes.add(cls_name)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    temp_detections.append((x1, y1, x2, y2, cls_name, conf))

        # 更新检测历史（只保留最近5次）
        detection_history.append(current_classes)
        if len(detection_history) > 5:
            detection_history.pop(0)

        current_detections = temp_detections
        detection_count += 1

        # 每5次模型检测，执行一次OpenCV检测逻辑并更新数据
        if detection_count % 5 == 0:
            print(f"\n===== 第 {detection_count} 次模型检测完成，开始执行OpenCV检测 =====")

            # 统计前5次模型检测中有效的类别（出现≥2次）
            class_counts = {}
            for classes in detection_history:
                for cls_name in classes:
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            valid_classes = {cls_name for cls_name, count in class_counts.items() if count >= 2}
            print(f"前5次模型检测验证有效的类别: {valid_classes}")

            # 临时存储本次OpenCV检测的结果（仅正确的结果会被保留）
            temp_plate_states = []
            plate_detections = [d for d in current_detections if d[4] == 'plate']

            # 处理每个表盘
            for plate_idx, plate_det in enumerate(plate_detections):
                plate_x1, plate_y1, plate_x2, plate_y2, _, _ = plate_det

                new_plate_state = {
                    'plate_box': (plate_x1, plate_y1, plate_x2, plate_y2),
                    'center_box': None, 'pointer_box': None, 'digit_detections': [],
                    'combined_numbers': [], 'combined_centers': [],
                    'max_val_center': None, 'current_range': global_range,  # 改用全局量程
                    'plate_circle': None, 'center_circle': None, 'final_circle': None,
                    'pointer_tip': None, 'reading_point': None, 'scale_lines': [],
                    'scale_binary': None, 'inner_radius': 0, 'digit_centers': [],
                    'max_scale_point': None,  # 仅保留最大刻度点
                    'max_scale_line': None,  # 仅保留最大刻度线
                    'pointer_line': None,  # 指针直线
                    'range_angle': 0, 'pointer_angle': 0,  # 角度
                    'current_reading': 0,  # 当前读数
                    'range_direction': True,  # 量程角度方向（True=顺时针）
                    'perpendicular_line': None,  # 垂线
                    'reading_angle': 0,  # 用于查表的读数角度
                    'is_symmetric': False,  # 是否对称表盘
                    'reading_region': 0  # 读数所在区域（0:右,1:左）
                }

                # 提取表盘区域内的检测结果
                for det in current_detections:
                    d_x1, d_y1, d_x2, d_y2, cls_name, _ = det
                    d_center = ((d_x1 + d_x2) // 2, (d_y1 + d_y2) // 2)

                    if (plate_x1 <= d_center[0] <= plate_x2 and
                            plate_y1 <= d_center[1] <= plate_y2):
                        if cls_name == 'center':
                            new_plate_state['center_box'] = (d_x1, d_y1, d_x2, d_y2)
                        elif cls_name == 'pointer':
                            new_plate_state['pointer_box'] = (d_x1, d_y1, d_x2, d_y2)
                        elif cls_name in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                            new_plate_state['digit_detections'].append(det)
                            new_plate_state['digit_centers'].append((d_center[0], d_center[1]))

                # 组合数字（仅当检测到数字类别）
                if '0' in valid_classes or '1' in valid_classes:
                    new_plate_state['combined_numbers'], new_plate_state['combined_centers'] = combine_digits(
                        new_plate_state['digit_detections'])

                    if new_plate_state['combined_numbers']:
                        print(
                            f"数字组合结果: {new_plate_state['combined_numbers']}，中心点: {new_plate_state['combined_centers']}")

                        # 检查是否为对称表盘（存在两组相同的数字组合）
                        number_counts = {}
                        for num in new_plate_state['combined_numbers']:
                            number_counts[num] = number_counts.get(num, 0) + 1
                        # 存在出现至少两次的数字视为可能对称
                        symmetric_nums = [num for num, cnt in number_counts.items() if cnt >= 2]
                        new_plate_state['is_symmetric'] = len(symmetric_nums) > 0

                # 圆检测（仅当plate类别有效）
                circle_error = False
                if 'plate' in valid_classes and new_plate_state['plate_box']:
                    x1, y1, x2, y2 = new_plate_state['plate_box']
                    margin = 10
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    plate_roi = frame[y1:y2, x1:x2]

                    if plate_roi.size > 0:
                        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
                        circles = cv2.HoughCircles(
                            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                            param1=100, param2=30, minRadius=20, maxRadius=0
                        )

                        largest_circle = None
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for circle in circles[0, :]:
                                if largest_circle is None or circle[2] > largest_circle[2]:
                                    largest_circle = circle

                        if largest_circle is not None:
                            cx, cy, r = largest_circle
                            new_plate_state['plate_circle'] = (cx + x1, cy + y1, r)
                        else:
                            circle_error = True
                            print("plate圆检测失败（未找到圆）")
                    else:
                        circle_error = True
                        print("plate ROI无效（超出图像范围）")

                # 中心圆检测（仅当center类别有效）
                if 'center' in valid_classes and new_plate_state['center_box']:
                    x1, y1, x2, y2 = new_plate_state['center_box']
                    margin = 5
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    center_roi = frame[y1:y2, x1:x2]

                    if center_roi.size > 0:
                        gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
                        circles = cv2.HoughCircles(
                            blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=30,
                            param1=100, param2=20, minRadius=5, maxRadius=50
                        )

                        largest_circle = None
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for circle in circles[0, :]:
                                if largest_circle is None or circle[2] > largest_circle[2]:
                                    largest_circle = circle

                        if largest_circle is not None:
                            cx, cy, r = largest_circle
                            new_plate_state['center_circle'] = (cx + x1, cy + y1, r)
                        else:
                            circle_error = True
                            print("center圆检测失败（未找到圆）")
                    else:
                        circle_error = True
                        print("center ROI无效（超出图像范围）")

                # 计算最终圆
                if new_plate_state['plate_circle'] and new_plate_state['center_circle']:
                    px, py, pr = new_plate_state['plate_circle']
                    cx, cy, cr = new_plate_state['center_circle']
                    new_plate_state['final_circle'] = (int((px + cx) / 2), int((py + cy) / 2), pr)
                elif new_plate_state['plate_circle']:
                    new_plate_state['final_circle'] = new_plate_state['plate_circle']
                elif new_plate_state['center_circle']:
                    new_plate_state['final_circle'] = new_plate_state['center_circle']
                else:
                    circle_error = True
                    print("plate和center均未检测到圆，无法计算最终圆")

                # 指针直线检测（仅当pointer类别有效且无圆错误）
                pointer_error = False
                if not circle_error and 'pointer' in valid_classes:
                    if not new_plate_state['pointer_box']:
                        pointer_error = True
                        print("指针框不存在（无法检测指针）")
                    elif not new_plate_state['final_circle']:
                        pointer_error = True
                        print("无最终圆（无法验证指针）")
                    else:
                        x1, y1, x2, y2 = new_plate_state['pointer_box']
                        margin = 5
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(frame.shape[1], x2 + margin)
                        y2 = min(frame.shape[0], y2 + margin)
                        pointer_roi = frame[y1:y2, x1:x2]

                        if pointer_roi.size == 0:
                            pointer_error = True
                            print("指针ROI无效（超出图像范围）")
                        else:
                            gray = cv2.cvtColor(pointer_roi, cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(gray, 50, 150)
                            lines = cv2.HoughLinesP(
                                edges, rho=1, theta=np.pi / 180, threshold=50,
                                minLineLength=20, maxLineGap=10
                            )

                            if lines is None:
                                pointer_error = True
                                print("未检测到任何直线（指针区域）")
                            else:
                                lines = [line[0] for line in lines]
                                if len(lines) < 2:
                                    pointer_error = True
                                    print(f"直线数量不足（仅检测到{len(lines)}条，需至少2条）")
                                else:
                                    # 按长度排序取前两条最长直线
                                    lines_sorted = sorted(lines, key=line_length, reverse=True)
                                    line1, line2 = lines_sorted[0], lines_sorted[1]
                                    line1_len = line_length(line1)
                                    line2_len = line_length(line2)
                                    final_r = new_plate_state['final_circle'][2]  # 最终圆半径

                                    # 核心判定1：任意一条直线长度 < 最终圆半径的1/2 → 误检测
                                    if line1_len < final_r / 2 or line2_len < final_r / 2:
                                        pointer_error = True
                                        print(
                                            f"直线长度不足（line1: {line1_len:.1f}, line2: {line2_len:.1f}, 需≥{final_r / 2:.1f}）")
                                    else:
                                        # 验证直线交点（指针尖端）
                                        roi_shape = (pointer_roi.shape[0], pointer_roi.shape[1])
                                        ext_line1 = extend_line(line1, roi_shape)
                                        ext_line2 = extend_line(line2, roi_shape)
                                        intersect = line_intersection(ext_line1, ext_line2)

                                        if intersect is None:
                                            pointer_error = True
                                            print("两条直线平行（无交点）")
                                        else:
                                            ix, iy = intersect
                                            pointer_tip = (int(ix) + x1, int(iy) + y1)
                                            center_point = (
                                                new_plate_state['final_circle'][0], new_plate_state['final_circle'][1])

                                            # 核心判定2：指针尖端到圆心的距离 < 最终圆半径的1/2 → 误检测
                                            tip_to_center_dist = distance_between_points(pointer_tip, center_point)
                                            if tip_to_center_dist < final_r / 2:
                                                pointer_error = True
                                                print(
                                                    f"指针尖端到圆心距离不足（{tip_to_center_dist:.1f}, 需≥{final_r / 2:.1f}）")
                                            else:
                                                # 新增：基于垂线区域划分的误检测判断
                                                # 1. 创建指针线（从圆心到指针尖端）
                                                pointer_line = (
                                                    center_point[0], center_point[1], pointer_tip[0], pointer_tip[1])

                                                # 2. 计算垂线
                                                perp_line = perpendicular_line(pointer_line, center_point)
                                                new_plate_state['perpendicular_line'] = perp_line

                                                # 3. 转换直线坐标到原始图像
                                                line1_orig = (
                                                    line1[0] + x1, line1[1] + y1, line1[2] + x1, line1[3] + y1)
                                                line2_orig = (
                                                    line2[0] + x1, line2[1] + y1, line2[2] + x1, line2[3] + y1)

                                                # 4. 计算两条直线在两个区域的总长度
                                                region0_len = (line_segment_in_region(line1_orig, perp_line, 0) +
                                                               line_segment_in_region(line2_orig, perp_line, 0))
                                                region1_len = (line_segment_in_region(line1_orig, perp_line, 1) +
                                                               line_segment_in_region(line2_orig, perp_line, 1))

                                                # 5. 确定主要区域
                                                main_region = 0 if region0_len >= region1_len else 1
                                                print(
                                                    f"区域0长度: {region0_len:.1f}, 区域1长度: {region1_len:.1f}, 主要区域: {main_region}")

                                                # 6. 计算读数点并判断其区域
                                                reading_point = find_intersection_with_circle(
                                                    center_point, final_r, center_point, pointer_tip)

                                                if reading_point:
                                                    reading_region = point_region(reading_point, perp_line)
                                                    print(f"读数点区域: {reading_region}")
                                                    new_plate_state['reading_region'] = reading_region  # 记录读数区域

                                                    # 7. 如果读数点区域与主要区域不同，则判定为误检测
                                                    if reading_region != main_region:
                                                        pointer_error = True
                                                        print(
                                                            f"指针误检测: 读数点在区域{reading_region}，而直线主要在区域{main_region}")
                                                else:
                                                    pointer_error = True
                                                    print("无法计算读数点，判定为指针误检测")

                                                if not pointer_error:
                                                    new_plate_state['pointer_tip'] = pointer_tip
                                                    # 记录指针直线（从圆心到指针尖端）
                                                    new_plate_state['pointer_line'] = pointer_line
                elif 'pointer' in valid_classes:
                    pointer_error = True
                    print("因圆检测错误，跳过指针检测")

                # 刻度线检测（仅当无任何误检测）
                if not (circle_error or pointer_error) and 'plate' in valid_classes:
                    if new_plate_state['final_circle'] and new_plate_state['plate_box']:
                        cx, cy, r = new_plate_state['final_circle']
                        plate_box = new_plate_state['plate_box']
                        p_x1, p_y1, p_x2, p_y2 = plate_box
                        plate_roi = frame[p_y1:p_y2, p_x1:p_x2]

                        if plate_roi.size > 0:
                            center_in_roi = (cx - p_x1, cy - p_y1)
                            digit_boxes_in_roi = []
                            for det in new_plate_state['digit_detections']:
                                d_x1, d_y1, d_x2, d_y2, _, _ = det
                                roi_d_x1 = d_x1 - p_x1
                                roi_d_y1 = d_y1 - p_y1
                                roi_d_x2 = d_x2 - p_x1
                                roi_d_y2 = d_y2 - p_y1
                                digit_boxes_in_roi.append((roi_d_x1, roi_d_y1, roi_d_x2, roi_d_y2))

                            scale_lines, scale_binary, inner_radius = detect_scale_lines(
                                plate_roi, center_in_roi, r, plate_box, digit_boxes_in_roi
                            )

                            new_plate_state['scale_lines'] = scale_lines
                            new_plate_state['scale_binary'] = scale_binary
                            new_plate_state['inner_radius'] = inner_radius

                            # 仅保留最大刻度点检测（删除最小刻度相关）
                            if new_plate_state['combined_centers'] and new_plate_state['final_circle']:
                                cx_final, cy_final, r_final = new_plate_state['final_circle']
                                final_center = (cx_final, cy_final)  # 最终圆圆心
                                max_search_radius = int(r_final * 0.8)  # 最大搜索半径为表盘半径的80%

                                # 对称表盘处理：根据读数区域选择对应区域的数字组合
                                target_numbers = new_plate_state['combined_numbers']
                                target_centers = new_plate_state['combined_centers']

                                if new_plate_state['is_symmetric'] and new_plate_state['final_circle']:
                                    # y轴为过圆心的竖直线
                                    y_axis = (cx_final, 0, cx_final, frame.shape[0])  # 竖直线
                                    reading_region = new_plate_state['reading_region']

                                    # 筛选读数区域内的数字组合（0:右半区,1:左半区）
                                    region_centers = []
                                    region_numbers = []
                                    for num, center in zip(target_numbers, target_centers):
                                        # 判断数字中心在y轴哪一侧
                                        center_region = 0 if center[0] >= cx_final else 1
                                        if center_region == reading_region:
                                            region_centers.append(center)
                                            region_numbers.append(num)

                                    if region_numbers:  # 若有区域内数字则使用
                                        target_numbers = region_numbers
                                        target_centers = region_centers
                                        print(f"对称表盘，使用区域{reading_region}的数字组合: {target_numbers}")

                                # 寻找最大值对应的刻度点（使用筛选后的数字组合）
                                if target_numbers:
                                    max_val = max(target_numbers)
                                    # 找到最大值对应的中心
                                    max_indices = [i for i, num in enumerate(target_numbers) if num == max_val]
                                    if max_indices:
                                        max_center = target_centers[max_indices[0]]
                                        new_plate_state['max_val_center'] = max_center

                                        # 检测最大刻度点
                                        max_scale_point, _ = find_scale_intersection(
                                            max_center,  # 数字组合中心
                                            scale_lines,  # 刻度线
                                            max_search_radius,  # 最大搜索半径
                                            final_center  # 最终圆圆心（用于距离判断）
                                        )
                                        new_plate_state['max_scale_point'] = max_scale_point

                                        # 记录最大刻度线（从圆心到刻度点）
                                        if max_scale_point:
                                            new_plate_state['max_scale_line'] = (
                                                cx_final, cy_final, max_scale_point[0], max_scale_point[1])

                # 计算读数点（仅当无任何误检测）
                if not (circle_error or pointer_error) and new_plate_state['final_circle'] and new_plate_state[
                    'pointer_tip']:
                    center_point = (new_plate_state['final_circle'][0], new_plate_state['final_circle'][1])
                    radius = new_plate_state['final_circle'][2]

                    # 指针线段的起点（圆心）和终点（指针尖端）
                    line_start = center_point
                    line_end = new_plate_state['pointer_tip']

                    # 计算交点
                    reading_point = find_intersection_with_circle(center_point, radius, line_start, line_end)
                    new_plate_state['reading_point'] = reading_point

                    # 计算角度和读数（使用新的不均匀刻度计算方法）
                    if (new_plate_state['max_scale_line'] and new_plate_state['pointer_line'] and angle_table):
                        # 获取各直线与X轴的夹角（弧度，范围[0, 2π)）
                        max_angle = line_angle_with_x_axis(new_plate_state['max_scale_line'])
                        pointer_angle = line_angle_with_x_axis(new_plate_state['pointer_line'])

                        # 获取最大量程角（从角度表中获取）
                        max_range_angle = angle_table[-1][1] if angle_table else 0

                        # 计算指针直线与最大刻度直线两直线所成角度
                        angle_between = abs(pointer_angle - max_angle)

                        # 计算读数角度：最大量程角 - 两直线所成角度
                        reading_angle = max_range_angle - angle_between
                        new_plate_state['reading_angle'] = reading_angle

                        # 根据读数角度从对照表中查找对应的读数
                        new_plate_state['current_reading'] = get_reading_from_angle(reading_angle, angle_table)

                        print(
                            f"角度计算: 最大量程角={max_range_angle:.2f}rad, 两直线夹角={angle_between:.2f}rad")
                        print(
                            f"读数计算: 读数角度={reading_angle:.2f}rad, 结果={new_plate_state['current_reading']:.2f}")
                    elif not angle_table:
                        print("警告：未加载角度对照表，无法计算读数")

                # 误检测处理
                if circle_error or pointer_error:
                    print(f"存在误检测（circle_error: {circle_error}, pointer_error: {pointer_error}），使用历史数据")
                    if plate_det in last_valid_data:
                        # 沿用最近一次正确数据
                        temp_plate_states.append(last_valid_data[plate_det].copy())
                        print("已加载历史正确数据")
                    else:
                        print("无历史正确数据，不更新该表盘")
                else:
                    print("检测完全有效，更新当前数据")
                    temp_plate_states.append(new_plate_state)
                    last_valid_data[plate_det] = new_plate_state.copy()  # 仅保存正确数据

            # 更新当前绘制数据（仅保留有效结果）
            if temp_plate_states:
                current_plate_states = temp_plate_states

                # 收集临时读数
                for i, plate in enumerate(current_plate_states):
                    if plate['current_reading'] > 0:  # 确保读数有效
                        temp_readings.append(plate['current_reading'])
                        print(f"临时读数 #{len(temp_readings)}: {plate['current_reading']:.2f}")

                reading_update_counter += 1

                # 每5次更新计算一次滑动平均
                if reading_update_counter % 5 == 0 and temp_readings:
                    print("\n===== 计算滑动平均读数 =====")
                    print(f"原始读数: {[round(r, 2) for r in temp_readings]}")

                    # 过滤异常值
                    filtered = filter_outliers(temp_readings)
                    print(f"过滤后读数: {[round(r, 2) for r in filtered]}")

                    # 计算平均值
                    if filtered:
                        avg_reading = sum(filtered) / len(filtered)
                        final_readings[plate_idx] = avg_reading
                        print(f"滑动平均读数: {avg_reading:.2f}\n")
                    else:
                        print("无有效读数进行平均计算\n")

                    # 重置临时读数
                    temp_readings = []

        print(f"第 {detection_count} 次数据更新完成\n")

    # 绘制图像（始终使用正确数据）
    display_frame = frame.copy()
    for i, plate_state in enumerate(current_plate_states):
        plate_x1, plate_y1, plate_x2, plate_y2 = plate_state['plate_box']

        # 绘制表盘框
        cv2.rectangle(display_frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"Plate {i + 1}", (plate_x1, plate_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制plate圆
        if plate_state['plate_circle']:
            cx, cy, r = plate_state['plate_circle']
            cv2.circle(display_frame, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(display_frame, (cx, cy), 2, (0, 255, 0), 3)

        # 绘制center圆
        if plate_state['center_circle']:
            cx, cy, r = plate_state['center_circle']
            cv2.circle(display_frame, (cx, cy), r, (255, 0, 0), 2)
            cv2.circle(display_frame, (cx, cy), 2, (255, 0, 0), 3)

        # 绘制最终圆和内圈圆
        if plate_state['final_circle']:
            cx, cy, r = plate_state['final_circle']
            cv2.circle(display_frame, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(display_frame, (cx, cy), r, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            inner_r = plate_state['inner_radius']
            cv2.circle(display_frame, (cx, cy), int(inner_r), (255, 0, 255), 2, lineType=cv2.LINE_AA)

            # 绘制最小距离阈值线（调试用）
            min_valid_r = r / 2
            cv2.circle(display_frame, (cx, cy), int(min_valid_r), (0, 128, 255), 1, lineType=cv2.LINE_AA)  # 橙色虚线
            cv2.putText(display_frame, "Min Valid R", (cx + int(min_valid_r), cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

            # 绘制y轴（对称判断参考线）
            if plate_state['is_symmetric']:
                cv2.line(display_frame, (cx, 0), (cx, frame.shape[0]), (0, 255, 255), 2)  # 黄色y轴

        # 绘制指针
        if plate_state['final_circle'] and plate_state['pointer_tip']:
            cx, cy, _ = plate_state['final_circle']
            cv2.line(display_frame, (cx, cy), plate_state['pointer_tip'], (0, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(display_frame, plate_state['pointer_tip'], 5, (0, 0, 255), -1)

        # 绘制读数点
        if plate_state['reading_point']:
            x, y = plate_state['reading_point']
            cv2.circle(display_frame, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(display_frame, (x, y), 8, (0, 255, 0), 2)

        # 显示量程（使用角度表中的量程）
        cv2.putText(display_frame, f"Range: {plate_state['current_range']}",
                    (plate_x1, plate_y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 仅标注最大值中心（删除最小值相关）
        if plate_state['max_val_center']:
            cx, cy = plate_state['max_val_center']
            cv2.circle(display_frame, (cx, cy), 10, (0, 0, 255), 3)
            cv2.putText(display_frame, "Max", (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 绘制刻度线
        for line in plate_state['scale_lines']:
            x1, y1, x2, y2 = line
            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # 橙色刻度线

        # 仅绘制最大刻度点和刻度线（删除最小刻度相关）
        if plate_state['max_scale_point']:
            x, y = plate_state['max_scale_point']
            cv2.circle(display_frame, (x, y), 6, (0, 0, 255), -1)  # 红色最大刻度点
        if plate_state['max_scale_line']:
            x1, y1, x2, y2 = plate_state['max_scale_line']
            cv2.line(display_frame, (x1, y1), (x2, y2), (100, 100, 255), 2, lineType=cv2.LINE_AA)  # 浅红色最大刻度线

        # 绘制垂线（调试用）
        if plate_state['perpendicular_line']:
            x1, y1, x2, y2 = plate_state['perpendicular_line']
            cv2.line(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2, lineType=cv2.LINE_AA)  # 青色垂线

        # 显示当前读数和滑动平均读数
        reading_text = f"Reading: {plate_state['current_reading']:.2f}"
        cv2.putText(display_frame, reading_text,
                    (plate_x1, plate_y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if i in final_readings:
            avg_text = f"Avg: {final_readings[i]:.2f}"
            cv2.putText(display_frame, avg_text,
                        (plate_x1, plate_y1 - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示角度信息（调试用）
        angle_text = f"Reading Angle: {math.degrees(plate_state['reading_angle']):.2f}°"
        cv2.putText(display_frame, angle_text,
                    (plate_x1, plate_y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        ptr_text = f"Pointer: {math.degrees(plate_state['pointer_angle']):.2f}°"
        cv2.putText(display_frame, ptr_text,
                    (plate_x1, plate_y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        # 显示对称信息
        if plate_state['is_symmetric']:
            sym_text = f"Symmetric (Region: {plate_state['reading_region']})"
            cv2.putText(display_frame, sym_text,
                        (plate_x1, plate_y2 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 显示刻度线二值图（调试）
    for plate_state in current_plate_states:
        if plate_state.get('scale_binary') is not None:
            scale_binary = plate_state['scale_binary']
            if scale_binary.size > 0:
                scale_binary_color = cv2.cvtColor(scale_binary, cv2.COLOR_GRAY2BGR)
                scale_binary_resized = cv2.resize(scale_binary_color, (320, 240))
                cv2.imshow('Scale Binary', scale_binary_resized)

    # 显示状态信息
    cv2.putText(display_frame, f"FPS: {current_fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(display_frame, f"Detection Count: {detection_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(display_frame, f"Valid Classes: {', '.join(valid_classes)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Pointer Meter Reading', display_frame)

    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()