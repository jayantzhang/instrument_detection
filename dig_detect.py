from ultralytics import YOLO
import cv2
from collections import defaultdict
import os
import shutil


# ?? YOLOv10 ??
model = YOLO(r"model/digital_clock.onnx")

# ????????????
index_to_class = {
    0: '-', 1: '0', 2: '1', 3: '2', 4: '3',
    5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: '.'
}

# ????
cap = None
save_dir = "runs/detect"
cap_dir = os.path.join(save_dir, "cap")  # ?????????
local_dir = os.path.join(save_dir, "local")  # ????????
save_counter = 1
result_counter = 1
frame_count = 0  # ??frame_count???
detection_counter = 0  # ?????
detection_history = []  # ??detection_history???

# ??????
DETECTION_INTERVAL = 10  # ?5?????
TOTAL_DETECTIONS = 5    # ????5?
CONSECUTIVE_THRESHOLD = 2  # ????2??????????
POSITION_THRESHOLD = 0.5  # ??????????????

# ??????????????
current_display_result = ""  # ?????????5??????
valid_result = ""  # ???????????
last_detection = ""  # ??????????
valid_length = 0  # ??????????????-?.?
initial_detection = True  # ???????


def initialize_camera():
    """??????"""
    global cap
    cap = cv2.VideoCapture(21)  # 0 ???????
    if not cap.isOpened():
        print("???????")
        exit()


def release_camera():
    """???????"""
    global cap
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


# def process_local_image():
#     """??????????????"""
#     global result_counter
#
#     root = tk.Tk()
#     root.withdraw()
#     file_path = filedialog.askopenfilename(
#         title="??????",
#         filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
#     )
#
#     if not file_path:
#         return
#
#     max_attempts = 1000
#     attempt = 0
#     result_folder = None
#     original_result_counter = result_counter
#
#     while attempt < max_attempts:
#         result_folder = os.path.join(local_dir, f"predict{result_counter}")
#         if not os.path.exists(result_folder):
#             os.makedirs(result_folder, exist_ok=True)
#             break
#         result_counter += 1
#         attempt += 1
#     else:
#         print(f"???{max_attempts}????????????????predict{original_result_counter}")
#         return
#
#     results = model(file_path, save=True, save_txt=True, project=result_folder, name="")
#
#     detections = []
#     for result in results:
#         annotated_frame = result.plot()
#         boxes = result.boxes
#         for box in boxes:
#             confidence = float(box.conf)
#             if confidence <= 0.40:
#                 continue
#             class_index = int(box.cls)
#             x_center = (box.xyxy[0, 0] + box.xyxy[0, 2]) / 2
#             detections.append((class_index, x_center))
#
#     sorted_detections = sorted(detections, key=lambda x: x[1])
#     sorted_classes_str = ''.join(index_to_class[detection[0]] for detection in sorted_detections)
#     print("????????:", sorted_classes_str)
#
#     cv2.putText(annotated_frame, f"Result: {sorted_classes_str}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     result_path = os.path.join(result_folder, "result.jpg")
#     cv2.imwrite(result_path, annotated_frame)
#
#     cv2.imshow("Local Image Detection Results", annotated_frame)
#     cv2.waitKey(0)
#     cv2.destroyWindow("Local Image Detection Results")
#     result_counter += 1


def capture_and_process_frame(frame):
    """??????????????"""
    global save_counter, current_display_result

    max_attempts = 1000
    attempt = 0
    save_folder = None
    temp_name = None
    original_save_counter = save_counter

    while attempt < max_attempts:
        save_folder = os.path.join(cap_dir, f"predict{save_counter}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
            temp_name = f"cap/predict{save_counter}/temp_predict{save_counter}"
            break
        save_counter += 1
        attempt += 1
    else:
        print(f"???{max_attempts}????????????????predict{original_save_counter}")
        return ""

    original_path = os.path.join(save_folder, "original.jpg")
    cv2.imwrite(original_path, frame)

    results = model(frame, save=True, save_txt=True, project=save_dir, name=temp_name)
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf)
            if confidence <= 0.40:
                continue
            class_index = int(box.cls)
            x_center = (box.xyxy[0, 0] + box.xyxy[0, 2]) / 2
            detections.append((class_index, x_center))

    sorted_detections = sorted(detections, key=lambda x: x[1])
    sorted_classes_str = ''.join(index_to_class[det[0]] for det in sorted_detections)
    print(f"??????: {sorted_classes_str}")

    temp_folder = os.path.join(save_dir, temp_name)
    if os.path.exists(temp_folder):
        labels_src = os.path.join(temp_folder, "labels")
        if os.path.exists(labels_src):
            labels_dst = os.path.join(save_folder, "labels")
            if os.path.exists(labels_dst):
                shutil.rmtree(labels_dst)
            os.rename(labels_src, labels_dst)
        for file in os.listdir(temp_folder):
            if file.endswith(".jpg"):
                src_path = os.path.join(temp_folder, file)
                dst_path = os.path.join(save_folder, "detection.jpg")
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                os.rename(src_path, dst_path)
                break
        shutil.rmtree(temp_folder)

    save_counter += 1
    current_display_result = sorted_classes_str  # ??????????
    return sorted_classes_str


def is_overlapping(box1, box2, threshold=0.5):
    """??????????????????????????"""
    x1_center = (box1[0] + box1[2]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    width1 = box1[2] - box1[0]
    width2 = box2[2] - box2[0]
    avg_width = (width1 + width2) / 2
    distance = abs(x1_center - x2_center)
    return distance <= avg_width * threshold


def merge_overlapping_boxes(boxes):
    """???????????????"""
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda x: x[2], reverse=True)
    final_boxes = []
    for box in sorted_boxes:
        overlapped = False
        for fb in final_boxes:
            if is_overlapping(box[0], fb[0]):
                overlapped = True
                break
        if not overlapped:
            final_boxes.append(box)
    return final_boxes


def count_valid_digits(result_str):
    """?????????-?.?"""
    valid_chars = [c for c in result_str if c not in ['-', '.']]
    return len(valid_chars)


def process_realtime_detection(frame):
    """????????????????"""
    global frame_count, detection_counter, detection_history
    global current_display_result, valid_result, last_detection, valid_length, initial_detection

    frame_count += 1
    current_detections = []  # ?????????

    # ?DETECTION_INTERVAL???????
    if frame_count % DETECTION_INTERVAL == 0:
        results = model(frame)
        detection_counter += 1

        # ???????????
        boxes_to_merge = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf)
                if confidence <= 0.40:
                    continue
                class_index = int(box.cls)
                box_coords = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                x_center = (box_coords[0] + box_coords[2]) / 2
                box_width = box_coords[2] - box_coords[0]
                boxes_to_merge.append((box_coords, class_index, confidence, x_center, box_width))

        merged_boxes = merge_overlapping_boxes(boxes_to_merge)
        current_detections = [(cls, x, conf, coords) for (coords, cls, conf, x, _) in merged_boxes]

        # ???????????
        detection_history.append(current_detections)

        # ?TOTAL_DETECTIONS????????
        if detection_counter == TOTAL_DETECTIONS:
            # 1. ???????????????????????????
            all_classes = defaultdict(lambda: defaultdict(list))  # all_classes[pos_idx][class_idx] = [(x_center, box_width), ...]
            for det in detection_history:
                sorted_det = sorted(det, key=lambda x: x[1])  # ?x????
                for pos_idx, (class_idx, x_center, _, box_coords) in enumerate(sorted_det):
                    box_width = box_coords[2] - box_coords[0]
                    all_classes[pos_idx][class_idx].append((x_center, box_width))

            # 2. ?????????-?.??????2??????
            special_chars = []
            # ?????class_index=0?
            neg_positions = []
            for pos_idx in all_classes:
                if 0 in all_classes[pos_idx]:  # 0??????
                    neg_positions = all_classes[pos_idx][0]
                    break
            if len(neg_positions) >= CONSECUTIVE_THRESHOLD:
                is_stable = True
                first_x, first_w = neg_positions[0]
                for x, w in neg_positions[1:]:
                    if abs(x - first_x) / first_w > POSITION_THRESHOLD:
                        is_stable = False
                        break
                if is_stable:
                    special_chars.append(('-', min(p[0] for p in neg_positions)))  # ????????x

            # ????class_index=11?
            dot_positions = []
            for pos_idx in all_classes:
                if 11 in all_classes[pos_idx]:  # 11?????
                    dot_positions = all_classes[pos_idx][11]
                    break
            if len(dot_positions) >= CONSECUTIVE_THRESHOLD:
                is_stable = True
                first_x, first_w = dot_positions[0]
                for x, w in dot_positions[1:]:
                    if abs(x - first_x) / first_w > POSITION_THRESHOLD:
                        is_stable = False
                        break
                if is_stable:
                    special_chars.append(('.', min(p[0] for p in dot_positions)))  # ???????x

            # 3. ???????1-10???0-9??????2??????
            normal_digits = []
            for pos_idx in sorted(all_classes.keys()):
                class_counts = all_classes[pos_idx]
                # ????????1-10?
                digit_classes = {cls: cnts for cls, cnts in class_counts.items() if 1 <= cls <= 10}
                if not digit_classes:
                    continue
                # ??????????
                max_cls = max(digit_classes.items(), key=lambda x: len(x[1]))[0]
                positions = digit_classes[max_cls]
                if len(positions) >= CONSECUTIVE_THRESHOLD:
                    is_stable = True
                    first_x, first_w = positions[0]
                    for x, w in positions[1:]:
                        if abs(x - first_x) / first_w > POSITION_THRESHOLD:
                            is_stable = False
                            break
                    if is_stable:
                        normal_digits.append((index_to_class[max_cls], min(p[0] for p in positions)))  # ??????x

            # 4. ??????????x?????
            all_elements = special_chars + normal_digits
            all_elements_sorted = sorted(all_elements, key=lambda x: x[1])  # ?x????
            current_result = ''.join([elem[0] for elem in all_elements_sorted])

            # 5. ???????
            current_length = count_valid_digits(current_result)
            global valid_result, last_detection, valid_length, initial_detection

            if initial_detection:
                # ????????????
                valid_result = current_result
                valid_length = current_length
                last_detection = current_result
                initial_detection = False
            else:
                last_detection = current_result  # ????????
                if current_length == valid_length:
                    # ???????????
                    valid_result = current_result
                    valid_length = current_length
                else:
                    # ???????????????????
                    last_det_length = count_valid_digits(last_detection)
                    if current_length == last_det_length:
                        # ???????????????????
                        valid_result = current_result
                        valid_length = current_length

            # 6. ??????????????
            current_display_result = valid_result
            print(f"???????5????: {current_display_result}")

            # ??????
            detection_counter = 0
            detection_history = []

    # ?????????????????????????????
    for (class_idx, x_center, confidence, box_coords) in current_detections:
        x1, y1, x2, y2 = map(int, box_coords)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{index_to_class[class_idx]} {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ???????????????????5??????
    if current_display_result:
        cv2.putText(frame, f"Result:{current_display_result}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, current_display_result


def main():
    """???"""
    global cap, detection_history, detection_counter
    detection_history = []  # ???????
    detection_counter = 0   # ????????

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cap_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)

    initialize_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame,_ = process_realtime_detection(frame.copy())
        cv2.imshow('????', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            release_camera()
            # process_local_image()
            initialize_camera()
        elif key == ord('c'):
            capture_and_process_frame(frame.copy())
        elif key == ord('q'):
            break

    release_camera()


if __name__ == "__main__":
    main()