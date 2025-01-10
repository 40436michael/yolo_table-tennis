from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')  

# 影片路徑與輸出
input_video_path = "table_tennis.mp4"  
output_video_path = "output_tracking.mp4"

# 顯示 YOLO 模型類型
print(model.names)

# 讀取影片
cap = cv2.VideoCapture(input_video_path)

# 獲取影片的參數
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


ball_points = []    # 用於儲存桌球軌跡點的列表
hit_count = 0   # 用於計算擊球次數
last_y_v = None  # 儲存上一幀的 Y 軸速度
isdown = False  # 標記桌球是否處於下降狀態
last_hit_position = None  # 最近一次擊球的中心點
height_difference = 0  # 當前球與擊球點的高度差

# 逐幀處理影片
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if not cap.isOpened():
        break

    # 使用 YOLO 模型進行偵測
    results = model(frame)

    # 獲取偵測到的物件
    detections = results[0].boxes.xyxy  # 邊界框 (x1, y1, x2, y2)
    class_ids = results[0].boxes.cls  # 類別索引
    confidences = results[0].boxes.conf  # 置信度

    for bbox, class_id, confidence in zip(detections, class_ids, confidences):
        if int(class_id) == 32:  # 只處理類別索引為 32 的物件（sports ball）
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 儲存軌跡點
            ball_points.append((cx, cy))

            # 畫出邊界框和中心點
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # 計算 Y 軸速度
            if len(ball_points) > 1:
                prev_cx, prev_cy = ball_points[-2]
                y_v = cy - prev_cy
                
                # Y 座標越大，位置越靠下
                
                # 判斷是否發生擊球
                if last_y_v is not None:
                    if y_v > 0:  # 目前速度為正（下降中）
                        isdown = True
                    elif y_v < 0 and isdown:  # 速度變負且之前處於下降狀態
                        hit_count += 1
                        last_hit_position = (cx, cy)  # 更新最近一次擊球點
                        isdown = False  # 重置狀態
                        print(f"擊球事件：第 {hit_count} 次，位置：({cx}, {cy})")

                last_y_v = y_v

            # 計算與最近一次擊球點的高度差
            if last_hit_position:
                height_difference = abs(cy - last_hit_position[1])

    # 繪製桌球的軌跡
    for i in range(1, len(ball_points)):
        if ball_points[i - 1] is None or ball_points[i] is None:
            continue
        cv2.line(frame, ball_points[i - 1], ball_points[i], (255, 0, 0), 2)

    # 在右上角顯示擊球次數
    hit_text = f"Hits: {hit_count}"
    (text_width, text_height), baseline = cv2.getTextSize(hit_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    x_position = frame_width - text_width - 10  
    y_position = text_height + 10  
    cv2.putText(frame, hit_text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 在左上角顯示與最近擊球點的高度差
    height_text = f"Height Diff: {height_difference:.1f}"
    cv2.putText(frame, height_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 寫入影片
    out.write(frame)
    
    # 顯示結果（可選）
    cv2.imshow('Ball Tracking', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
print("總擊球次數為:", hit_count)
