"""
FPS 人物检测 —— 使用 YOLO 实时检测画面中的人
================================================
这个脚本会打开你的摄像头，实时检测画面中的人（person），
并在每个人周围画框，显示 FPS 帧率。

使用方法：python person_detect.py
按 'q' 键退出
"""

from ultralytics import YOLO
import cv2


def main():
    # ============================================================
    # 第 1 步：加载 YOLO 模型
    # ============================================================
    # YOLO 模型有多种尺寸，n/s/m/l/x 分别代表：
    #   n = nano（最小最快，精度最低）
    #   s = small（小而快）
    #   m = medium（中等，平衡）
    #   l = large（大而准，但慢）
    #   x = xlarge（最大最准，最慢）
    #
    # 这里用 nano 版本，首次运行会自动下载约 6MB 的模型文件
    print("正在加载 YOLO 模型...")
    model = YOLO("yolo26n.pt")  # 使用最新的 YOLO26 nano 模型
    print("模型加载完成！\n")

    # ============================================================
    # 第 2 步：打开摄像头
    # ============================================================
    # 0 表示第一个摄像头（通常是内置摄像头或第一个 USB 摄像头）
    # 如果有多个摄像头，可以改成 1, 2 等
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 设置分辨率宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # 设置分辨率高度

    if not cap.isOpened():
        print("❌ 无法打开摄像头！请检查摄像头是否连接。")
        return

    print("✅ 摄像头已打开，开始检测...")
    print("按 'q' 键退出程序\n")

    # ============================================================
    # 第 3 步：实时循环检测
    # ============================================================
    while True:
        # 读取一帧画面
        success, frame = cap.read()
        if not success:
            print("读取画面失败")
            break

        # ----- YOLO 检测 -----
        # model() 返回一个结果列表，我们取第一个
        # verbose=False 关闭控制台输出，让画面更清爽
        results = model(frame, verbose=False)

        # ----- 过滤：只保留 person 类 -----
        # COCO 数据集中，person 的类别 ID 是 0
        # results[0].boxes 包含所有检测到的目标
        result = results[0]
        person_boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])  # 类别 ID
                if class_id == 0:  # 0 = person
                    person_boxes.append(box)

        # ----- 在画面上画框 -----
        # 创建一个带标注的画面
        annotated_frame = result.plot()  # YOLO 自带画框功能

        # 获取 FPS（处理速度）
        # speed 是一个包含预处理、推理、后处理时间的字典（单位：毫秒）
        speed = result.speed
        inference_time = speed.get("inference", 0)  # 推理时间（毫秒）
        fps = 1000 / inference_time if inference_time > 0 else 0

        # ----- 在画面左上角显示 FPS 和人数 -----
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f} | 检测到 {len(person_boxes)} 人",
            (10, 35),                               # 文字位置（左上角）
            cv2.FONT_HERSHEY_SIMPLEX,               # 字体
            1.0,                                     # 字体大小
            (0, 255, 0),                             # 绿色
            2,                                       # 线宽
        )

        # ----- 显示画面 -----
        cv2.imshow("YOLO 人物检测 - 按 Q 退出", annotated_frame)

        # 按 'q' 键退出（waitKey 返回按键的 ASCII 码）
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("用户按下了退出键")
            break

    # ============================================================
    # 第 4 步：清理资源
    # ============================================================
    cap.release()            # 释放摄像头
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
    print("程序已退出。")


if __name__ == "__main__":
    main()
