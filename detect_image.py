"""
YOLO 图片人物检测 —— 在单张图片中检测人
==========================================
不需要摄像头，直接用图片测试 YOLO。

使用方法：python detect_image.py
按任意键关闭结果窗口
"""

from ultralytics import YOLO
import cv2
import os


# ============================================================
# 如果你有自己的图片，把下面的路径改成你的图片路径
# ============================================================
# 使用 YOLO 自带的测试图片（会自动下载）
IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
# 或者使用本地图片：
# IMAGE_PATH = r"C:\Users\xiaob\Pictures\your_photo.jpg"


def main():
    # 加载模型
    print("正在加载 YOLO26n 模型...")
    model = YOLO("yolo26n.pt")
    print("模型加载完成！\n")

    # 加载图片
    # 如果图片在本地不存在，尝试从网络下载
    image_path = "test_image.jpg"

    if not os.path.exists(image_path):
        print("正在下载测试图片...")
        import urllib.request

        urllib.request.urlretrieve(IMAGE_URL, image_path)
        print(f"测试图片已保存到: {image_path}")

    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ 无法读取图片！")
        return

    print(f"图片尺寸: {frame.shape[1]} x {frame.shape[0]}")

    # 检测
    print("正在检测...")
    results = model(frame)

    # 过滤 person
    result = results[0]
    person_count = 0
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            if class_name == "person":
                person_count += 1
            print(f"  检测到: {class_name} (置信度: {float(box.conf[0]):.2%})")

    print(f"\n共检测到 {person_count} 个人")

    # 显示结果
    annotated = result.plot()
    cv2.putText(
        annotated,
        f"检测到 {person_count} 人",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    cv2.imshow("YOLO 检测结果", annotated)
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    output_path = "result.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
