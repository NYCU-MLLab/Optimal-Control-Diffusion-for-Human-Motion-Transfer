import os
import sys
import cv2
import numpy as np

def process_image(in_path, out_path, area_thresh=1000, thresh_val=10):
    """
    讀 in_path，找出手部輪廓後，只畫出其外框的填滿白色長方形，
    其餘皆為黑。存成同尺寸黑底白框的輸出圖到 out_path。
    """
    img = cv2.imread(in_path)
    if img is None:
        # print(f"[WARN] 無法讀取：{in_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 建立純黑的輸出畫布
    mask = np.zeros_like(gray)

    for cnt in contours:
        if cv2.contourArea(cnt) < area_thresh:
            continue
        # 取得外框
        x, y, w, h = cv2.boundingRect(cnt)
        # 畫滿白色長方形
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)

    cv2.imwrite(out_path, mask)


def main(root_dir):
    if not os.path.isdir(root_dir):
        # print(f"請提供存在的資料夾路徑，您給的是：{root_dir}")
        sys.exit(1)

    for sub in sorted(os.listdir(root_dir)):
        sub_dir = os.path.join(root_dir, sub)
        hands_dir = os.path.join(sub_dir, 'hands')
        mask_dir  = os.path.join(sub_dir, 'hand_mask')

        if not os.path.isdir(hands_dir):
            continue

        os.makedirs(mask_dir, exist_ok=True)

        files = sorted(f for f in os.listdir(hands_dir)
                       if f.lower().endswith(('.png','.jpg','.jpeg','bmp','tif','tiff')))
        for fname in files:
            in_path  = os.path.join(hands_dir, fname)
            out_path = os.path.join(mask_dir, fname)
            process_image(in_path, out_path)
            # print(f"[OK] {sub}/hands/{fname} → {sub}/hand_mask/{fname}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    root = sys.argv[1]
    main(root)
