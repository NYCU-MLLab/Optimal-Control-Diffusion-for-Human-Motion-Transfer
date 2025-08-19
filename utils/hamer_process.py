import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["EGL_DEVICE_ID"] = "1"
import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mmcv")
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image

from hamer_model.hamer.configs import CACHE_DIR_HAMER
from hamer_model.hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer_model.hamer.utils import recursive_to
from hamer_model.hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer_model.hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

# 初始化預設的 detector 與 keypoint 檢測器
from hamer_model.hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
from hamer_model import hamer
from hamer_model.vitpose_model import ViTPoseModel

from PIL import Image
from torchvision import transforms

from tqdm import tqdm


class HAMER:
    def __init__(self, 
                 checkpoint: str = DEFAULT_CHECKPOINT, 
                 rescale_factor: float = 2.0, 
                 device: torch.device = None):
        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        # 下載並載入 HaMeR 模型與設定檔
        download_models(CACHE_DIR_HAMER)
        self.model, self.model_cfg = load_hamer(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        self.rescale_factor = rescale_factor
        self.image_size = self.model_cfg.MODEL.IMAGE_SIZE
        
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        checkpoint_path = Path(hamer.__file__).parent / 'configs' / 'model_final_f05665.pkl'
        detectron2_cfg.train.init_checkpoint = str(checkpoint_path)
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        
        # 初始化 keypoint 檢測器 (ViTPoseModel)
        self.cpm = ViTPoseModel(self.device)

        # Setup the renderer
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向傳播：  
          輸入：pixel_values, 形狀 [B, sample_num, 3, H, W]  
          輸出：預測 hand mesh 頂點, 形狀 [B, sample_num, hands, 778, 3]  
          （若不同影像檢測到的手數不同，將以零 padding 補齊至 batch 中最大手數）
        """
        B, sample_num, C, H, W = pixel_values.shape
        # print("pixel: ", pixel_values.shape)
        # 用於儲存所有預測結果，結構為 list of list, 外層長度 B, 內層長度 sample_num，
        # 每個元素為 tensor，形狀 [num_hands, 778, 3]
        all_preds = []
        
        for b in range(B):
            preds_per_sample = []
            for s in range(sample_num):
                # 取出單張影像，形狀 [3, H, W]
                img_tensor = pixel_values[b, s]
                if img_tensor.dtype != torch.uint8:
                    # 將 [0, 1] 的數值乘以 255，再轉成 uint8
                    img_tensor = (((img_tensor + 1) / 2) * 255).clamp(0, 255).to(torch.uint8)
                # 轉換為 numpy，形狀 [H, W, 3] (RGB)
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                # 轉為 BGR 給 detector 使用
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                # 執行 detector 偵測
                det_out = self.detector(img_bgr)
                instances = det_out['instances']
                valid_idx = (instances.pred_classes == 0) & (instances.scores > 0.5)

                pred_bboxes=instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                pred_scores=instances.scores[valid_idx].cpu().numpy()
                
                # 組合 bbox 與分數，送入 keypoint 檢測器 (需 RGB 影像)
                boxes_input = np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)
                rgb_img = img_np.copy()
                vitposes_out = self.cpm.predict_pose(rgb_img, [boxes_input])
                
                # 根據 keypoint 檢測結果，嘗試分別提取左手與右手 bbox
                bboxes = []
                is_right = []

                for vitpose in vitposes_out:
                    left_keyp = vitpose['keypoints'][-42:-21]
                    right_keyp = vitpose['keypoints'][-21:]

                    keyp = left_keyp
                    valid = keyp[:,2] > 0.5
                    if sum(valid) > 3:
                        bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                        bboxes.append(bbox)
                        is_right.append(0)
                    keyp = right_keyp
                    valid = keyp[:,2] > 0.5
                    if sum(valid) > 3:
                        bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                        bboxes.append(bbox)
                        is_right.append(1)
                
                # 若沒有檢測到任何手部 bbox，則 fallback 為全圖 bbox
                if len(bboxes) == 0:
                    zeros_tensor = torch.zeros(3, H, W, dtype=torch.float32, device="cpu")
                    preds_per_sample.append(zeros_tensor)
                    continue

                boxes = np.stack(bboxes)
                right = np.stack(is_right)

                # 利用 ViTDetDataset 將原圖與 bbox 轉換成 HaMeR 模型所需的輸入格式
                dataset = ViTDetDataset(self.model_cfg, img_bgr, boxes, right, rescale_factor=self.rescale_factor)
                dataloader = DataLoader(dataset, batch_size=len(bboxes), shuffle=False, num_workers=0)
                
                all_verts = []
                all_cam_t = []
                all_right = []

                for batch in dataloader:
                    batch = recursive_to(batch, self.device)
                    with torch.no_grad():
                        out = self.model(batch)

                    multiplier = (2*batch['right']-1)
                    pred_cam = out['pred_cam']
                    pred_cam[:,1] = multiplier*pred_cam[:,1]
                    box_center = batch["box_center"].float()
                    box_size = batch["box_size"].float()
                    img_size = batch["img_size"].float()
                    multiplier = (2*batch['right']-1)
                    scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
                    break  # dataset 只有一個 batch

                # Render the result
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    # Get filename from path img_path
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                    input_patch = input_patch.permute(1,2,0).numpy()
                    
                    # Add all verts and cams to list
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    is_right = batch['right'][n].cpu().numpy()
                    verts[:,0] = (2*is_right-1)*verts[:,0]
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

                if len(all_verts) > 0:
                    misc_args = dict(
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                        focal_length=scaled_focal_length,
                    )
                    cam_view = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)
                    cam_view_out = cam_view[:,:,:3] * cam_view[:,:,3:]
                    # cv2.imwrite(os.path.join("ex_out", f'cam_view_{s}.jpg'), 255*cam_view_out[:, :, ::-1])
                    preds_per_sample.append(torch.from_numpy(cam_view_out).permute(2, 0, 1))  #[3,H,W]
            all_preds.append(preds_per_sample)  #[sample_num, 3, H,W]
            # print("all_preds: ", len(all_preds), len(all_preds[0]), all_preds[0][0].shape)
        output = torch.from_numpy(np.array(all_preds))
        # print("final_out: ", output.shape)
        return output
    
    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward(pixel_values)


def load_images_from_folder(folder_path, img_size=(512, 512)):
    # 定義轉換操作，將圖片轉換為 RGB 並 resize 成固定尺寸，再轉換為 tensor
    transform = transforms.Compose([      # 調整圖片大小 (H, W)
        transforms.Resize(img_size),
        transforms.ToTensor(),            # 轉換成 [C, H, W] 且將像素值標準化到 [0, 1]
    ])
    
    images = []
    # 遍歷資料夾中所有圖片檔案
    for filename in sorted(os.listdir(folder_path)):
        # 篩選常見的圖片格式
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(folder_path, filename)
            # 讀取圖片並轉換為 RGB
            image = Image.open(img_path).convert('RGB')
            # 轉換圖片
            # img_pil = image.resize(img_size)
            img_tensor = torch.from_numpy(np.array(image)).float()
            img_normalized = img_tensor/127.5 - 1
            img_nor_tensor = img_normalized.permute(2, 0, 1)
            # print("image_tensor: ", image_tensor.shape)
            images.append(img_nor_tensor)
    
    # 如果資料夾中有圖片，使用 torch.stack 將它們堆疊起來
    if images:
        images_tensor = torch.stack(images)
        return images_tensor
    else:
        raise ValueError("指定資料夾中沒有發現圖片。")

def main():
    parser = argparse.ArgumentParser(description="使用 HAMER 模型處理多資料夾圖片")
    parser.add_argument("--root_path", type=str, required=True, help="根目錄路徑")
    args = parser.parse_args()

    root_path = args.root_path
    # 找出 root_path 底下所有子資料夾
    subdirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    # 初始化 HAMER 模型 (只載入一次)
    model = HAMER()
    
    # 針對每個子資料夾處理
    for subdir in tqdm(subdirs):
        input_dir = os.path.join(root_path, subdir, "images")
        output_dir = os.path.join(root_path, subdir, "hands")
        if not os.path.exists(input_dir):
            print(f"資料夾 {input_dir} 不存在，跳過。")
            continue
        os.makedirs(output_dir, exist_ok=True)
        
        # 取得 input_dir 中所有圖片檔案（依副檔名篩選），並依檔名排序
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        file_list = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)])
        if not file_list:
            print(f"在 {input_dir} 沒有發現圖片，跳過。")
            continue
        
        # 載入圖片 (並調整至固定尺寸)
        images_tensor = load_images_from_folder(input_dir)
        # 增加 sample_num 維度 (這裡設定為 1)
        images_tensor = images_tensor.unsqueeze(1)  # [B, 1, 3, H, W]
        
        # 模型推論
        with torch.no_grad():
            outputs = model(images_tensor)   # 預期 shape: [B, 1, 3, H, W]
        outputs = outputs.squeeze(1).cpu().numpy()   # [B, 3, H, W]
        
        # 儲存結果 (使用與原始檔名對應)
        for i, filename in enumerate(file_list):
            out_img = outputs[i]                     # 取出第 i 張圖片
            out_img = np.transpose(out_img, (1, 2, 0))
            out_img = (out_img * 255).clip(0, 255).astype(np.uint8)
            out_filepath = os.path.join(output_dir, filename)
            cv2.imwrite(out_filepath, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        print(f"處理完成 {subdir}，輸出儲存在: {output_dir}")

if __name__ == '__main__':
    main()
