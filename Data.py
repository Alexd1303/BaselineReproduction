from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import os
import json

import torch
from torch.utils.data import DataLoader, Dataset


from tqdm import tqdm
import albumentations as A

os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"

EMOTION_LABEL = ['Anxiety', 'Peace', 'Weariness', 'Happiness', 'Anger']
DRIVER_BEHAVIOR_LABEL = ['Smoking', 'Making Phone', 'Looking Around', 'Dozing Off', 'Normal Driving', 'Talking', 'Body Movement']
SCENE_CENTRIC_CONTEXT_LABEL = ['Traffic Jam', 'Waiting', 'Smooth Traffic']
VEHICLE_BASED_CONTEXT_LABEL = ['Parking', 'Turning', 'Backward Moving', 'Changing Lane', 'Forward Moving']

class CarDataset(Dataset):

    def __init__(self, csv_file: str, dataset_root: str, device: torch.device = torch.device("cpu"), horizontal_flip_prob: float = 0.5, vertical_flip_prob: float = 0.5, use_experimental_augmentation: bool = False, synchronize_transforms: bool = False):
        """
        Args:
            csv_file (str): Path to the CSV file containing the dataset information.
            dataset_root (str): Root directory of the dataset.
            device (torch.device): Device to which the data will be moved (e.g., 'cpu' or 'cuda').
            horizontal_flip_prob (float): Probability of applying horizontal flip augmentation.
            vertical_flip_prob (float): Probability of applying vertical flip augmentation.
            use_experimental_augmentation (bool): Whether to use experimental augmentation techniques.
            synchronize_transforms (bool): Whether to synchronize the transformations across different camera views.
        """
        self.path = pd.read_csv(csv_file).astype(str).rename(columns={'0': 'frames', '1': 'label'}).copy()
        assert os.path.exists(dataset_root), f"Dataset root directory {dataset_root} does not exist."
        self.dataset_root = dataset_root
        self.resize_height = 224
        self.resize_width = 224
        self.body_height = 224 #112
        self.body_width = 224 #112
        self.face_height = 224 #64
        self.face_width = 224 #64
        self.synchronize_transforms = synchronize_transforms
        self.use_experimental_augmentation = use_experimental_augmentation
        self.device = device
        
        self.resize_imgs = A.Compose([
            A.Resize(height=self.resize_height, width=self.resize_width),
            A.ToFloat(),
            A.pytorch.ToTensorV2(),
            
        ])
        self.resize_body = A.Compose([
            A.Resize(height=self.body_height, width=self.body_width),
            A.ToFloat(),
            A.pytorch.ToTensorV2(),
            
        ])
        self.resize_face = A.Compose([
            A.Resize(height=self.face_height, width=self.face_width),
            A.ToFloat(),
            A.pytorch.ToTensorV2(),
        ])

        self.transform = A.Compose([
            A.HorizontalFlip(p=horizontal_flip_prob),
            A.VerticalFlip(p=vertical_flip_prob),
        ])
        
        if use_experimental_augmentation:
            self.transform = A.Compose([
                self.transform,
                A.AdditiveNoise(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.2),
                A.Rotate(angle_range=(-30, 30), crop_border=True, p=0.2),
                A.Resize(height=self.resize_height, width=self.resize_width),
            ])
                

        self.body_transform = A.Compose([
        ])

        self.face_transform = A.Compose([
        ])

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frames_path, label_path = str(self.path.iloc[idx]['frames']), str(self.path.iloc[idx]['label'])
        frames_path = self.dataset_root / frames_path
        label_path = self.dataset_root / label_path

        label_json = json.load(open(label_path))
        pose_list = label_json['pose_list']

        buffer, buffer_front, buffer_left, buffer_right, buffer_face, buffer_body, posture, gesture = self.load_frames(frames_path, pose_list)

        if self.synchronize_transforms:
            synch_buffer = self.transform(images=torch.stack([buffer, buffer_front, buffer_left, buffer_right], dim=0).view(64, 3, 224, 224))['images']
            buffer = synch_buffer.view(64, 3, 224, 224)[0:16, :, :, :]
            buffer_front = synch_buffer.view(64, 3, 224, 224)[16:32, :, :, :]
            buffer_left = synch_buffer.view(64, 3, 224, 224)[32:48, :, :, :]
            buffer_right = synch_buffer.view(64, 3, 224, 224)[48:64, :, :, :]
        else:
            buffer = self.transform(images=buffer)['images']
            buffer_front = self.transform(images=buffer_front)['images']
            buffer_left =  self.transform(images=buffer_left)['images']
            buffer_right = self.transform(images=buffer_right)['images']

        buffer_body = self.body_transform(images=buffer_body)['images']
        buffer_face = self.face_transform(images=buffer_face)['images']

        emotion_label = EMOTION_LABEL.index((label_json['emotion_label'].capitalize()))
        behavior_label = DRIVER_BEHAVIOR_LABEL.index((label_json['driver_behavior_label']))
        context_label = SCENE_CENTRIC_CONTEXT_LABEL.index((label_json['scene_centric_context_label']))
        if label_json['vehicle_based_context_label'] == "Forward":
            label_json['vehicle_based_context_label'] = "Forward Moving"
        vehicle_label = VEHICLE_BASED_CONTEXT_LABEL.index((label_json['vehicle_based_context_label']))
      

        return buffer, buffer_front, buffer_left, buffer_right, buffer_face, buffer_body, posture, gesture, emotion_label, behavior_label, context_label, vehicle_label


    def load_frames(self, file_dir, pose_list):

        incar_path = os.path.join(file_dir, 'incarframes')
        front_frames = os.path.join(file_dir, 'frontframes')
        left_frames = os.path.join(file_dir, 'leftframes')
        right_frames = os.path.join(file_dir, 'rightframes')
        face_frames = os.path.join(file_dir, 'face')
        body_frames = os.path.join(file_dir, 'body')


        frames = [os.path.join(incar_path, img) for img in sorted(os.listdir(incar_path), key=lambda x: int(os.path.basename(x).split('.')[0])) if img.endswith('.jpg')]
        front_frames = [os.path.join(front_frames, img) for img in sorted(os.listdir(front_frames), key=lambda x: int(os.path.basename(x).split('.')[0])) if img.endswith('.jpg')]
        left_frames = [os.path.join(left_frames, img) for img in sorted(os.listdir(left_frames), key=lambda x: int(os.path.basename(x).split('.')[0])) if img.endswith('.jpg')]
        right_frames = [os.path.join(right_frames, img) for img in sorted(os.listdir(right_frames), key=lambda x: int(os.path.basename(x).split('.')[0])) if img.endswith('.jpg')]

        face_frames = [os.path.join(face_frames, img) for img in sorted(os.listdir(face_frames), key=lambda x: int(os.path.basename(x).split('_')[0])) if img.endswith('.jpg')]
        if len(face_frames)!=45:
            face_frames.extend([face_frames[-1]] * (45 - len(face_frames)))
        body_frames = [os.path.join(body_frames, img) for img in sorted(os.listdir(body_frames), key=lambda x: int(os.path.basename(x).split('_')[0])) if img.endswith('.jpg')]
        if len(body_frames)!=45:
            body_frames.extend([body_frames[-1]] * (45 - len(body_frames)))


        buffer, buffer_front, buffer_left, buffer_right, keypoints_list, buffer_face, buffer_body= [], [], [], [], [], [], []
        posture_list, gesture_list = [], [] 

        for i, frame_name in enumerate(frames):
            if not i == 0 and not i % 3 == 2:
                continue
            if i >= 45:
                break

            img = cv2.imread(frame_name)
            front_img = cv2.imread(front_frames[i])
            left_img = cv2.imread(left_frames[i])
            right_img = cv2.imread(right_frames[i])
            img_face = cv2.imread(face_frames[i])
            img_body = cv2.imread(body_frames[i])
            
            assert img is not None, f"Image is None for frame {i} in {frame_name}"
            assert front_img is not None, f"Front image is None for frame {i} in {front_frames[i]}"
            assert left_img is not None, f"Left image is None for frame {i} in {left_frames[i]}"
            assert right_img is not None, f"Right image is None for frame {i} in {right_frames[i]}"
            assert img_face is not None, f"Face image is None for frame {i} in {face_frames[i]}"
            assert img_body is not None, f"Body image is None for frame {i} in {body_frames[i]}"

            keypoints = np.array(pose_list[i]['result'][0]['keypoints']).reshape(-1, 3)          
            posture =  keypoints[:26]
            gesture = keypoints[94:136]
            posture_list.append(posture)
            gesture_list.append(gesture)

            if img_face is None:
                img_face = img_body.copy()

            img = self.resize_imgs(image=img)['image']
            front_img = self.resize_imgs(image=front_img)['image']
            left_img = self.resize_imgs(image=left_img)['image']
            right_img = self.resize_imgs(image=right_img)['image']
            img_body = self.resize_body(image=img_body)['image']
            img_face = self.resize_face(image=img_face)['image']

            buffer.append(img)
            buffer_front.append(front_img)
            buffer_left.append(left_img)
            buffer_right.append(right_img)
            buffer_body.append(img_body)
            buffer_face.append(img_face)

            posture_array = np.array(posture_list, dtype=np.float32)
            gesture_array = np.array(gesture_list, dtype=np.float32)
            posture_tensor = torch.from_numpy(posture_array)
            gesture_tensor = torch.from_numpy(gesture_array)

        return torch.stack(buffer), torch.stack(buffer_front), torch.stack(buffer_left), torch.stack(buffer_right), torch.stack(buffer_face), torch.stack(buffer_body), posture_tensor, gesture_tensor


if __name__ == "__main__":
    device = torch.device("cpu")
    dataset = CarDataset(csv_file=Path(r"F:/Stage Project/AIDE_dataset/training.csv"), dataset_root=Path(r"F:/Stage Project/AIDE_dataset"), device=device, horizontal_flip_prob=0.5, vertical_flip_prob=0.5, use_experimental_augmentation=True, synchronize_transforms=True)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, pin_memory=True, drop_last=False)

    for epoch, (inside, front, left, right, face, body, posture, gesture, emotion_label, behavior_label, context_label, vehicle_label) in enumerate(tqdm(test_dataloader)):
        #print(f"Subepoch: {epoch}, img1 shape: {img1.shape}, img2 shape: {img2.shape}, img3 shape: {img3.shape}, img4 shape: {img4.shape}, face shape: {face.shape}, body shape: {body.shape}, posture shape: {posture.shape}, gesture shape: {gesture.shape}, emotion_label: {emotion_label}, behavior_label: {behavior_label}, context_label: {context_label}, vehicle_label: {vehicle_label}")
        for frame_n in range(inside.shape[1]):
            cv2.imwrite(f"F:/Stage Project/code/BaselineReproduction/dataset_test/inside_{epoch}_{frame_n}.jpg", inside[0, frame_n, :, :, :].permute(1, 2, 0).numpy() * 255)
            cv2.imwrite(f"F:/Stage Project/code/BaselineReproduction/dataset_test/front_{epoch}_{frame_n}.jpg", front[0, frame_n, :, :, :].permute(1, 2, 0).numpy() * 255)
            cv2.imwrite(f"F:/Stage Project/code/BaselineReproduction/dataset_test/left_{epoch}_{frame_n}.jpg", left[0, frame_n, :, :, :].permute(1, 2, 0).numpy() * 255)
            cv2.imwrite(f"F:/Stage Project/code/BaselineReproduction/dataset_test/right_{epoch}_{frame_n}.jpg", right[0, frame_n, :, :, :].permute(1, 2, 0).numpy() * 255)
            cv2.imwrite(f"F:/Stage Project/code/BaselineReproduction/dataset_test/face_{epoch}_{frame_n}.jpg", face[0, frame_n, :, :, :].permute(1, 2, 0).numpy() * 255)
            cv2.imwrite(f"F:/Stage Project/code/BaselineReproduction/dataset_test/body_{epoch}_{frame_n}.jpg", body[0, frame_n, :, :, :].permute(1, 2, 0).numpy() * 255)
            
        if epoch == 10: break
        pass
