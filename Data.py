import cv2
import numpy as np
import pandas as pd
import os
import json

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


from tqdm import tqdm

EMOTION_LABEL = ['Anxiety', 'Peace', 'Weariness', 'Happiness', 'Anger']
DRIVER_BEHAVIOR_LABEL = ['Smoking', 'Making Phone', 'Looking Around', 'Dozing Off', 'Normal Driving', 'Talking', 'Body Movement']
SCENE_CENTRIC_CONTEXT_LABEL = ['Traffic Jam', 'Waiting', 'Smooth Traffic']
VEHICLE_BASED_CONTEXT_LABEL = ['Parking', 'Turning', 'Backward Moving', 'Changing Lane', 'Forward Moving']

class CarDataset(Dataset):

    def __init__(self, csv_file: str, dataset_root: str, device: torch.device, horizontal_flip_prob: float = 0.5, vertical_flip_prob: float = 0.5):
        """
        Args:
            csv_file (str): Path to the CSV file containing the dataset information.
            dataset_root (str): Root directory of the dataset.
            device (torch.device): Device to which the data will be moved (e.g., 'cpu' or 'cuda').
            horizontal_flip_prob (float): Probability of applying horizontal flip augmentation.
            vertical_flip_prob (float): Probability of applying vertical flip augmentation.
        """
        self.path = pd.read_csv(csv_file).astype(str).rename(columns={0: 'frames', 1: 'label'}).copy()
        self.dataset_root = dataset_root
        self.resize_height = 224
        self.resize_width = 224
        self.body_height = 112
        self.body_width = 112
        self.face_height = 64
        self.face_width = 64

        self.device = device

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((self.resize_height, self.resize_width)),
            v2.RandomVerticalFlip(p=vertical_flip_prob),
            v2.RandomHorizontalFlip(p=horizontal_flip_prob),
            #v2.Normalize(mean=[0.3529, 0.3843, 0.4], std=[0.3529, 0.3843, 0.4]),
        ]).to(self.device, non_blocking=True)

        self.body_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((self.resize_height, self.resize_width)),
            #v2.Normalize(mean=[0.3529, 0.3843, 0.4], std=[0.3529, 0.3843, 0.4]),
        ]).to(self.device, non_blocking=True)

        self.face_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((self.face_height, self.face_width)),
            #v2.Normalize(mean=[0.3529, 0.3843, 0.4], std=[0.3529, 0.3843, 0.4]),
        ]).to(self.device, non_blocking=True)

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frames_path, label_path = self.path.iloc[idx]
        frames_path = self.dataset_root + "/" + frames_path
        label_path = self.dataset_root + "/" + label_path

        label_json = json.load(open(label_path))
        pose_list = label_json['pose_list']

        buffer, buffer_front, buffer_left, buffer_right, buffer_face, buffer_body, posture, gesture = self.load_frames(frames_path, pose_list)

        buffer = buffer.to(self.device, dtype=torch.float32, non_blocking=True)
        buffer_front = buffer_front.to(self.device, dtype=torch.float32, non_blocking=True)
        buffer_left = buffer_left.to(self.device, dtype=torch.float32, non_blocking=True)
        buffer_right = buffer_right.to(self.device, dtype=torch.float32, non_blocking=True)

        buffer_body = buffer_body.to(self.device, dtype=torch.float32, non_blocking=True)
        buffer_face = buffer_face.to(self.device, dtype=torch.float32, non_blocking=True)

        emotion_label = EMOTION_LABEL.index((label_json['emotion_label'].capitalize()))
        behavior_label = DRIVER_BEHAVIOR_LABEL.index((label_json['driver_behavior_label']))
        context_label = SCENE_CENTRIC_CONTEXT_LABEL.index((label_json['scene_centric_context_label']))
        vehicle_label = VEHICLE_BASED_CONTEXT_LABEL.index((label_json['vehicle_based_context_label']))

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

            keypoints = np.array(pose_list[i]['result'][0]['keypoints']).reshape(-1, 3)          
            posture =  keypoints[:26]
            gesture = keypoints[94:136]
            posture_list.append(posture)
            gesture_list.append(gesture)

            if img_face is None:
                img_face = img_body.copy()

            img = self.transform(img)
            front_img = self.transform(front_img)
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
            img_body = self.body_transform(img_body)
            img_face = self.face_transform(img_face)

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
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"  # For testing on CPU, change to "cuda" for GPU
    dataset = CarDataset(csv_file=r"F:/Stage Project/AIDE_dataset/testing.csv", dataset_root=r"F:/Stage Project/AIDE_dataset", device=device)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, pin_memory=True, drop_last=False)

    for epoch, (img1,img2,img3,img4,face,body,posture, gesture,emotion_label, behavior_label, context_label, vehicle_label) in enumerate(tqdm(test_dataloader)):
        print(f"Subepoch: {epoch}, img1 shape: {img1.shape}, img2 shape: {img2.shape}, img3 shape: {img3.shape}, img4 shape: {img4.shape}, face shape: {face.shape}, body shape: {body.shape}, posture shape: {posture.shape}, gesture shape: {gesture.shape}, emotion_label: {emotion_label}, behavior_label: {behavior_label}, context_label: {context_label}, vehicle_label: {vehicle_label}")
        cv2.imwrite(f"F:/Stage Project/code/BaselineReproduction/dataset_test/img1_{epoch}.jpg", img1[0, 0, :, :, :].permute(1, 2, 0).numpy() * 255)
        cv2.imwrite(f"F:/Stage Project/code/BaselineReproduction/dataset_test/img2_{epoch}.jpg", img2[0, 0, :, :, :].permute(1, 2, 0).numpy() * 255)
        cv2.imwrite(f"F:/Stage Project/code/BaselineReproduction/dataset_test/img3_{epoch}.jpg", img3[0, 0, :, :, :].permute(1, 2, 0).numpy() * 255)
        cv2.imwrite(f"F:/Stage Project/code/BaselineReproduction/dataset_test/img4_{epoch}.jpg", img4[0, 0, :, :, :].permute(1, 2, 0).numpy() * 255)
        cv2.imwrite(f"F:/Stage Project/code/BaselineReproduction/dataset_test/face_{epoch}.jpg", face[0, 0, :, :, :].permute(1, 2, 0).numpy() * 255)
        cv2.imwrite(f"F:/Stage Project/code/BaselineReproduction/dataset_test/body_{epoch}.jpg", body[0, 0, :, :, :].permute(1, 2, 0).numpy() * 255)
        break
        #pass
