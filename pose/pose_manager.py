import json
import os

class PoseManager:
    def __init__(self, filepath="poses.json"):
        self.filepath = filepath
        self.target_poses = {}
        self.load_poses()

    def load_poses(self):
        """JSON 파일에서 포즈 데이터를 읽어옵니다."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"{self.filepath} 파일이 필요합니다.")
            
        with open(self.filepath, 'r', encoding='utf-8') as f:
            self.target_poses = json.load(f)
            
    def get_pose_names(self):
        """선택 가능한 포즈 이름 리스트를 반환합니다."""
        return list(self.target_poses.keys())

    def get_pose_angles(self, pose_name):
        """특정 포즈의 관절 각도 데이터를 반환합니다."""
        return self.target_poses.get(pose_name)