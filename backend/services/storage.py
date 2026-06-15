"""文件存储服务。"""

import shutil
from pathlib import Path
import aiofiles
from fastapi import UploadFile

from config import settings


class StorageService:
    """文件存储服务。"""

    def __init__(self):
        self.upload_dir = settings.upload_dir
        self.model_dir = settings.model_dir
        self.report_dir = settings.report_dir

    def _get_upload_path(self, dataset_id: str, filename: str) -> Path:
        """获取上传文件路径。"""
        dir_path = self.upload_dir / dataset_id
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / filename

    def _get_model_dir(self, run_id: str) -> Path:
        """获取模型目录。"""
        dir_path = self.model_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _get_report_dir(self, run_id: str) -> Path:
        """获取报告目录。"""
        dir_path = self.report_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    async def save_upload(self, dataset_id: str, file: UploadFile) -> Path:
        """保存上传文件。"""
        if not file.filename:
            raise ValueError("文件名不能为空")

        file_path = self._get_upload_path(dataset_id, file.filename)

        async with aiofiles.open(file_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)

        return file_path

    def delete_dataset_files(self, dataset_id: str) -> None:
        """删除数据集相关文件。"""
        dir_path = self.upload_dir / dataset_id
        if dir_path.exists():
            shutil.rmtree(dir_path)

    def delete_run_files(self, run_id: str) -> None:
        """删除训练任务相关文件。"""
        model_path = self.model_dir / run_id
        report_path = self.report_dir / run_id
        if model_path.exists():
            shutil.rmtree(model_path)
        if report_path.exists():
            shutil.rmtree(report_path)

    def get_model_path(self, run_id: str) -> Path:
        """获取模型路径。"""
        return self._get_model_dir(run_id)

    def get_report_path(self, run_id: str) -> Path:
        """获取报告路径。"""
        return self._get_report_dir(run_id)


storage_service = StorageService()
