"""
任务队列管理模块
支持批量任务的排队和执行
"""
import json
import os
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "待执行"
    RUNNING = "运行中"
    COMPLETED = "已完成"
    FAILED = "失败"
    CANCELLED = "已取消"


class Task:
    """单个任务"""
    def __init__(self, task_id: str, task_type: str, params: Dict[str, Any]):
        self.task_id = task_id
        self.task_type = task_type  # training, sampling, evaluation, plotting
        self.params = params
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.error_message: Optional[str] = None
        self.result: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'params': self.params,
            'status': self.status.value,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'error_message': self.error_message,
            'result': self.result
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典创建"""
        task = cls(data['task_id'], data['task_type'], data['params'])
        task.status = TaskStatus(data['status'])
        task.created_at = data['created_at']
        task.started_at = data.get('started_at')
        task.completed_at = data.get('completed_at')
        task.error_message = data.get('error_message')
        task.result = data.get('result')
        return task


class TaskQueue:
    """任务队列"""
    def __init__(self, save_path: str = "task_queue.json"):
        self.tasks: List[Task] = []
        self.save_path = save_path
        self.load()
    
    def add_task(self, task_type: str, params: Dict[str, Any]) -> str:
        """添加任务"""
        task_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.tasks)}"
        task = Task(task_id, task_type, params)
        self.tasks.append(task)
        self.save()
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_pending_tasks(self) -> List[Task]:
        """获取待执行任务"""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]
    
    def get_running_tasks(self) -> List[Task]:
        """获取运行中任务"""
        return [t for t in self.tasks if t.status == TaskStatus.RUNNING]
    
    def get_completed_tasks(self) -> List[Task]:
        """获取已完成任务"""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          error_message: Optional[str] = None,
                          result: Optional[Any] = None):
        """更新任务状态"""
        task = self.get_task(task_id)
        if task:
            task.status = status
            if status == TaskStatus.RUNNING:
                task.started_at = datetime.now().isoformat()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                task.completed_at = datetime.now().isoformat()
            if error_message:
                task.error_message = error_message
            if result:
                task.result = result
            self.save()
    
    def remove_task(self, task_id: str):
        """移除任务"""
        self.tasks = [t for t in self.tasks if t.task_id != task_id]
        self.save()
    
    def clear_completed(self):
        """清除已完成任务"""
        self.tasks = [t for t in self.tasks if t.status not in 
                     [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]]
        self.save()
    
    def save(self):
        """保存到文件"""
        data = [task.to_dict() for task in self.tasks]
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self):
        """从文件加载"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.tasks = [Task.from_dict(t) for t in data]
            except Exception as e:
                print(f"加载任务队列失败: {e}")
                self.tasks = []
    
    def get_statistics(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            'total': len(self.tasks),
            'pending': len(self.get_pending_tasks()),
            'running': len(self.get_running_tasks()),
            'completed': len(self.get_completed_tasks()),
            'failed': len([t for t in self.tasks if t.status == TaskStatus.FAILED]),
            'cancelled': len([t for t in self.tasks if t.status == TaskStatus.CANCELLED])
        }
