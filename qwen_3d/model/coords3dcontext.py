import threading
import torch

class Coords3DContext:
    """
    修改版：使用类级别存储，解决 PEFT 深拷贝导致的对象引用断裂问题。
    """
    # 使用类变量作为共享存储
    _shared_state = {
        "coords_3d": None,
        "mask": None,
        "text_embeddings": None,
        "question_mask": None,
        "latest_gate": None,  # 【新增】专门用于存放 gate 的激活值
        "aux_losses": []  # 【新增】用于收集辅助 Loss
    }
    # 简单的线程锁，防止极其罕见的多线程竞争（在标准DDP训练中其实不需要，但为了安全）
    _lock = threading.Lock()

    # 【新增】添加 loss
    def add_aux_loss(self, loss_tensor):
        with self._lock:
            # 必须是一个 scalar tensor (标量)，且带有 grad_fn
            Coords3DContext._shared_state["aux_losses"].append(loss_tensor)
            
    def pop_aux_losses(self):
        with self._lock:
            losses = Coords3DContext._shared_state["aux_losses"]
            # 取出后立即清空列表，防止累积到下一个 batch
            Coords3DContext._shared_state["aux_losses"] = [] 
            
            if len(losses) == 0:
                return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # 将列表中的 loss 求和
            return sum(losses)

    def update(self, coords_3d, mask, text_embeddings, question_mask):
        with self._lock:
            Coords3DContext._shared_state["coords_3d"] = coords_3d
            Coords3DContext._shared_state["mask"] = mask
            Coords3DContext._shared_state["text_embeddings"] = text_embeddings
            Coords3DContext._shared_state["question_mask"] = question_mask

    # 【新增】专门用于更新 gate 的方法
    def update_gate(self, gate_tensor):
        with self._lock:
            # 建议在这里直接 detach 并转 CPU，节省显存，防止梯度滞留
            if isinstance(gate_tensor, torch.Tensor):
                Coords3DContext._shared_state["latest_gate"] = gate_tensor.detach().cpu()
            else:
                Coords3DContext._shared_state["latest_gate"] = gate_tensor

    def clear(self):
        with self._lock:
            Coords3DContext._shared_state["coords_3d"] = None
            Coords3DContext._shared_state["mask"] = None
            Coords3DContext._shared_state["text_embeddings"] = None
            Coords3DContext._shared_state["question_mask"] = None
            # Coords3DContext._shared_state["latest_gate"] = None # 【新增】清理

    @property
    def coords_3d(self):
        return Coords3DContext._shared_state["coords_3d"]

    @property
    def mask(self):
        return Coords3DContext._shared_state["mask"]
    
    @property
    def text_embeddings(self):
        return Coords3DContext._shared_state["text_embeddings"]

    @property
    def question_mask(self):
        return Coords3DContext._shared_state["question_mask"]

    # 【新增】读取 gate
    @property
    def latest_gate(self):
        return Coords3DContext._shared_state["latest_gate"]



# 1. 定义一个简单的状态容器 (不是 nn.Module，因此不会被 PyTorch 递归遍历)
class Coords3DContext_0:
    def __init__(self):
        self.coords_3d = None
        self.mask = None
        self.grid_thw = None  # 新增：用于存储图像的空间尺寸 (t, h, w)
    
    def update(self, coords, mask, grid_thw):
        self.coords_3d = coords
        self.mask = mask
        self.grid_thw = grid_thw
    
    def clear(self):
        self.coords_3d = None
        self.mask = None
        self.grid_thw = None