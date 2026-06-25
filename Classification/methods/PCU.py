import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


class PCU_Engine:
    """
    PCU Engine: Incremental PCA subspace analysis and failure mode detection

    Functions:
    1. incremental PCA
    2. Cal ER (Energy Ratio) and CO (Coefficient)
    3. ACC
    4. ER for forward failure
    5. CO for backward failure
    """

    def __init__(self, r=1, max_ws=10, min_ws=3, significance_level=0.05, layer_name=None):
        self.top_k = r
        self._params_before_task = None
        self._param_shapes = None

        self.window_acc = []
        self.max_window_acc = []
        self.min_window_acc = []
        self.window_ER = []
        self.max_window_ER = []
        self.min_window_ER = []
        self.window_CO = []

        self.max_ws = max_ws
        self.min_ws = min_ws
        self.significance_level = significance_level

        self.n_samples = 0
        self._init_buffer = []

        self.components = None
        self.singular_values = None

        self.mean = None
        self.threshold = 1e-4
        self.u_t = None
        self.layer_name = layer_name  # the layer for analysis


    def flatten_model_params(self, model):
        params = []
        shapes = []
        for p in model.parameters():
            params.append(p.data.clone().view(-1))
            shapes.append(p.shape)
        self._param_shapes = shapes
        return torch.cat(params)

    # ========== 合并自 IncrementalSubspaceAnalyzer 的方法 ==========

    def update_and_project(self, u_t):
        """
        1. 先投影到当前 basis（更新前）
        2. 用 u_t 增量更新统计信息

        Args:
            u_t: 参数更新向量 (tensor)

        Returns:
            projection_result: 包含 proj_norm, ER, CO, basis_rank 的字典
        """
        u_t_cpu = u_t.detach().cpu().to(torch.float32)

        # Step 1: 投影到当前的 Dominant Basis（更新前）
        projection_result = self._project_vector(u_t_cpu)

        # Step 2: 增量更新统计信息
        self.u_t = u_t_cpu
        self._incremental_update()

        return projection_result

    def project_only(self, u_t):
        """
        只投影，不更新 basis（用于实时监控 gradient 的 ER/CO）

        Args:
            u_t: 参数更新向量 (tensor)

        Returns:
            projection_result: 包含 proj_norm, ER, CO, basis_rank 的字典
        """
        u_t_cpu = u_t.detach().cpu().to(torch.float32)
        return self._project_vector(u_t_cpu)

    def update_only(self, u_t):
        """
        只更新 basis，不返回投影结果（用于 task 结束后更新 subspace）

        Args:
            u_t: 参数更新向量 (tensor)
        """
        u_t_cpu = u_t.detach().cpu().to(torch.float32)
        self.u_t = u_t_cpu
        self._incremental_update()

    def _project_vector(self, u_t_cpu):
        """
        将 u_t 投影到当前的 Dominant Basis（内部方法）

        注意：所有数据都已在 CPU 上，无需再调用 .cpu()
        """
        if self.components is None:
            return {
                "proj_norm": 0.0,
                "ER": 0.0,
                "CO": [],
                "basis_rank": 0,
            }

        # 中心化（所有数据都在 CPU 上）
        if self.mean is not None:
            u_centered = u_t_cpu - self.mean
        else:
            u_centered = u_t_cpu

        # 计算投影系数: coeffs = Q^T @ u_centered
        coeffs = torch.matmul(self.components.t(), u_centered)

        # 计算投影向量: proj = Q @ coeffs
        projection = torch.matmul(self.components, coeffs)

        # 计算范数
        proj_norm = torch.norm(projection).item()
        orig_norm = torch.norm(u_t_cpu).item()
        ratio = proj_norm / orig_norm if orig_norm > self.threshold else 0

        return {
            "proj_norm": proj_norm,
            "ER": ratio,
            "CO": coeffs.tolist(),
            "basis_rank": self.components.shape[1],
        }


    def _flatten_grads_with_zeros(self, model):
        grads = []
        has_grad_mask = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
                has_grad_mask.append(True)
            else:
                grads.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
                has_grad_mask.append(False)
        return torch.cat(grads), has_grad_mask

    def _get_layer_params(self, model):
        """
        获取指定层的参数，展平并转换为 CPU float32

        Args:
            model: 模型

        Returns:
            展平的参数向量 (CPU, float32)
        """
        state_dict = model.state_dict()
        if self.layer_name not in state_dict:
            raise KeyError(f"Layer '{self.layer_name}' not found in model. Available: {list(state_dict.keys())}")
        return state_dict[self.layer_name].detach().cpu().to(torch.float32).view(-1)

    def save_params_before_task(self, model):
        """
        保存 task 开始前的参数（在 CPU 上以节省 GPU 内存）

        Args:
            model: 模型
            layer_name: 如果指定，只保存该层的参数；否则保存整个模型
        """
        if self.layer_name:
            self._params_before_task = self._get_layer_params(model)
        else:
            self._params_before_task = self.flatten_model_params(model).detach().cpu().to(torch.float32)

    def record_task_update(self, model):
        """
        计算 task 更新向量 u_t = θ_after - θ_before

        Args:
            model: 模型
            layer_name: 如果指定，只计算该层的更新；否则计算整个模型
        """
        if self._params_before_task is None:
            return

        # 获取当前参数（已在 CPU 上，float32）
        if self.layer_name:
            params_after = self._get_layer_params(model)
        else:
            params_after = self.flatten_model_params(model).detach().cpu().to(torch.float32)

        # u_t 直接在 CPU 上计算，无需再 .cpu()
        self.u_t = params_after - self._params_before_task

        self._params_before_task = None

    def project(self, model):
        """
        将当前参数更新投影到 Dominant Basis

        Args:
            model: 模型
        """
        # 检查 _params_before_task 是否已设置
        if self._params_before_task is None:
            return {
                "proj_norm": 0.0,
                "ER": 0.0,
                "CO": [0.0],
                "basis_rank": 0,
            }

        # 获取当前参数（已在 CPU 上，float32）
        if self.layer_name:
            params_after = self._get_layer_params(model)
        else:
            params_after = self.flatten_model_params(model).detach().cpu().to(torch.float32)

        # 计算 u_t（已在 CPU 上）
        u_t_temp = params_after - self._params_before_task

        if self.components is None:
            return {
                "proj_norm": 0.0,
                "ER": 0.0,
                "CO": [0.0],
                "basis_rank": 0,
            }

        # 中心化（所有数据都在 CPU 上）
        if self.mean is not None:
            u_centered = u_t_temp - self.mean
        else:
            u_centered = u_t_temp

        # 计算投影系数: coeffs = Q^T @ u_centered
        coeffs = torch.matmul(self.components.t(), u_centered)

        # 计算投影向量: proj = Q @ coeffs
        projection = torch.matmul(self.components, coeffs)

        # 计算范数
        proj_norm = torch.norm(projection).item()
        orig_norm = torch.norm(u_t_temp).item()
        ratio = proj_norm / orig_norm if orig_norm > self.threshold else 0

        return {
            "proj_norm": proj_norm,
            "ER": ratio,
            "CO": coeffs.tolist(),
            "basis_rank": self.components.shape[1],
        }

    def _incremental_update(self):
        """
        增量更新PCA，不保存历史数据（除了初始化阶段）

        使用增量SVD方法：当有新数据x时，更新均值和主成分基

        注意：self.u_t 已经在 CPU 上（由 record_task_update 或 update_and_project 保证）
        """
        if self.u_t is None:
            return

        # self.u_t 已经在 CPU 上，直接使用，无需再 .cpu()
        u_t = self.u_t

        self.n_samples += 1
        n = self.n_samples

        if n == 1:
            # 第一个样本：只记录，等第二个样本来了再初始化
            self._init_buffer.append(u_t.clone())
            self.mean = u_t.clone()
            return

        if n == 2:
            # 第二个样本：用两个样本初始化PCA
            self._init_buffer.append(u_t.clone())

            # 计算均值
            self.mean = (self._init_buffer[0] + self._init_buffer[1]) / 2

            # 中心化
            x1_centered = self._init_buffer[0] - self.mean
            x2_centered = self._init_buffer[1] - self.mean

            # 用两个中心化向量做SVD初始化
            X = torch.stack([x1_centered, x2_centered], dim=1)  # [D, 2]
            U, S, _ = torch.linalg.svd(X, full_matrices=False)

            # 只保留有效的成分
            valid = S > self.threshold
            k = min(self.top_k, valid.sum().item())
            if k > 0:
                self.components = U[:, :k]
                self.singular_values = S[:k]

            # 清空初始化缓存
            self._init_buffer = []
            return

        # n >= 3: 增量更新
        # 更新均值（增量公式）：μ_new = μ_old + (x - μ_old) / n
        old_mean = self.mean
        self.mean = old_mean + (u_t - old_mean) / n

        # 中心化新数据（相对于新均值）
        u_t_centered = u_t - self.mean

        # 增量SVD更新
        self._incremental_svd_update(u_t_centered)

    def _incremental_svd_update(self, u_t_centered):
        """
        增量SVD更新算法

        给定当前的 U [D, k], S [k] 和新数据 x_centered [D]
        更新 U 和 S

        注意：所有数据都已在 CPU 上，无需再调用 .cpu()
        """
        if self.components is None:
            # 如果还没有主成分，用x初始化
            norm_x = torch.norm(u_t_centered)
            if norm_x > self.threshold:
                self.components = (u_t_centered / norm_x).unsqueeze(1)  # [D, 1]
                self.singular_values = norm_x.unsqueeze(0)  # [1]
            return

        # 所有数据都在 CPU 上，直接使用
        U = self.components  # [D, k]
        S = self.singular_values  # [k]
        k = U.shape[1]

        # 将x投影到当前子空间
        # a = U^T @ x  (在子空间内的坐标)
        a = torch.matmul(U.t(), u_t_centered)  # [k]

        # 计算残差（正交于当前子空间的部分）
        # r = x - U @ a
        r = u_t_centered - torch.matmul(U, a)  # [D]
        r_norm = torch.norm(r)

        if r_norm > self.threshold:
            # 残差显著，需要扩展子空间
            # 归一化残差得到新的基向量
            p = r / r_norm  # [D]

            # 构建增广矩阵 B = [[diag(S), a], [0, r_norm]]
            B = torch.zeros(k + 1, k + 1, dtype=u_t_centered.dtype)
            B[:k, :k] = torch.diag(S)
            B[:k, k] = a
            B[k, k] = r_norm

            # 对小矩阵做SVD
            Ub, Sb, _ = torch.linalg.svd(B, full_matrices=False)

            # 更新主成分基: U_new = [U, p] @ Ub
            U_aug = torch.cat([U, p.unsqueeze(1)], dim=1)  # [D, k+1]
            U_new = torch.matmul(U_aug, Ub)  # [D, k+1]

            # 保留top_k个成分
            new_k = min(self.top_k, len(Sb))
            valid = Sb[:new_k] > self.threshold
            if valid.sum() > 0:
                self.components = U_new[:, :new_k][:, valid]
                self.singular_values = Sb[:new_k][valid]

        else:
            # 残差很小，x基本落在当前子空间内
            # 只需要在子空间内旋转/更新奇异值
            B = torch.zeros(k, k + 1, dtype=u_t_centered.dtype)
            B[:, :k] = torch.diag(S)
            B[:, k] = a

            # SVD
            Ub, Sb, _ = torch.linalg.svd(B, full_matrices=False)

            # 更新: U_new = U @ Ub (旋转)
            new_k = min(self.top_k, k)
            self.components = torch.matmul(U, Ub[:, :new_k])
            self.singular_values = Sb[:new_k]

            # 过滤太小的奇异值
            valid = self.singular_values > self.threshold
            if valid.sum() > 0:
                self.components = self.components[:, valid]
                self.singular_values = self.singular_values[valid]

    def update_window_acc(self, acc):
        """更新 ACC window"""
        self.window_acc.append(acc)
        self.max_window_acc = self.window_acc[-self.max_ws-1:-1]
        self.min_window_acc = self.window_acc[-self.min_ws:]

    def update_window_ER(self, ER):
        """更新 ER window"""
        self.window_ER.append(ER)
        self.max_window_ER = self.window_ER[-self.max_ws-1:-1]
        self.min_window_ER = self.window_ER[-self.min_ws:]

    def update_window_CO(self, CO):
        """更新 ER window"""
        self.window_CO.append(CO)

    def get_failure_mode_acc(self, temp_acc=None):
        """
        基于 ACC 判断 failure mode（使用统计显著性检验）

        - forward failure: max_window 的 avg acc < min_window 的 avg acc（显著）
        - backward failure: max_window 的 avg acc > min_window 的 avg acc（显著）

        使用 Welch's t-test 进行显著性检验
        """
        if len(self.max_window_acc) < 1 or len(self.min_window_acc) < 1:
            return None, {}

        max_window = self.max_window_acc + ([temp_acc] if temp_acc is not None else [])
        min_window = self.min_window_acc + ([temp_acc] if temp_acc is not None else [])

        max_avg = np.mean(max_window)
        min_avg = np.mean(min_window)

        # 使用 Welch's t-test（不假设方差相等）
        if len(max_window) >= 2 and len(min_window) >= 2:
            t_stat, p_value = stats.ttest_ind(max_window, min_window, equal_var=False)
        else:
            p_value = 1.0
            t_stat = 0.0

        stats_info = {
            'max_avg_acc': max_avg,
            'min_avg_acc': min_avg,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level
        }

        if min_avg-max_avg > 0 and (min_avg - max_avg) > max_avg * 0.1:
            return 'forward', stats_info
        else:
            return None, stats_info


    def get_failure_mode_ER(self, temp_ER=None):
        """
        基于 ER 判断 forward failure（只检测 forward failure）

        - forward failure: max_window 的 avg ER > min_window 的 avg ER（显著）
          （ER 增大意味着当前更新更多地落在历史子空间内，偏离新方向）

        注意：backward failure 应使用 get_failure_mode_co() 基于 CO 来判断

        使用 Welch's t-test 进行显著性检验
        """
        if len(self.max_window_ER) < 2 or len(self.min_window_ER) < 2:
            return None, {}

        max_window = self.max_window_ER + ([temp_ER] if temp_ER is not None else [])
        min_window = self.min_window_ER + ([temp_ER] if temp_ER is not None else [])

        max_avg = np.mean(max_window)
        min_avg = np.mean(min_window)

        # 使用 Welch's t-test
        if len(max_window) >= 2 and len(min_window) >= 2:
            t_stat, p_value = stats.ttest_ind(max_window, min_window, equal_var=False)
        else:
            p_value = 1.0
            t_stat = 0.0

        stats_info = {
            'max_avg_ER': max_avg,
            'min_avg_ER': min_avg,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level
        }

        if max_avg - min_avg > 0 and (max_avg - min_avg) > min_avg * 0.1:
            return 'forward', stats_info
        else:
            return None, stats_info


    def get_failure_mode_acc_c(self):
        if len(self.window_acc) < 2:
            return None, {}

        window_len = min(self.max_ws, len(self.window_ER))
        slope, intercept = np.polyfit(range(window_len), self.window_acc[-window_len:], 1)

        stats_info = {
            'slope': slope,
            'intercept': intercept
        }

        if slope > 0:
            return 'forward', stats_info
        else:
            return None, stats_info


    def get_failure_mode_ER_c(self):
        if len(self.window_ER) < 2:
            return None, {}

        window_len = min(self.max_ws, len(self.window_ER))
        slope, intercept = np.polyfit(range(window_len), self.window_ER[-window_len:], 1)

        stats_info = {
            'slope': slope,
            'intercept': intercept
        }

        if slope < 0:
            return 'forward', stats_info
        else:
            return None, stats_info


    def get_failure_mode_CO(self):
        if len(self.window_CO) < 3:
            return None, {'reason': 'not enough history'}

        # 计算历史 CO 差值（使用 max_window 范围内的相邻差值）
        co_array = np.array(self.window_CO)

        # 取最近 max_ws 个样本（不包括最后一个）作为历史窗口
        history_start = max(0, len(co_array) - self.max_ws - 1)
        history_end = len(co_array) - 1

        if history_end - history_start < 2:
            return None, {'reason': 'not enough history for comparison'}

        # 计算历史窗口内的相邻差值
        history_co = co_array[history_start:history_end]
        difference_CO_history = np.abs(np.diff(history_co))  # 相邻元素的差
        avg_difference_CO = np.mean(difference_CO_history) if len(difference_CO_history) > 0 else 0

        # 当前 CO 差值（最后两个 CO 的差）
        difference_CO_current = np.abs(self.window_CO[-1] - self.window_CO[-2])

        stats_info = {
            'avg_difference_CO_history': float(avg_difference_CO),
            'difference_CO_current': float(difference_CO_current),
            'co_last': float(self.window_CO[-1]),
            'co_second_last': float(self.window_CO[-2]),
            'sign_changed': self.window_CO[-1] * self.window_CO[-2] < 0
        }

        # 判断条件：
        # 1. CO 符号发生变化（反方向）
        # 2. 当前差值比历史平均差值大
        # 3. 差值变化超过 10%
        # if (self.window_CO[-1] * self.window_CO[-2] < 0
        #         and difference_CO_current > avg_difference_CO
        #         and (difference_CO_current - avg_difference_CO) > avg_difference_CO * 3):
        if len(self.window_CO)<=3:
            return None, stats_info
        elif (difference_CO_current > avg_difference_CO
                and (difference_CO_current - avg_difference_CO) > avg_difference_CO * 3):
            return 'backward', stats_info
        else:
            return None, stats_info


    def get_empirical_fisher_on_subspace(self, model, loader, beta):
        """ 计算 (Q^T * g * g^T * Q) / n """
        device = next(model.parameters()).device

        # 检查 components 是否已初始化
        if self.components is None:
            return None, None

        Q = self.components.to(device)

        all_projected_grads_r = []
        all_projected_grads_f = []

        for batch_i, batch in enumerate(loader):
            if batch_i > 20: break

            images, clabels, labels = batch
            images, clabels, labels = images.to(device), clabels.to(device), labels.to(device)

            out = model(images)

            retain_mask = (labels == 0)
            forget_mask = (labels == 1)

            # retain
            model.zero_grad()
            retain_logits = out[retain_mask]
            retain_clabels = clabels[retain_mask]
            loss_retain = F.cross_entropy(retain_logits, retain_clabels)
            loss_retain.backward(retain_graph=True)

            grad_vec_r = self._flatten_grads_for_fisher(model)
            all_projected_grads_r.append(Q.t() @ grad_vec_r)

            # forget
            model.zero_grad()
            forget_logits = out[forget_mask]
            forget_clabels = clabels[forget_mask]
            loss_forget = F.cross_entropy(forget_logits, forget_clabels)
            loss_forget = -loss_forget * beta
            loss_forget.backward()

            grad_vec_f = self._flatten_grads_for_fisher(model)
            all_projected_grads_f.append(Q.t() @ grad_vec_f)

        G_hat_r = torch.stack(all_projected_grads_r).t()
        G_hat_f = torch.stack(all_projected_grads_f).t()
        return (G_hat_r @ G_hat_r.t()) / G_hat_r.size(1), (G_hat_f @ G_hat_f.t()) / G_hat_f.size(1)

    def _flatten_grads_for_fisher(self, model):
        """
        收集梯度

        如果 self.layer_name 已设置，只返回该层的梯度；
        否则返回整个模型的梯度（与 flatten_model_params 维度一致）
        """
        if self.layer_name:
            # 只获取指定层的梯度
            for name, p in model.named_parameters():
                if name == self.layer_name:
                    if p.grad is not None:
                        return p.grad.view(-1)
                    else:
                        return torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
            raise KeyError(f"Layer '{self.layer_name}' not found in model parameters")
        else:
            # 获取整个模型的梯度
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
                else:
                    grads.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            return torch.cat(grads)

    def solve_adaptive_lambda(self, model, train_loader, beta, epsilon=1e-3):
        """
        计算 adaptive lambda

        Returns:
            adaptive_lambda: 计算的 lambda 值（如果无法计算则为 None）
            M_R_hat: retain Fisher 矩阵
            M_F_hat: forget Fisher 矩阵
        """
        if self.components is None:
            return None, None, None

        model.eval()
        M_R_hat, M_F_hat = self.get_empirical_fisher_on_subspace(model, train_loader, beta)

        if M_R_hat is None or M_F_hat is None:
            return None, None, None

        # 求解 (M_R + eps*I)^-1 * M_F 的最大特征值
        inv_R = torch.inverse(M_R_hat + epsilon * torch.eye(M_R_hat.size(0)).to(M_R_hat.device))
        matrix_to_solve = inv_R @ M_F_hat

        # 最大特征值
        eigenvalues = torch.linalg.eigvals(matrix_to_solve).real
        return torch.max(eigenvalues).item(), M_R_hat, M_F_hat

    def compute_energy_ratio(self, model):
        if self.components is None:
            return None

        g, _ = self._flatten_grads_with_zeros(model)

        if g.numel() == 0:
            return None

        Q = self.components.to(g.device)

        if Q.shape[0] != g.shape[0]:
            return None

        g_para = Q @ (Q.T @ g)

        g_norm_sq = g.norm() ** 2
        if g_norm_sq < 1e-10:
            return None

        er = (g_para.norm() ** 2) / g_norm_sq
        return er.item()

    def compute_projection_coefficient(self, model):
        if self.components is None:
            return None

        g, _ = self._flatten_grads_with_zeros(model)

        if g.numel() == 0:
            return None

        Q = self.components.to(g.device)

        if Q.shape[0] != g.shape[0]:
            return None

        co = Q.T @ g
        return co.cpu()
