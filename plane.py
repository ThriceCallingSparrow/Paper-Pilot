from optuna.exceptions import ExperimentalWarning
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import platform
import warnings
import optuna
import random
import math
import json
import os

warnings.filterwarnings("ignore", category=ExperimentalWarning)
# 屏蔽Optuna中间trial打印
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 固定鼠标按下坐标
FIXED_START_X = 182
FIXED_START_Y = 265

# X轴边界
END_DRAG_X_MIN = FIXED_START_X - 80
END_DRAG_X_MAX = FIXED_START_X - 1

# Y轴边界
max_dragY = math.sqrt(3) * 80
min_dragY = -math.sqrt(3) * 80
END_DRAG_Y_MIN = math.floor(FIXED_START_Y - max_dragY)
END_DRAG_Y_MAX = math.ceil(FIXED_START_Y - min_dragY)

# 智能自适应中文
sys_type = platform.system()
if sys_type == "Windows":
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
elif sys_type == "Darwin":  # MacOS
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC"]
else:  # Linux
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.facecolor'] = '#f0f0f0' # 浅灰底色和原图风格统一

# 创建结果文件夹
save_dir = "plane_optimize_result"
os.makedirs(save_dir, exist_ok=True)

class PaperPlaneAerodynamics:
    def __init__(self):
        # 预设飞机配置
        self.plane_configs = {
            1: {"cd0": 0.025, "ar": 1, "area": 0.1736, "clmax": 0.92, "name": "Eagle"},
            2: {"cd0": 0.03, "ar": 1.825, "area": 0.2028, "clmax": 0.55, "name": "Square"},
            3: {"cd0": 0.03, "ar": 0.85, "area": 0.2049, "clmax": 0.5, "name": "Dart"}
        }
        
        # 拖拽坐标
        self.startdragX = FIXED_START_X
        self.startdragY = FIXED_START_Y
        self.enddragX = FIXED_START_X - 45   # 默认释放X
        self.enddragY = FIXED_START_Y - 28   # 默认释放Y
        
        # 飞机设计参数
        self.plane_type = 3
        self.weight = 0.012
        self.element = 5
        self.winglets = 0
        
        # 初始角度、动力
        self.initial_angle = 0.0
        self.power = 0.0
        
        # 固定物理参数
        self.dt = 0.025
        self.g = 32.2
        self.err = 0.0
        self.onLanding = None
        
        # 状态变量
        self.cd = 0.0
        self.cl = 0.0
        self.difangle = 0.0
        self.drag = 0.0
        self.dy = 60.0
        self.lift = 0.0
        self.q = 0.0
        self.v = 0.0
        self.x = 0
        self.y = 0
        self.dx = 0.0
        self.angle = 0.0
    
    @staticmethod
    def as_mod(a, b):
        """模拟 ActionScript 取模规则: 结果符号与被除数保持一致"""
        mod = a % b
        if (a < 0 and mod > 0) or (a > 0 and mod < 0):
            mod -= b
        return mod
    
    def set_plane(self, plane_type):
        config = self.plane_configs.get(plane_type, self.plane_configs[1])
        self.plane_type = plane_type
        self.cd0 = config["cd0"]
        self.ar = config["ar"]
        self.area = config["area"]
        self.clmax = config["clmax"]
    
    def calc_angle_and_power(self):
        # 计算dragX/dragY
        dragX = self.startdragX - self.enddragX
        dragY = self.startdragY - self.enddragY
        
        # dragX 小于 0 强制置 0
        if dragX < 0:
            dragX = 0
        
        # 计算反正切弧度
        if dragX == 0:
            radians = 0.0
        else:
            radians = math.atan(dragY / dragX)
        
        # 转换为角度并钳位
        degrees = -math.degrees(radians)
        if degrees < -60:
            degrees = -60
            radians = -math.radians(degrees)
        if degrees > 60:
            degrees = 60
            radians = -math.radians(degrees)
        
        # 计算拖拽矢量长度
        loc3 = math.hypot(dragX, dragY)
        if loc3 > 80:
            loc3 = 80
        
        # 计算投掷动力
        self.power = loc3 * 3.4
        # 赋值初始飞行角度
        self.initial_angle = degrees
        # 同步飞行迭代用的 angle
        self.angle = self.initial_angle
        
    def apply_throw_penalty(self):
        """应用投掷力惩罚机制"""
        threshold = 200 + self.winglets * 10 + (self.weight - 0.012) * 1000
        self.err = self.power - threshold
        if self.err > 0:
            # 用力过猛惩罚
            if self.element > 5:
                self.element = 100
            else:
                self.element = -200
            self.weight = 0.012
    
    def init(self):
        """初始化飞行状态"""
        self.cl = self.clmax * (self.element / 1600)
        self.cd = self.cd0 + (self.cl ** 2) / (2.2 * self.ar)
        self.dy = 60.0
        self.y = 1
        self.v = self.power
    
    def calculate_initial_position(self):
        """根据初始角度和power计算初始dx/dy"""
        # 先执行坐标 → 角度/动力换算
        self.calc_angle_and_power()
        
        if self.power <= 0:
            self.dx = 0.0
            self.dy = 60.0
            return
        
        loc2 = self.power / 3.4
        loc2 = min(loc2, 80.0)
        radians = -math.radians(self.initial_angle)
        
        self.dx = math.cos(radians) * loc2 - math.sin(radians) * 50
        self.dy = math.sin(radians) * loc2 + math.cos(radians) * 50
    
    def fly_step(self):
        """执行单步飞行模拟"""
        if self.y <= -50:
            return False
        print(self.y)
        # 动压 q = 0.00115 * v²
        self.q = 0.00115 * (self.v ** 2)
        # 有效机翼面积, 扣除小翼影响
        effective_area = self.area - self.winglets * 0.01
        # 升力 & 阻力
        self.lift = self.q * effective_area * self.cl
        self.drag = self.q * effective_area * self.cd
        
        # 计算速度和角度更新
        theta_rad = math.radians(self.angle)
        self.v -= (self.weight * math.sin(theta_rad) + self.drag) * self.g * self.dt * (180/math.pi)
        
        # 更新轨迹角
        numerator = (self.lift - self.weight * math.cos(theta_rad)) * self.g * self.dt * (180/math.pi)
        denominator = self.weight * self.v
        self.angle += numerator / denominator
        
        # 速度方向修正
        if self.v < 0:
            self.v = -self.v
            self.angle = -self.angle
        self.angle = self.as_mod(self.angle, 360)
        
        self.dx += self.v * self.dt * math.cos(theta_rad) * 10
        self.dy += self.v * self.dt * math.sin(theta_rad) * 10
        
        self.x = round(self.dx)
        self.y = round(self.dy)
        
        # 着陆检测
        if self.y <= -50:
            self.y = -50
            if self.onLanding is not None:
                self.onLanding()
            return False
        return True
    
    @staticmethod
    def calc_length(xpos):
        """将游戏像素坐标转换为米"""
        length = (xpos - 40) / 44.5
        return max(0.0, length)
    
    def simulate_full_flight(self):
        """模拟完整飞行过程, 返回最终飞行距离 (米)"""
        self.set_plane(self.plane_type)
        self.calculate_initial_position()
        self.apply_throw_penalty()
        self.init()
        
        while self.fly_step():
            pass
        
        return self.calc_length(self.x)
    
def create_objective(plane_type):
    def objective(trial):
        # 固定随机种子保证可复现
        seed = trial.user_attrs.get("fixed_seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        
        plane = PaperPlaneAerodynamics()
        
        # 参数搜索空间
        plane.plane_type = plane_type  # 固定飞机类型
        weight_slider = trial.suggest_int("weight_slider", 0, 65)
        plane.weight = 0.012 + (weight_slider / 5000)
        plane.element = trial.suggest_int("element", 0, 80)
        plane.winglets = trial.suggest_categorical("winglets", [0, 1])
        
        # 优化拖拽结束坐标
        plane.enddragX = trial.suggest_int("enddragX", END_DRAG_X_MIN, END_DRAG_X_MAX)
        plane.enddragY = trial.suggest_int("enddragY", END_DRAG_Y_MIN, END_DRAG_Y_MAX)
        
        return plane.simulate_full_flight()
    return objective

def run_optimize_mode():
    """运行优化模式"""
    # 执行分飞机优化
    N_trials = 1000
    fixed_seed = 42
    best_results = {}
    for plane_type in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"正在优化飞机类型: {plane_type} - {PaperPlaneAerodynamics().plane_configs[plane_type]['name']}")
        print(f"{'='*60}")
        
        sampler = optuna.samplers.TPESampler(
            seed=fixed_seed,
            n_startup_trials=50,
            multivariate=True,
            n_ei_candidates=50
        )
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.NopPruner()
        )
        
        study.optimize(create_objective(plane_type), n_trials=N_trials)
        
        best_results[plane_type] = {
            "distance": study.best_value,
            "params": study.best_params,
            "study": study  # 保留study对象用于后续可视化
        }
        
        print(f"优化完成, 当前最优距离: {study.best_value:.2f} 米")
        
    # 统一输出所有飞机结果
    print("\n" + "="*80)
    print("三种纸飞机最优参数对比结果")
    print("="*80)
    
    # 按飞行距离排序
    sorted_planes = sorted(best_results.items(), key=lambda x: x[1]["distance"], reverse=True)
    
    df_list = []
    for rank, (plane_type, result) in enumerate(sorted_planes, 1):
        distance = result["distance"]
        params = result["params"]
        plane_config = PaperPlaneAerodynamics().plane_configs[plane_type]
        
        print(f"\n第{rank}名: {plane_config['name']} (编号: {plane_type})")
        print("-"*50)
        
        # 设计参数
        print("飞机设计参数")
        weight_slider = params["weight_slider"]
        weight_physical = 0.012 + (weight_slider / 5000)
        print(f"  重量滑块: {weight_slider:3d} (物理值: {weight_physical:.6f})")
        print(f"  升降舵滑块: {params['element']:3d}")
        winglets_status = "开启" if params["winglets"] == 1 else "关闭"
        print(f"  小翼: {winglets_status}")
        
        # 投掷坐标参数
        print("\n投掷坐标参数")
        print(f"  初始拖拽坐标: X={FIXED_START_X}, Y={FIXED_START_Y}")
        print(f"  最终释放坐标: X={params['enddragX']}, Y={params['enddragY']}")
        
        # 投掷操作参数
        print("\n投掷操作参数")
        temp_plane = PaperPlaneAerodynamics()
        temp_plane.enddragX = params["enddragX"]
        temp_plane.enddragY = params["enddragY"]
        temp_plane.calc_angle_and_power()
        initial_angle = temp_plane.initial_angle
        power = temp_plane.power
        
        if initial_angle > 0:
            angle_desc = "低头向下"
        elif initial_angle < 0:
            angle_desc = "抬头向上"
        else:
            angle_desc = "水平"
        print(f"  投掷角度: {initial_angle:5.1f}° ({angle_desc})")
        
        # 惩罚验证
        threshold = 200 + params["winglets"] * 10 + (weight_physical - 0.012) * 1000
        penalty_status = "未触发惩罚" if power <= threshold else "触发用力过猛惩罚"
        print(f"  惩罚检查: {penalty_status}")
        print(f"  安全阈值: {threshold:.2f} | 实际动力: {power:.2f}")
        
        # 参数有效性检查
        print("\n参数有效性检查")
        valid = True
        if not (0 <= weight_slider <= 100):
            print("  重量滑块超出范围 (0-100)")
            valid = False
        if not (0 <= params["element"] <= 100):
            print("  升降舵超出范围 (0-100)")
            valid = False
        if not (-60 <= initial_angle <= 60):
            print("  投掷角度超出范围 (-60°~60°)")
            valid = False
        if not (0 <= power <= 272):
            print("  初始速度超出范围 (0-272)")
            valid = False
        if valid:
            print("  所有参数均在有效范围内")
        
        # 最终结果
        print(f"\n理论最大飞行距离: {distance:.2f} 米")
        print("-"*50)
        
        # 组装表格数据
        df_list.append({
            "排名": rank,
            "机型编号": plane_type,
            "机型名称": plane_config['name'],
            "最优飞行距离(m)": round(distance,2),
            "重量滑块值": weight_slider,
            "升降舵滑块": params["element"],
            "小翼(1开0关)": params["winglets"],
            "初始拖拽X": FIXED_START_X,
            "初始拖拽Y": FIXED_START_Y,
            "最终释放X": params["enddragX"],
            "最终释放Y": params["enddragY"],
            "投掷角度(°)": initial_angle,
            "动力power": power,
            "安全阈值": round(threshold,2),
            "惩罚状态": penalty_status,
            "参数合法": "是" if valid else "否"
        })
        
    # 最终总结
    print("\n" + "="*80)
    print("最终总结")
    print("="*80)
    best_plane_type, best_result = sorted_planes[0]
    best_plane_name = PaperPlaneAerodynamics().plane_configs[best_plane_type]["name"]
    print(f"综合最优飞机: {best_plane_name} (编号: {best_plane_type})")
    print(f"最大飞行距离: {best_result['distance']:.2f} 米")
    print(f"比第二名远: {best_result['distance'] - sorted_planes[1][1]['distance']:.2f} 米")
    print("="*80)
    
    # 保存结果文件 Excel + JSON
    df_result = pd.DataFrame(df_list)
    excel_path = os.path.join(save_dir, "纸飞机优化结果汇总.xlsx")
    df_result.to_excel(excel_path, index=False)
    
    # 保存原始参数字典json
    json_save = {}
    for ptype, item in best_results.items():
        json_save[ptype] = {
            "max_distance": item["distance"],
            "best_params": item["params"]
        }
    json_path = os.path.join(save_dir, "最优参数原始数据.json")
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(json_save, f, ensure_ascii=False, indent=2)
        
    print(f"\n结果文件已保存至: {save_dir}")
    print(f"Excel: {excel_path}")
    print(f"JSON: {json_path}")
    
    # 批量绘制 Optuna 可视化图表并保存
    def plot_optuna_fig(study, plane_id, save_root):
        os.makedirs(save_root, exist_ok=True)
        save_prefix = os.path.join(save_root, f"机型{plane_id}")
        trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        param_names = list(study.best_params.keys())
        n_params = len(param_names)
        
        # ========== 优化迭代历史 ==========
        plt.figure(figsize=(9, 5.5))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title(f"机型{plane_id} 优化迭代历史", fontsize=13, pad=12)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_迭代历史.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # ========== 参数重要性 ==========
        from optuna.importance import get_param_importances, FanovaImportanceEvaluator
        imp_dict = get_param_importances(study, evaluator=FanovaImportanceEvaluator())
        params_imp = list(imp_dict.items())
        params_imp.sort(key=lambda x:x[1]) # 从小到大, 横向绘图从下往上
        
        plt.figure(figsize=(9,5.2))
        y_labels = [i[0] for i in params_imp]
        imp_vals = [i[1] for i in params_imp]
        bars = plt.barh(y_labels, imp_vals, color="#347ebc")
        # 柱状右侧标注数值
        for bar, val in zip(bars, imp_vals):
            if val < 0.01:
                plt.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2, "<0.01", va="center")
            else:
                plt.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2, f"{val:.2f}", va="center")
        plt.xlabel("超参数重要度")
        plt.ylabel("超参数名称")
        plt.title(f"机型{plane_id} 参数重要性排序", fontsize=14, pad=13)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_参数重要性.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # ========== 平行坐标 ==========
        # 提取参数+目标值
        data_arr = []
        obj_list = []
        top_k = 30
        sorted_trials = sorted(trials, key=lambda x:x.value)[:top_k]
        for t in sorted_trials:
            row = [t.params[p] for p in param_names]
            data_arr.append(row)
            obj_list.append(t.value)
        data_arr = np.array(data_arr)
        obj_arr = np.array(obj_list)
        
        fig, ax = plt.subplots(figsize=(13,6.5))
        # x轴位置
        x_pos = np.arange(n_params)
        norm = plt.Normalize(obj_arr.min(), obj_arr.max())
        cmap = plt.cm.Blues
        
        # 逐行画折线
        for idx in range(len(sorted_trials)):
            color = cmap(norm(obj_arr[idx]))
            ax.plot(x_pos, data_arr[idx], color=color, linewidth=0.7)
        # 坐标轴设置
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_names, rotation=22)
        ax.set_title(f"机型{plane_id} 参数平行坐标", fontsize=14, pad=13)
        ax.grid(alpha=0.3)
        # 右侧色条
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Objective Value")
        plt.subplots_adjust(left=0.06, right=0.93, top=0.91, bottom=0.16)
        plt.savefig(f"{save_prefix}_平行坐标.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # ========== 参数切片分布图 ==========
        if n_params <= 4:
            nrow, ncol = 1, n_params
        else:
            nrow = 2
            ncol = int(np.ceil(n_params / nrow))
            
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 5.8), squeeze=False)
        axes = axes.flatten()
        
        # 提取所有样本
        xs_dict = {p: [] for p in param_names}
        ys_list = []
        for t in trials:
            ys_list.append(t.value)
            for p in param_names:
                xs_dict[p].append(t.params[p])
        ys = np.array(ys_list)
        # 用目标值映射颜色
        norm = plt.Normalize(ys.min(), ys.max())
        
        for i, pname in enumerate(param_names):
            ax = axes[i]
            x = np.array(xs_dict[pname])
            sc = ax.scatter(x, ys, c=ys, cmap="Blues", norm=norm, s=16, edgecolor="#555555", linewidth=0.4)
            ax.set_xlabel(pname)
            ax.set_ylabel("Objective")
            ax.grid(alpha=0.3)
            
        # 多余子图隐藏
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)
            
        # 总标题 + 精细间距
        fig.suptitle(f"机型{plane_id} 各参数采样分布", fontsize=13, y=0.98)
        fig.subplots_adjust(top=0.91, bottom=0.13, left=0.05, right=0.94, wspace=0.35, hspace=0.38)
        # 右侧统一色条
        cbar_ax = fig.add_axes([0.955, 0.13, 0.016, 0.78])
        fig.colorbar(sc, cax=cbar_ax, label='目标函数值')
        
        plt.savefig(f"{save_prefix}_参数切片分布.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    # 循环每个机型绘图
    for plane_type, res in best_results.items():
        plot_optuna_fig(res["study"], plane_type, save_dir)

    # 三种飞机最优距离对比柱状图
    plt.figure(figsize=(10,5))
    names = [PaperPlaneAerodynamics().plane_configs[k]['name'] for k,_ in sorted_planes]
    dists = [v["distance"] for _,v in sorted_planes]
    bars = plt.bar(names, dists, color=["#ff7f0e","#1f77b4","#2ca02c"])
    for bar,d in zip(bars,dists):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{d:.2f}m",ha="center")
    plt.title("三种机型最优飞行距离对比",fontsize=14)
    plt.ylabel("飞行距离(米)")
    plt.grid(axis="y",alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"机型最优距离对比柱状图.png"),dpi=300)
    plt.close()

def run_debug_mode():
    """调试模式: 给定参数计算飞行距离"""
    print("\n" + "="*60)
    print("进入调试模式 - 输入参数计算飞行距离")
    print("="*60)
    
    # 获取用户输入的参数
    try:
        # 飞机类型
        plane_type = int(input("请输入飞机类型 (1=Eagle, 2=Square, 3=Dart): "))
        if plane_type not in [1,2,3]:
            print("飞机类型只能是1/2/3, 默认使用3(Dart)")
            plane_type = 3
        
        # 重量滑块
        weight_slider = int(input("请输入重量滑块值 (0-100): "))
        weight_slider = max(0, min(100, weight_slider))  # 钳位到合法范围
        
        # 升降舵滑块
        element = int(input("请输入升降舵滑块值 (0-100): "))
        element = max(0, min(100, element))  # 钳位到合法范围
        
        # 小翼
        winglets = int(input("请输入小翼状态 (0=关闭, 1=开启): "))
        winglets = 1 if winglets == 1 else 0
        
        # 释放坐标X
        enddragX = int(input(f"请输入释放坐标X ({END_DRAG_X_MIN} - {END_DRAG_X_MAX}): "))
        enddragX = max(END_DRAG_X_MIN, min(END_DRAG_X_MAX, enddragX))
        
        # 释放坐标Y
        enddragY = int(input(f"请输入释放坐标Y ({END_DRAG_Y_MIN} - {END_DRAG_Y_MAX}): "))
        enddragY = max(END_DRAG_Y_MIN, min(END_DRAG_Y_MAX, enddragY))
        
    except ValueError:
        print("输入参数格式错误, 使用默认参数")
        plane_type = 1
        weight_slider = 44
        element = 34
        winglets = 1
        enddragX = 118
        enddragY = 264
    
    # 实例化飞机并设置参数
    plane = PaperPlaneAerodynamics()
    plane.plane_type = plane_type
    plane.weight = 0.012 + (weight_slider / 5000)
    plane.element = element
    plane.winglets = winglets
    plane.enddragX = enddragX
    plane.enddragY = enddragY
    
    # 计算飞行距离
    distance = plane.simulate_full_flight()
    
    # 输出调试结果
    print("\n" + "-"*50)
    print("调试模式计算结果")
    print("-"*50)
    print(f"飞机类型: {plane_type} ({plane.plane_configs[plane_type]['name']})")
    weight_physical = 0.012 + (weight_slider / 5000)
    print(f"重量滑块: {weight_slider} (物理值: {weight_physical:.6f})")
    print(f"升降舵滑块: {element}")
    print(f"小翼状态: {'开启' if winglets == 1 else '关闭'}")
    print(f"初始拖拽坐标: X={FIXED_START_X}, Y={FIXED_START_Y}")
    print(f"最终释放坐标: X={enddragX}, Y={enddragY}")
    
    # 计算投掷角度和动力
    temp_plane = PaperPlaneAerodynamics()
    temp_plane.enddragX = enddragX
    temp_plane.enddragY = enddragY
    temp_plane.calc_angle_and_power()
    initial_angle = temp_plane.initial_angle
    power = temp_plane.power
    angle_desc = "低头向下" if initial_angle > 0 else "抬头向上" if initial_angle < 0 else "水平"
    print(f"投掷角度: {initial_angle:.1f}° ({angle_desc})")
    print(f"投掷动力: {power:.2f}")
    
    # 惩罚检查
    threshold = 200 + winglets * 10 + (weight_physical - 0.012) * 1000
    penalty_status = "未触发惩罚" if power <= threshold else "触发用力过猛惩罚"
    print(f"惩罚状态: {penalty_status} (安全阈值: {threshold:.2f})")
    
    # 最终距离
    print(f"\n计算得出飞行距离: {distance:.2f} 米")
    print("-"*50)
    
if __name__ == "__main__":
    # 获取模式选择标志
    mode_flag = input("请输入运行模式 (1=优化模式, 2=调试模式): ")
    
    # 根据标志选择运行模式
    if mode_flag == "1":
        run_optimize_mode()
    elif mode_flag == "2":
        run_debug_mode()
    else:
        print(f"无效的模式标志: {mode_flag}, 请输入1或2")