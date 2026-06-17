from optuna.importance import get_param_importances
from flask import Flask, request, Response
from collections import deque
import pandas as pd
import threading
import optuna
import time
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 固定鼠标按下坐标配置
FIXED_START_X = 182
FIXED_START_Y = 265
# X边界
END_DRAG_X_MIN = FIXED_START_X - 80
END_DRAG_X_MAX = FIXED_START_X - 1
# Y边界
max_dragY = math.sqrt(3) * 80
min_dragY = -math.sqrt(3) * 80
END_DRAG_Y_MIN = math.floor(FIXED_START_Y - max_dragY)
END_DRAG_Y_MAX = math.ceil(FIXED_START_Y - min_dragY)

# 优化次数
N_trials = 500

app = Flask(__name__)

results = {}
task_counter = 0
trial_records = []
task_queue = deque()
opt_finished = False
result_lock = threading.Lock()
opt_finish_lock = threading.Lock()

# 优化目标函数
def objective(trial):
    global task_counter

    # 定义参数空间
    params = {
        "task_id": task_counter,
        # 飞机设计参数
        "selectedPlane": trial.suggest_int("selectedPlane", 1, 3),  # 1=Eagle, 2=Square, 3=Dart
        "weight": trial.suggest_int("weight", 1, 100),  # 重量
        "element": trial.suggest_int("element", 0, 100),  # 升降舵
        "winglets": trial.suggest_int("winglets", 0, 1),  # 小翼
        # 鼠标拖拽坐标参数
        "startdragX": FIXED_START_X,
        "startdragY": FIXED_START_Y,
        "enddragX": trial.suggest_int("enddragX", END_DRAG_X_MIN, END_DRAG_X_MAX),  # 鼠标释放X坐标（必须小于startdragX）
        "enddragY": trial.suggest_int("enddragY", END_DRAG_Y_MIN, END_DRAG_Y_MAX)   # 鼠标释放Y坐标
    }
    '''
    params = {
        "task_id": task_counter, # 48.2
        "selectedPlane": 1,
        "weight": 44,
        "element": 34,
        "winglets": 1,
        "startdragX": FIXED_START_X,
        "startdragY": FIXED_START_Y,
        "enddragX": 118,
        "enddragY": 264
    }
    '''
    
    task_counter += 1
    task_queue.append(params)
    
    # 等待结果+超时
    wait_start = time.time()
    timeout_limit = 35
    while params["task_id"] not in results:
        if time.time() - wait_start > timeout_limit:
            print(f"任务{params['task_id']}超时无返回，判定失败")
            # 超时直接记录空数据
            trial_records.append({
                "task_id": params["task_id"],
                "selectedPlane": params["selectedPlane"],
                "weight": params["weight"],
                "element": params["element"],
                "winglets": params["winglets"],
                "startdragX": params["startdragX"],
                "enddragX": params["enddragX"],
                "enddragY": params["enddragY"],
                "throw_power": 0,
                "throw_angle": 0,
                "distance": 0,
                "success": False
            })
            return 0.0
        time.sleep(0.01)
        
    # pop捕获键不存在异常
    try:
        with result_lock:
            result = results.pop(params["task_id"])
    except KeyError:
        trial_records.append({
            "task_id": params["task_id"],
            "selectedPlane": params["selectedPlane"],
            "weight": params["weight"],
            "element": params["element"],
            "winglets": params["winglets"],
            "startdragX": params["startdragX"],
            "enddragX": params["enddragX"],
            "enddragY": params["enddragY"],
            "throw_power": 0,
            "throw_angle": 0,
            "distance": 0,
            "success": False
        })
        return 0.0
    
    # 存入记录
    trial_records.append({
        "task_id": params["task_id"],
        "selectedPlane": params["selectedPlane"],
        "weight": params["weight"],
        "element": params["element"],
        "winglets": params["winglets"],
        "startdragX": params["startdragX"],
        "enddragX": params["enddragX"],
        "enddragY": params["enddragY"],
        "throw_power": result["power"],
        "throw_angle": result["angle"],
        "distance": result["distance"],
        "success": result["success"]
    })

    # 如果模拟失败，返回0
    if not result["success"]:
        return 0.0
    # 优化目标：最大化飞行距离
    return result["distance"]

# crossdomain路由
@app.route("/crossdomain.xml")
def crossdomain():
    xml = '''<?xml version="1.0"?>
<!DOCTYPE cross-domain-policy SYSTEM "http://www.adobe.com/xml/dtds/cross-domain-policy.dtd">
<cross-domain-policy>
    <allow-access-from domain="*" to-ports="5000"/>
</cross-domain-policy>'''
    return Response(xml, mimetype="application/xml")

# Flask路由
@app.route('/get_task')
def get_task():
    if not task_queue:
        return "response=no_task"
    
    # 取出最早的任务
    task = task_queue.popleft()
    
    # 以键值对格式返回
    return "&".join([f"{k}={v}" for k, v in task.items()])

@app.route('/')
def index():
    return "Service Running OK, use /get_task"

@app.route('/submit_result', methods=['POST'])
def submit_result():
    try:
        task_id = int(request.form['task_id'])
        distance = float(request.form['distance'])
        success = request.form['success'] == 'true'
        power = float(request.form['power'])
        angle = float(request.form['angle'])
    except:
        return "ret=err"
    with result_lock:
        results[task_id] = {
            "distance": distance,
            "success": success,
            "power": power,
            "angle": angle
        }
    return "ret=ok"

def monitor_opt():
    global opt_finished
    while True:
        with opt_finish_lock:
            if opt_finished:
                print("\n优化全部完成，即将关闭服务...")
                # 给文件写入缓冲时间
                time.sleep(1)
                # 终止进程
                import os
                os._exit(0)
        time.sleep(1)

# 运行优化的线程函数
def run_optimization():
    # 创建Optuna研究
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),  # 固定种子保证可复现
        study_name="paperpilot_optimization_as2"
    )
    
    # 运行优化
    print("开始优化...")
    study.optimize(objective, n_trials=N_trials)  # 500次试验足够找到最优解
    
    # 打印结果
    print("\n优化完成!")
    print("最优距离:", study.best_value)
    best_in = study.best_params

    # 筛选最优参数对应的记录
    best_row = None
    for item in trial_records:
        if (item["selectedPlane"] == best_in["selectedPlane"] and
            item["weight"] == best_in["weight"] and
            item["element"] == best_in["element"] and
            item["winglets"] == best_in["winglets"] and
            item["enddragX"] == best_in["enddragX"] and
            item["enddragY"] == best_in["enddragY"]):
            best_row = item
            break

    print("最优设计参数:")
    for key, value in best_in.items():
        if key == "selectedPlane":
            plane_names = {1: "Eagle", 2: "Square", 3: "Dart"}
            print(f"  {key}: {value} ({plane_names[value]})")
        else:
            print(f"  {key}: {value:.2f}")

    if best_row is not None:
        print("最优出手参数:")
        print(f"  throw_power: {best_row['throw_power']:.2f}")
        print(f"  throw_angle: {best_row['throw_angle']:.2f}")

    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_name = "optimization_results_as2.csv"
    csv_path = os.path.join(base_dir, csv_name)
    img1_path = os.path.join(base_dir, "param_importance.png")
    img2_path = os.path.join(base_dir, "optimization_convergence.png")

    # 保存结果
    df = pd.DataFrame(trial_records)
    df.to_csv(csv_path, index=False)
    if os.path.exists(csv_path):
        print(f"CSV生成成功，大小：{os.path.getsize(csv_path)} bytes")
    else:
        print("CSV写入失败")

    # 参数重要性图
    importance = get_param_importances(study)
    params = list(importance.keys())
    values = list(importance.values())

    plt.figure(figsize=(8, 5))
    plt.barh(params, values)
    plt.xlabel("Importance")
    plt.title("Parameter Importance (Optuna)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(img1_path, dpi=300)
    plt.close()
    plt.clf()
    if os.path.exists(img1_path):
        print(f"参数重要性图生成成功")

    # 收敛曲线
    best_values = []
    current_best = -float("inf")
    for t in study.trials:
        if t.value is not None:
            current_best = max(current_best, t.value)
            best_values.append(current_best)

    plt.figure(figsize=(8, 5))
    plt.plot(best_values)
    plt.xlabel("Iteration (Trial)")
    plt.ylabel("Best Value So Far")
    plt.title("Optimization Convergence Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(img2_path, dpi=300)
    plt.close()
    plt.clf()
    if os.path.exists(img2_path):
        print(f"收敛曲线图生成成功")

    # 标记优化完成
    with opt_finish_lock:
        global opt_finished
        opt_finished = True

if __name__ == '__main__':
    import logging
    # 屏蔽werkzeug访问日志
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    # 启动优化线程
    optimization_thread = threading.Thread(target=run_optimization)
    optimization_thread.start()

    # 监控线程
    monitor_thread = threading.Thread(target=monitor_opt, daemon=True)
    monitor_thread.start()

    # 启动Flask服务器
    print("服务器启动在 http://127.0.0.1:5000\n")
    print("请现在打开修改后的PaperPilot SWF文件...\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)