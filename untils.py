import math

FIXED_START_X = 182
FIXED_START_Y = 265
def calc_power_angle(sx, sy, ex, ey):
    # 拖拽偏移
    dragX = sx - ex
    dragY = sy - ey

    # 求弧度、角度
    if dragX == 0:
        radians = 0
    else:
        radians = math.atan(dragY / dragX)
    degrees = -radians * 180 / math.pi

    # 角度限幅 ±60°
    degrees = max(-60, min(60, degrees))
    radians = -degrees * math.pi / 180

    # 拖拽长度
    dist = math.hypot(dragX, dragY)
    dist = min(dist, 80) #上限80

    # 最终真实力量
    power = dist * 3.4
    return power, degrees

ex, ey = 338, 289
power, angle = calc_power_angle(FIXED_START_X,FIXED_START_Y,ex,ey)
print("初始力量: ", power)
print("初始角度: ", angle)

# 已知初始参数
target_power = 180.2
target_angle = -31.89079180184571
def reverse_calc_endpoint(power, degrees, sx, sy):
    # 1. 反推拖拽距离dist
    dist = power / 3.4
    # 正向有dist上限80，这里同步限制
    dist = min(dist, 80)
    
    # 2. 还原弧度（和正向公式对应 radians = -degrees * pi/180）
    rad = -degrees * math.pi / 180
    
    # 3. 由角度和距离求dragX, dragY
    dragX = dist * math.cos(rad)
    dragY = dist * math.sin(rad)
    
    # 4. dragX = sx - ex → ex = sx - dragX
    #    dragY = sy - ey → ey = sy - dragY
    ex = sx - dragX
    ey = sy - dragY
    
    return ex, ey

# 执行反推
ex, ey = reverse_calc_endpoint(target_power, target_angle, FIXED_START_X, FIXED_START_Y)

# 校验：用正向函数验算，看是否还原力量角度
def calc_power_angle(sx, sy, ex, ey):
    dragX = sx - ex
    dragY = sy - ey

    if dragX == 0:
        radians = 0
    else:
        radians = math.atan(dragY / dragX)
    degrees = -radians * 180 / math.pi

    degrees = max(-60, min(60, degrees))
    radians = -degrees * math.pi / 180

    dist = math.hypot(dragX, dragY)
    dist = min(dist, 80)

    power = dist * 3.4
    return power, degrees

check_power, check_angle = calc_power_angle(FIXED_START_X, FIXED_START_Y, ex, ey)

# 打印结果
print(f"\n反推得到拖拽终点 ex = {ex:.6f}, ey = {ey:.6f}")
print("\n正向校验结果：")
print(f"校验力量: {check_power:.6f} (目标{target_power})")
print(f"校验角度: {check_angle:.6f} (目标{target_angle})")