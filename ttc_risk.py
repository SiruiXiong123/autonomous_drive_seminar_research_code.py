import math


def compute_ttc_risk(vehicles_info, tau=2, v_s=10,
                     lane_width=2,v_l=5,v_w=3):
    risk_list = []
    for idx, vehicle in enumerate(vehicles_info):
        if len(vehicle) != 6:
            print(f"⚠️ WARNING: Invalid vehicle data format: {vehicle}")
            continue

        dx, dy, dvx, dvy, length, width = vehicle  # 解析 6 维信息
        if dx == -50 and dy == -50 and dvx == -120 and dvy == -120:
            vertical_risk = 0.0
            lateral_risk = 0.0
        else:

            dvx_s = dvx/3.6
            dvy_s = dvy/3.6
            distsafe_x = (0.5 * length + 0.5 * v_l)
            distsafe_y = (0.5 * width+0.5*v_w)
            dx_edge = max(0, abs(dx) - distsafe_x)
            dy_edge = max(0, abs(dy) - distsafe_y)
            if dx*dvx and dx>1.5*distsafe_x > 0:
                decay_factor = math.exp(-dx/dvx_s)
            else:
                decay_factor = 1
            ttc_x = abs(dx_edge) / (abs(dvx_s) + 0.01)  # 防止除 0
            vertical_risk = math.exp(-ttc_x / tau)*decay_factor
            lateral_risk = 1/(1+(dy_edge/0.45*lane_width)**2)

            if idx == 0:
                print("dy,dvy,纵向风险,横向风险",dy,dvy,vertical_risk,lateral_risk)
        # **最终风险**
        full_risk = 0.5*(lateral_risk + vertical_risk)

        risk_list.append(full_risk)

    sum_risk = sum(risk_list)

    return sum_risk

#===============================风险分析

# dx是其他车中心纵向距离-ego中心距离，当小车正在直线行驶还没到路中央时，
# dx可能小于0(此时其他车从右往左直行),然后如果此时由于dvx=其他车纵向距离-ego车纵向距离
# ，如果ego车速很快，dvx也是小于0的，那么dx*dvx就会大于0，此时纵向碰撞风险失效。
# 然后分析横向碰撞风险，两车中心点的距离横向在缩短，但如过其他车的车长较大(例如大货车),
# 而ego是私家车，其他车从右往左直行，ego执行，此时dy<0，如果两辆车保持直行横向速度都为0，那横向碰撞风险也会失效