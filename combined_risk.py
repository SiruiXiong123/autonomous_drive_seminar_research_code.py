import math


def compute_combined_risk(vehicles_info, tau=2, v_s=10,
                          lane_width=2, v_l=5, v_w=3, weight_ttc=0.2, weight_ellipse=0.8):

    risk_list = []

    for idx, vehicle in enumerate(vehicles_info):
        if len(vehicle) != 6:
            print(f"⚠️ WARNING: Invalid vehicle data format: {vehicle}")
            continue

        dx, dy, dvx, dvy, length, width = vehicle  # 解析 6 维信息

        # **TTC 风险计算**
        if dx == -50 and dy == -50 and dvx == -120 and dvy == -120:
            vertical_risk = 0.0
            lateral_risk = 0.0
        else:
            dvx_s = dvx / 3.6
            dvy_s = dvy / 3.6
            distsafe_x = (0.5 * length + 0.5 * v_l)
            distsafe_y = (0.5 * width + 0.5 * v_w)
            dx_edge = max(0, abs(dx) - distsafe_x)
            dy_edge = max(0, abs(dy) - distsafe_y)

            if dx * dvx > 0:
                decay_factor = math.exp(-dx / (dvx_s + 1e-3))  # 避免除零
            else:
                decay_factor = 1

            ttc_x = dx_edge / (abs(dvx_s) + 1e-3)  # 避免除零
            vertical_risk = math.exp(-ttc_x / tau) * decay_factor
            lateral_risk = 1 / (1 + (dy_edge / (0.45 * lane_width)) ** 2)

        ttc_risk = 0.5 * (lateral_risk + vertical_risk)  # TTC 总风险

        # **椭圆势场风险计算**
        dego = math.sqrt(0.5 * v_w ** 2 + 0.5 * v_l ** 2)
        dother = math.sqrt(0.5 * width ** 2 + 0.5 * length ** 2)
        dxy = math.sqrt(dx ** 2 + dy ** 2) - dego - dother

        g = 9.8  # m/s² 重力加速度
        a_c = distsafe_x + 6
        b_c = distsafe_y + 3 + 0.5 * length
        u_x = 0.3 * g
        u_y = 0.1 * g
        dvxb = dvx_s if dx * dvx < 0 else 0
        dvyb = dvy_s if dy * dvy < 0 and abs(dy) > 0.5 * distsafe_y else 0

        a = a_c + dvxb ** 2 / (2 * abs(u_x)) + 0.1 * v_s
        b = b_c + dvyb ** 2 / (2 * abs(u_y))

        # 计算椭圆势场的风险
        Ec = 0.5 / (((dx / a_c) ** 4 + (dy / b_c) ** 4 + 1) ** 2)  # 关键区域势场
        Eb = 0.5 / (((dx / a) ** 2 + (dy / b) ** 2 + 1) ** 2)  # 扩展区域势场

        ellipse_risk = Ec + Eb

        # **最终风险加权求和**
        final_risk = weight_ttc * ttc_risk + weight_ellipse * ellipse_risk
        risk_list.append(final_risk)

        print(f"车辆索引:{idx}, TTC 风险: {ttc_risk:.3f}, 椭圆势场风险: {ellipse_risk:.3f}, 最终风险: {final_risk:.3f}")

    # 归一化最终风险，设定上限 1.5
    sum_risk = min(1.5, sum(risk_list))
    print(f"总风险: {sum_risk:.3f}")

    return sum_risk
