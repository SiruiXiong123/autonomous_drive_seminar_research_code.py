import math

def compute_tuoyuan_risk(vehicles_info, tau=2, v_s=10,
                     lane_width=2,v_l = 5,v_w=3):
    risk_list = []
    for idx, vehicle in enumerate(vehicles_info):
        dx, dy, dvx, dvy, length, width = vehicle

        dego = (0.5*v_w**2+0.5*v_l**2)**0.5
        dother = (0.5*width**2+0.5*length**2)**0.5
        dxy = (dx**2+dy**2)**0.5-dego-dother
        dvx_s = dvx / 3.6
        dvy_s = dvy / 3.6
        distsafe_x = (0.5 * length + 0.5 * v_l)
        distsafe_y = (0.5 * width + 0.5 * v_w)
        dvxb = dvx_s if dx * dvx < 0 else 0
        g = 9.8  # m/s**2
        a_c = distsafe_x + 5
        b_c = distsafe_y + 5
        u_x = 0.2 * g
        u_y = 0.1 * g
        dvyb = dvy_s if dy * dvy < 0 and abs(dy) > 0.5 * distsafe_y else 0
        a = a_c + dvxb ** 2 / (2 * abs(u_x)) + 0.2 * v_s
        b = b_c + dvyb ** 2 / (2 * abs(u_y))

        # Critical Region Repulsive Field
        Ec = 0.5 / (((dx / (a_c)) ** 4 + (dy / (b_c)) ** 4 + 1) ** 2)
        # Broader Region Repulsive Field
        Eb = 0.5 / (((dx / (a)) ** 2 + (dy / (b)) ** 2
                         + 1) ** 2)

        if dx == 50 and dy == 50 and dvx == 120 and dvy == 120:
            Ec = 0.0
            Eb = 0.0
        else:
            cfj = Ec + Eb
            risk_list.append(cfj)

        # print(f"车辆索引:{idx} ,关键域危险:{round(Ec, 3)},广域危险: {round(Eb, 3)},纵向距离:{round(a, 3)},横向距离:{round(b, 3)}")




    sum_risk = min(1.5,sum(risk_list))

    # print("总风险",round(sum_risk,3))

    return sum_risk




