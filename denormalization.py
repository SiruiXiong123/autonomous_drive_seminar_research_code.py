import math


def denormalize_lidar_data(vehicles_info, perceive_distance=50, max_speed_km_h=120):
    denormalized_data = []

    if isinstance(vehicles_info, list) and len(vehicles_info) % 6 == 0:
        vehicles_info = [vehicles_info[i:i + 6] for i in range(0, len(vehicles_info), 6)]

    if not all(isinstance(v, list) and len(v) == 6 for v in vehicles_info):
        print(f"ERROR: Invalid vehicles_info format: {vehicles_info}, expected list of [dx, dy, dvx, dvy, length, width].")
        return []


    for vehicle_info in vehicles_info:
        dx_norm, dy_norm, dvx_norm, dvy_norm, length_norm, width_norm = vehicle_info

        dx_real = -(dx_norm * 2 - 1) * perceive_distance
        dy_real = -(dy_norm * 2 - 1) * perceive_distance
        dvx_real = -(dvx_norm * 2 - 1) * max_speed_km_h
        dvy_real = -(dvy_norm * 2 - 1) * max_speed_km_h

        length_real = length_norm * 10
        width_real = width_norm * 2.5

        denormalized_data.append([dx_real, dy_real, dvx_real, dvy_real, length_real, width_real])

    return denormalized_data