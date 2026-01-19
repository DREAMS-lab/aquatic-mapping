"""ground truth field generation matching ROS2 field generators"""
import numpy as np


def generate_ground_truth_field(field_type, width=25.0, height=25.0, resolution=0.5):
    """generate EXACT ground truth field matching field generators

    args:
        field_type: radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt
        width: field width [m]
        height: field height [m]
        resolution: grid spacing [m]

    returns:
        X: (H, W) x coordinates
        Y: (H, W) y coordinates
        temp_field: (H, W) temperature field [°C]
    """
    x_grid = np.arange(0, width + 1e-9, resolution)
    y_grid = np.arange(0, height + 1e-9, resolution)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')

    center_x, center_y = 12.5, 12.5
    base_temp = 20.0
    amplitude = 10.0

    if field_type == 'radial':
        sigma = 5.0
        gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        temp_field = base_temp + amplitude * gaussian

    elif field_type == 'y_compress':
        sigma_x, sigma_y = 7.0, 2.5
        gaussian = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) +
                          (Y - center_y)**2 / (2 * sigma_y**2)))
        temp_field = base_temp + amplitude * gaussian

    elif field_type == 'x_compress':
        sigma_x, sigma_y = 2.5, 7.0
        gaussian = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) +
                          (Y - center_y)**2 / (2 * sigma_y**2)))
        temp_field = base_temp + amplitude * gaussian

    elif field_type == 'y_compress_tilt':
        sigma_x, sigma_y = 7.0, 2.5
        theta = np.pi / 4.0
        X_rot = (X - center_x) * np.cos(theta) + (Y - center_y) * np.sin(theta)
        Y_rot = -(X - center_x) * np.sin(theta) + (Y - center_y) * np.cos(theta)
        gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
        temp_field = base_temp + amplitude * gaussian

    elif field_type == 'x_compress_tilt':
        sigma_x, sigma_y = 2.5, 7.0
        theta = np.pi / 4.0
        X_rot = (X - center_x) * np.cos(theta) + (Y - center_y) * np.sin(theta)
        Y_rot = -(X - center_x) * np.sin(theta) + (Y - center_y) * np.cos(theta)
        gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
        temp_field = base_temp + amplitude * gaussian

    else:
        raise ValueError(f"unknown field type: {field_type}")

    return X, Y, temp_field
