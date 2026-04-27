import torch
import numpy as np
def gather_nd_torch(params, indices, batch_dim=1):
    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    n_indices = indices.size(-2)  # x
    n_pos = indices.size(-1)  # m

    # reshape leadning batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    indices = indices.reshape(batch_size, n_indices, n_pos)

    # build gather indices
    # gather for each of the data point in this "batch"
    batch_enumeration = torch.arange(batch_size).unsqueeze(1)
    gather_dims = [indices[:, :, i] for i in range(len(grid_dims))]
    gather_dims.insert(0, batch_enumeration)
    gathered = params[gather_dims]

    # reshape back to the shape with leading batch dims
    gathered = gathered.reshape(*batch_dims, n_indices, c_dim)
    return gathered

def interpolate(grid_3d,
                sampling_points):
    """Trilinear interpolation on a 3D regular grid.
    This is a porting of TensorFlow Graphics implementation of trilinear interpolation.
    Check https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/trilinear.py
    for more details.
    Args:
      grid_3d: A tensor with shape `[A1, ..., An, H, W, D, C]` where H, W, D are
        height, width, depth of the grid and C is the number of channels.
      sampling_points: A tensor with shape `[A1, ..., An, M, 3]` where M is the
        number of sampling points. Sampling points outside the grid are projected
        in the grid borders.
    Returns:
      A tensor of shape `[A1, ..., An, M, C]`
    """

    grid_3d_shape = grid_3d.size()#; print('the grid size is:', grid_3d_shape) 
    sampling_points_shape = sampling_points.size()#; print('the spts size is:', sampling_points_shape) 
    voxel_cube_shape = grid_3d_shape[-4:-1]  # [H, W, D]
    batch_dims = sampling_points_shape[:-2]  # [A1, ..., An]
    num_points = sampling_points_shape[-2]  # M
    # print('voxel_cube_shape is :',voxel_cube_shape)
    # print('batch_dims is :', batch_dims)
    # print('num_points is :', num_points)

    bottom_left = torch.floor(sampling_points)
    top_right = bottom_left + 1
    bottom_left_index = bottom_left.type(torch.int32)
    top_right_index = top_right.type(torch.int32)
    # print('bottom_left_index:',bottom_left_index)
    # print('top_right_index:',top_right_index)

    x0_index, y0_index, z0_index = torch.unbind(bottom_left_index, dim=-1)
    x1_index, y1_index, z1_index = torch.unbind(top_right_index, dim=-1)
    index_x = torch.concat([x0_index, x1_index, x0_index, x1_index,
                            x0_index, x1_index, x0_index, x1_index], dim=-1)
    index_y = torch.concat([y0_index, y0_index, y1_index, y1_index,
                            y0_index, y0_index, y1_index, y1_index], dim=-1)
    index_z = torch.concat([z0_index, z0_index, z0_index, z0_index,
                            z1_index, z1_index, z1_index, z1_index], dim=-1)
    indices = torch.stack([index_x, index_y, index_z], dim=-1)
    # print('interpolation index:', indices)

    # clip_value_max = torch.from_numpy(np.ndarray(list(voxel_cube_shape)) - 1)
    # clip_value_min = torch.zeros_like(clip_value_max)
    clip_value_max = torch.tensor(voxel_cube_shape)
    clip_value_min = torch.zeros_like(clip_value_max)
    # # print(clip_value_max,clip_value_min)
    # print(indices.shape)
    # print(clip_value_max.shape,clip_value_min.shape)
    indices = torch.clamp(indices, min=clip_value_min.to(indices.device), max=clip_value_max.to(indices.device))
    # print('interpolation index clamped:', indices)

    content = gather_nd_torch(
        params=grid_3d, indices=indices.long(), batch_dim=len(batch_dims))

    distance_to_bottom_left = sampling_points - bottom_left
    distance_to_top_right = top_right - sampling_points
    x_x0, y_y0, z_z0 = torch.unbind(distance_to_bottom_left, dim=-1)
    x1_x, y1_y, z1_z = torch.unbind(distance_to_top_right, dim=-1)
    weights_x = torch.concat([x1_x, x_x0, x1_x, x_x0,
                              x1_x, x_x0, x1_x, x_x0], dim=-1)
    weights_y = torch.concat([y1_y, y1_y, y_y0, y_y0,
                              y1_y, y1_y, y_y0, y_y0], dim=-1)
    weights_z = torch.concat([z1_z, z1_z, z1_z, z1_z,
                              z_z0, z_z0, z_z0, z_z0], dim=-1)

    weights = weights_x * weights_y * weights_z
    weights = weights.unsqueeze(-1)

    interpolated_values = weights * content

    return sum(torch.split(interpolated_values, [num_points] * 8, dim=-2))

