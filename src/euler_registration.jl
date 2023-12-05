function translate_z(image::Array, shift::Int, fill_value::Real)
    # Get the size of the image
    dims = size(image)

    # Create a new array to hold the translated image
    translated_image = fill(fill_value, dims)

    # Decide on the range of the original and the new image based on the shift
    if shift > 0
        original_range = 1:dims[3]-shift
        new_range = shift+1:dims[3]
    else
        original_range = -shift+1:dims[3]
        new_range = 1:dims[3]+shift
    end

    # Copy the translated image data
    translated_image[:, :, new_range] = image[:, :, original_range]

    return translated_image
end

function euler_register!(param_path, param, fixed_image, moving_image, memory_dict)
    downsample_factor = param["euler_downsample_factor"]
    batch_size = param["euler_batch_size"]
    euler_gpu = pyimport("euler_gpu")
    torch = pyimport("torch")
    device = torch.device("cuda:0")
    # start_time = time()
    outcomes = Dict()

    resized_moving_image_xyz = moving_image
    resized_fixed_image_xyz = fixed_image

    outcomes["registered_image_xyz_gncc_0"] = euler_gpu.calculate_gncc(
            resized_fixed_image_xyz,
            resized_moving_image_xyz
    )

    best_gncc = outcomes["registered_image_xyz_gncc_0"]
    best_img = resized_moving_image_xyz

    downsampled_resized_fixed_image_xy = Float32.(euler_gpu.max_intensity_projection_and_downsample(
                            resized_fixed_image_xyz,
                            downsample_factor,
                            projection_axis=2))

    downsampled_resized_moving_image_xy = Float32.(euler_gpu.max_intensity_projection_and_downsample(
                            resized_moving_image_xyz,
                            downsample_factor,
                            projection_axis=2))

    # println("Time taken to downsample: $(time() - start_time) seconds")

    if "xy" in keys(memory_dict)
        memory_dict_xy = memory_dict["xy"]

        memory_dict_xy["fixed_images_repeated"] = torch.tensor(
            downsampled_resized_fixed_image_xy,
            device=device,
            dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        memory_dict_xy["moving_images_repeated"] = torch.tensor(
            downsampled_resized_moving_image_xy,
            device=device,
            dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else
        memory_dict_xy = euler_gpu.initialize(
            downsampled_resized_fixed_image_xy,
            downsampled_resized_moving_image_xy,
            param["euler_x_translation_range_1"],
            param["euler_y_translation_range_1"],
            param["euler_theta_rotation_range_xy"],
            batch_size,
            device
        )

        memory_dict["xy"] = memory_dict_xy
    end

    # println("Time taken to initialize: $(time() - start_time) seconds")

    best_score_xy, best_transformation_xy = euler_gpu.grid_search(memory_dict_xy)

    # println("Time taken to grid search: $(time() - start_time) seconds")

    outcomes["registered_image_xyz_gncc_xy"] = best_score_xy.item()
    outcomes["best_transformation_xy"] = [best_param_xy.cpu().numpy()[1] for best_param_xy in best_transformation_xy]

    if "_xy" in keys(memory_dict)
        _memory_dict_xy = memory_dict["_xy"]
    else
        _memory_dict_xy = euler_gpu.initialize(
            zeros(Float32, size(resized_moving_image_xyz[:,:,1])),
            zeros(Float32, size(resized_fixed_image_xyz[:,:,1])),
            zeros(Float32, size(resized_moving_image_xyz, 3)),
            zeros(Float32, size(resized_moving_image_xyz, 3)),
            zeros(Float32, size(resized_moving_image_xyz, 3)),
            size(resized_moving_image_xyz, 3),
            device
        )

        memory_dict["_xy"] = _memory_dict_xy
    end

    transformed_moving_image_xyz = euler_gpu.transform_image_3d(
        resized_moving_image_xyz,
        _memory_dict_xy,
        best_transformation_xy,
        device,
        2
    )

    outcomes["registered_image_xyz_gncc_1"] = euler_gpu.calculate_gncc(
            resized_fixed_image_xyz,
            transformed_moving_image_xyz
    )

    # println("Time taken to transform: $(time() - start_time) seconds")

    downsampled_resized_fixed_image_xz = Float32.(euler_gpu.max_intensity_projection_and_downsample(
        resized_fixed_image_xyz,
        downsample_factor,
        projection_axis=1))

    downsampled_resized_moving_image_xz = Float32.(euler_gpu.max_intensity_projection_and_downsample(
            transformed_moving_image_xyz,
            downsample_factor,
            projection_axis=1))

    if "xz" in keys(memory_dict)
        memory_dict_xz = memory_dict["xz"]

        memory_dict_xz["fixed_images_repeated"] = torch.tensor(
            downsampled_resized_fixed_image_xz,
            device=device,
            dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        memory_dict_xz["moving_images_repeated"] = torch.tensor(
            downsampled_resized_moving_image_xz,
            device=device,
            dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else
        memory_dict_xz = euler_gpu.initialize(
            downsampled_resized_fixed_image_xz,
            downsampled_resized_moving_image_xz,
            param["euler_x_translation_range_2"],
            param["euler_z_translation_range_1"],
            param["euler_theta_rotation_range_xz"],
            batch_size,
            device
        )

        memory_dict["xz"] = memory_dict_xz
    end

    # println("Time taken to initialize: $(time() - start_time) seconds")

    best_score_xz, best_transformation_xz = euler_gpu.grid_search(memory_dict_xz)
    outcomes["registered_image_xyz_gncc_xz"] = best_score_xz.item()
    outcomes["best_transformation_xz"] = [best_param_xz.cpu().numpy()[1] for best_param_xz in best_transformation_xz]
    # println("Time taken to grid search: $(time() - start_time) seconds")

    if "_xz" in keys(memory_dict)
        _memory_dict_xz = memory_dict["_xz"]
    else
        _memory_dict_xz = euler_gpu.initialize(
            zeros(Float32, size(resized_moving_image_xyz[:,1,:])),
            zeros(Float32, size(resized_fixed_image_xyz[:,1,:])),
            zeros(Float32, size(resized_moving_image_xyz, 2)),
            zeros(Float32, size(resized_moving_image_xyz, 2)),
            zeros(Float32, size(resized_moving_image_xyz, 2)),
            size(resized_moving_image_xyz, 2),
            device
        )

        memory_dict["_xz"] = _memory_dict_xz
    end

    z_transform = best_transformation_xz[2].cpu().numpy()[1]

    # println(z_transform)

    transformed_moving_image_xyz = translate_z(transformed_moving_image_xyz, Int(round(z_transform * size(resized_moving_image_xyz, 3) / 2)), 0.0)

    return outcomes, transformed_moving_image_xyz
end

function euler_transform_roi(roi_image, parameters_xy, parameters_xz, memory_dict; interpolation="nearest")
    euler_gpu = pyimport("euler_gpu")
    torch = pyimport("torch")
    device = torch.device("cuda:0")
    if "_xy" in keys(memory_dict)
        _memory_dict_xy = memory_dict["_xy"]
    else
        _memory_dict_xy = euler_gpu.initialize(
            zeros(Float32, size(roi_image[:,:,1])),
            zeros(Float32, size(roi_image[:,:,1])),
            zeros(Float32, size(roi_image, 3)),
            zeros(Float32, size(roi_image, 3)),
            zeros(Float32, size(roi_image, 3)),
            size(roi_image, 3),
            device
        )

        memory_dict["_xy"] = _memory_dict_xy
    end

    best_transformation_xy = torch.tensor(
        parameters_xy,
        device=device,
        dtype=torch.float32)

    transformed_roi_image = euler_gpu.transform_image_3d(
        roi_image,
        _memory_dict_xy,
        best_transformation_xy,
        device,
        2,
        interpolation=interpolation
    )
    transformed_roi_image = translate_z(transformed_roi_image, Int(round(parameters_xz[2] * size(roi_image, 3) / 2)), 0.0)
    return transformed_roi_image
end