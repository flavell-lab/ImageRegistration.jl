function crop_pad_bkg_subtract(matrix, crop_size)
    """
    Compute the center of mass of the matrix, then crop and/or pad the matrix to the crop size,
    centered at the center of mass.

    Parameters:
    - matrix: 3D array.
    - crop_size: Tuple of 3 integers specifying the desired size after cropping/padding.
    """
    # Background subtract
    matrix = matrix .- floor(Int32, median(matrix))

    # Calculate the center of mass
    function center_of_mass(matrix)
        total = sum(matrix)
        indices = CartesianIndices(matrix)
        com = map(1:ndims(matrix)) do i
            sum(map(idx -> idx[i] * matrix[idx], indices)) / total
        end
        return round.(Int, com)
    end
    
    com = center_of_mass(matrix)

    # Initialize cropped/padded matrix
    cropped_padded = zeros(crop_size...)

    # Calculate start and end indices for cropping and padding
    start_idx = max.(1, com .- crop_size .÷ 2)
    end_idx = start_idx .+ crop_size .- 1

    crop_pad_start_idx = ones(Int, ndims(matrix))
    crop_pad_end_idx = copy(collect(crop_size))

    for dim = 1:ndims(matrix)
        if end_idx[dim] > size(matrix, dim)
            end_idx[dim] = size(matrix, dim)
            start_idx[dim] = max(1, end_idx[dim] - crop_size[dim] + 1)
        end
        if size(matrix, dim) < crop_size[dim]
            crop_pad_start_idx[dim] = crop_pad_start_idx[dim] + (crop_size[dim] - size(matrix, dim)) ÷ 2
            crop_pad_end_idx[dim] = crop_pad_end_idx[dim] - (crop_size[dim] - size(matrix, dim) + 1) ÷ 2
        end
    end

    # Calculate slices for the original matrix
    orig_slices = [start_idx[d]:end_idx[d] for d in 1:ndims(matrix)]

    # Calculate slices for the cropped_padded matrix
    padded_slices = [crop_pad_start_idx[d]:crop_pad_end_idx[d] for d in 1:ndims(cropped_padded)]

    # Use array slicing for efficient cropping and padding
    cropped_padded[padded_slices...] = matrix[orig_slices...]

    return floor.(UInt16, clamp.(cropped_padded, typemin(UInt16), typemax(UInt16))), orig_slices, padded_slices
end