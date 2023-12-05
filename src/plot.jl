function overlay_matrices(matrix1, matrix2, dim; contrast_factor=1.0)
    """
    Create a plot that overlays the MIPs of two matrices along a specified dimension.

    Parameters:
    - matrix1, matrix2: 3D arrays.
    - dim: The dimension along which to compute and overlay the MIPs.
    - contrast_factor: Factor by which to divide the maximum intensity 
                       to increase the image contrast. Must be >= 1.
    """
    # Ensure the matrices are 3D
    if ndims(matrix1) != 3 || ndims(matrix2) != 3
        throw(ArgumentError("Input matrices must be 3D"))
    end
    
    # Ensure contrast_factor is valid
    if contrast_factor < 1
        throw(ArgumentError("contrast_factor must be >= 1"))
    end
    
    # Compute MIPs
    mip1 = maxprj(matrix1, dims=dim)
    mip2 = maxprj(matrix2, dims=dim)

    # Ensure MIPs are 2D
    if ndims(mip1) != 2 || ndims(mip2) != 2
        throw(ArgumentError("MIPs must be 2D"))
    end
    
    # Normalize MIPs to [0, 1] based on the adjusted maximum values
    mip1_norm = mip1 ./ (maximum(mip1) / contrast_factor)
    mip2_norm = mip2 ./ (maximum(mip2) / contrast_factor)
    
    # Ensure values are clipped to [0, 1] after adjusting contrast
    mip1_norm = clamp.(mip1_norm, 0, 1)
    mip2_norm = clamp.(mip2_norm, 0, 1)
    
    # Create an RGB image: R corresponds to mip1, G to mip2, B is kept 0
    rgb_image = RGB.(mip1_norm, mip2_norm, zeros(size(mip1_norm)))
    
    return rgb_image
end
