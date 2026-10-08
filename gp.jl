function gauss_process(x_observed,y_observed, kernel_length, kernel_sko, noise_dispersion, jitter)
    if length(x_observed) != length(y_observed) || length(x_observed)==0 || length(y_observed) ==0 || kernel_length <= 0 || kernel_sko <= 0 || noise_dispersion < 0 || jitter<= 0
        # println("GAUSS_FUNCTION_ERROR")
        throw(ArgumentError("Массивы должны быть непустыми и одинаковой длины; параметры ядра и jitter — положительными, дисперсия шума — неотрицательной"))
    end
    matr = matrix_kor(x_observed,x_observed,kernel_length,kernel_sko)
    identity_matrix = Matrix{Float64}(I, length(x_observed), length(x_observed))
    C_matr = matr + (noise_dispersion + jitter)*identity_matrix
    try
        chol = cholesky(Symmetric(C_matr))
        alpha = chol \ y_observed
        return (
                x_observed = copy(x_observed),
                kernel_length = kernel_length,
                kernel_sko = kernel_sko,
                noise_dispersion = noise_dispersion,
                jitter = jitter,
                chol = chol,
                alpha = alpha
            )
    catch e
        if isa(e, PosDefException)
            throw(ArgumentError("Матрица не является положительно определённой!"))
        else
            rethrow(e) # Если ошибка другая (например, BoundsError), пробрасываем её дальше
        end
    end

end

function predict_data(model_gauss_proc_res, coords_array_x_for_prediction)
    k_star = matrix_kor(model_gauss_proc_res.x_observed,coords_array_x_for_prediction,model_gauss_proc_res.kernel_length, model_gauss_proc_res.kernel_sko)
    mu = transpose(k_star)*model_gauss_proc_res.alpha
    V = model_gauss_proc_res.chol.L \ k_star
    vect_disp_function = Vector{Float64}(undef, length(coords_array_x_for_prediction))
    for j in 1:size(V)[2]
        sums = 0.0
        for i in 1:size(V)[1]
            sums += (V[i,j])^2 
        end
        vect_disp_function[j] = (model_gauss_proc_res.kernel_sko)^2 - sums
    end
    vect_disp_observed = copy(vect_disp_function).+model_gauss_proc_res.noise_dispersion

    return (
            mu = mu,
            vect_disp_function = vect_disp_function,
            vect_disp_observed = vect_disp_observed
    )
end