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