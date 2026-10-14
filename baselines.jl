function rand_finding(x_min,x_max,budget,noise_disp,rng_noise, rng_search, x_grid, kernel_length, kernel_sko, jitter)
    if check_param(x_min,x_max,budget,noise_disp)
        ans = Vector{Float64}()
        u = rand(rng_search, budget)
        coords = x_min .+ (x_max - x_min) .* u
        for i in 1:budget
            push!(ans,measurment(coords[i], sqrt(noise_disp),rng_noise))
        end

        model = gauss_process(coords,ans, kernel_length, kernel_sko, noise_disp, jitter)
        predicted = predict_data(model,x_grid)
        ind_max = argmax(predicted.mu)

        recommended_x = x_grid[ind_max]
        recommended_mean = predicted.mu[ind_max]
        recommended_disp = predicted.vect_disp_function[ind_max]

        return (
            final_model = model,
            x_coords = coords, 
            y_coords = ans,
            strategy = "Random",
            recommended_x = recommended_x,
            recommended_mean = recommended_mean,
            recommended_disp = recommended_disp              
            )
    end
    throw(ArgumentError("Введенные значения не корректны"))

end


function equable_finding(x_min,x_max,budget,noise_disp,rng_noise, x_grid, kernel_length, kernel_sko, jitter)
    if check_param(x_min,x_max,budget,noise_disp) && budget >= 2
        ans = Vector{Float64}()
        coords = collect(range(x_min, x_max; length=budget))
        for i in 1:budget
            push!(ans,measurment(coords[i], sqrt(noise_disp),rng_noise))
        end

        model = gauss_process(coords,ans, kernel_length, kernel_sko, noise_disp, jitter)
        predicted = predict_data(model,x_grid)
        ind_max = argmax(predicted.mu)

        recommended_x = x_grid[ind_max]
        recommended_mean = predicted.mu[ind_max]
        recommended_disp = predicted.vect_disp_function[ind_max]

        return (
            final_model = model,
            x_coords = coords, 
            y_coords = ans,
            strategy = "Grid",
            recommended_x = recommended_x,
            recommended_mean = recommended_mean,
            recommended_disp = recommended_disp              
            )

    end
    throw(ArgumentError("Введенные значения не корректны"))
end


function check_param(x_min,x_max,budget,noise_disp)
    if x_min >= x_max || budget <= 0 || !(budget isa Integer) || noise_disp < 0 || !all(isfinite, (x_min, x_max, noise_disp))
        return false
    else
        return true
    end
end