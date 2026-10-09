function optimize(x_grid,budget,ucb_kappa,noise_sko,rng_noise, 
    x_observed,y_observed,kernel_length,kernel_sko,noise_dispersion,jitter,strategy)
    acquisition_history = Vector{Float64}()
    if !(budget isa Integer) || budget < length(y_observed) || (strategy != "UCB" && strategy != "EI")
        throw(ArgumentError("БЮДЖЕТ ЭТО ПОЛОЖИТЕЛЬНОЕ ЦЕЛОЕ ЧИСЛО"))
    end
    initial_points = length(y_observed)
    scores = Vector{Float64}()

    while budget > length(y_observed)

        

        model = gauss_process(x_observed,y_observed,kernel_length,kernel_sko,noise_dispersion,jitter)
       
        predicted_cort =predict_data(model,x_grid)

        if strategy == "UCB"
            mu = predicted_cort.mu
            ucb = UCB(mu,predicted_cort.vect_disp_function,ucb_kappa)
            scores = ucb           




        elseif strategy == "EI"
            if noise_dispersion == 0
                b_koef = maximum(y_observed)
            else
                pr = predict_data(model, x_observed)
                b_koef = maximum(pr.mu)
            end

            scores = EI(
                predicted_cort.mu,
                predicted_cort.vect_disp_function,
                b_koef
            )

        end

    
        ind_max = argmax(scores)
        x_new = x_grid[ind_max]
        scores_ind_max = scores[ind_max]
        y_new = measurment(x_new,noise_sko,rng_noise)

        push!(x_observed, x_new)
        push!(acquisition_history, scores_ind_max)
        push!(y_observed,y_new)

    

        println("------------------------------------")
        println(length(y_observed)-initial_points)
        println(length(y_observed))
        println(x_new)
        println(y_new)
        println("------------------------------------")
        
    end
    model = gauss_process(x_observed, y_observed,
    kernel_length, kernel_sko, noise_dispersion, jitter)    


    return (
        final_model = model,
        x_coords = x_observed,
        y_coords = y_observed,
        acquisition_history = acquisition_history,
        strategy = strategy
    )

end