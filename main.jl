using Random
using Statistics
using Plots
using LinearAlgebra

include("config.jl")
include("metrics.jl")
include("objective.jl")
include("kernel.jl")
include("gp.jl")
include("acquisition.jl")
include("optimizer.jl")


rng_noise = Xoshiro(seed)
rng_search = Xoshiro(seed + 1)

x_grid = collect(range(x_min, x_max; length=grid_size))
initial_data = rand(rng_search,initial_points)
x_observed =  x_min .+ initial_data.*(x_max-x_min)
y_observed = Vector{Float64}()

y_true = Vector{Float64}()
for i in x_observed
    push!(y_observed,measurment(i,noise_sko,rng_noise))
end

used_budget = length(y_observed)
remaining_budget = budget - used_budget


for x in x_grid
    push!(y_true,f(x))
end

opt = optimize(x_grid,budget,ucb_kappa,noise_sko,rng_noise, 
    x_observed,y_observed,kernel_length,kernel_sko,noise_dispersion,jitter,"EI")

predicted_cort = predict_data(opt.final_model, x_grid)
mu = predicted_cort.mu
interval = 1.96 .* sqrt.(predicted_cort.vect_disp_function)

if noise_dispersion == 0
    b_koef = maximum(opt.y_coords)
else
    pr = predict_data(opt.final_model, opt.x_coords)
    b_koef = maximum(pr.mu)
end

scores = EI(mu, predicted_cort.vect_disp_function, b_koef)
ind_max = argmax(scores)
x_next = x_grid[ind_max]

p1 = plot(x_grid, y_true; label="Настоящая функция")
scatter!(p1, opt.x_coords, opt.y_coords; label="Измерения")
plot!(p1, x_grid, mu;
    ribbon=interval, label="Прогноз", fillalpha=0.2)
vline!(p1, [x_next];
    label="Следующая точка по EI", linestyle=:dash)

p2 = plot(x_grid, scores; label="EI", xlabel="x", ylabel="Оценка")
scatter!(p2, [x_next], [scores[ind_max]]; label="Максимум EI")

p = plot(p1, p2; layout=(2, 1))
display(p)
savefig(p, "ei_search.png")

used_budget = length(opt.y_coords)
remaining_budget = budget - used_budget
