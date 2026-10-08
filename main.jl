using Random
using Statistics
using Plots
using LinearAlgebra

include("config.jl")
include("metrics.jl")
include("objective.jl")
include("kernel.jl")
include("gp.jl")

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




println(matrix_kor(x_observed,x_observed,kernel_length,kernel_sko))

model = gauss_process(x_observed,y_observed,kernel_length,kernel_sko,noise_dispersion,jitter)
# matr = matrix_kor(x_observed,x_observed,kernel_length,kernel_sko)
# identity_matrix = Matrix{Float64}(I, length(x_observed), length(x_observed))
# C_matr = matr + (noise_dispersion + jitter)*identity_matrix

println(model.alpha)
predicted_cort =predict_data(model,x_grid)
mu = predicted_cort.mu
interval = 1.96.*sqrt.(predicted_cort.vect_disp_function)
# println(C_matr*model.alpha)
# println(y_observed)

p = plot(x_grid, y_true; label="Настоящая функция")
scatter!(p, x_observed, y_observed; label="Измерения")
plot!(p, x_grid, mu; ribbon=interval, label="Прогноз", fillalpha=0.2)

display(p)
savefig(p, "initial_observations.png")

