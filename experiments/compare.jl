using Random
using LinearAlgebra
using Plots
using Test

include("../config.jl")
include("../metrics.jl")
include("../objective.jl")
include("../kernel.jl")
include("../gp.jl")
include("../acquisition.jl")
include("../optimizer.jl")
include("../baselines.jl")

x_grid = collect(range(x_min, x_max; length=grid_size))

# Общие начальные данные для UCB и EI
rng_noise = Xoshiro(seed)
rng_search = Xoshiro(seed + 1)

initial_data = rand(rng_search, initial_points)
x_observed = x_min .+ (x_max - x_min) .* initial_data
y_observed = Float64[]

for x in x_observed
    push!(y_observed, measurment(x, noise_sko, rng_noise))
end

x_ucb = copy(x_observed)
y_ucb = copy(y_observed)
rng_noise_ucb = copy(rng_noise)

x_ei = copy(x_observed)
y_ei = copy(y_observed)
rng_noise_ei = copy(rng_noise)

ucb_res = optimize(
    x_grid, budget, ucb_kappa, noise_sko, rng_noise_ucb,
    x_ucb, y_ucb, kernel_length, kernel_sko,
    noise_dispersion, jitter, "UCB"
)

ei_res = optimize(
    x_grid, budget, ucb_kappa, noise_sko, rng_noise_ei,
    x_ei, y_ei, kernel_length, kernel_sko,
    noise_dispersion, jitter, "EI"
)

# Независимые генераторы для методов сравнения
rng_noise_random = Xoshiro(seed)
rng_search_random = Xoshiro(seed + 1)
rng_noise_grid = Xoshiro(seed)

random_res = rand_finding(
    x_min, x_max, budget, noise_dispersion,
    rng_noise_random, rng_search_random,
    x_grid, kernel_length, kernel_sko, jitter
)

grid_res = equable_finding(
    x_min, x_max, budget, noise_dispersion,
    rng_noise_grid, x_grid, kernel_length, kernel_sko, jitter
)

@testset "Сравнение методов" begin
    for result in (ucb_res, ei_res, random_res, grid_res)
        @test length(result.x_coords) == budget
        @test length(result.y_coords) == budget
    end

    @test length(x_observed) == initial_points
    @test length(y_observed) == initial_points

    for result in (ucb_res, ei_res)
        @test result.x_coords[1:initial_points] == x_observed
        @test result.y_coords[1:initial_points] == y_observed
    end

    @test ucb_res.strategy == "UCB"
    @test ei_res.strategy == "EI"
    @test random_res.strategy == "Random"
    @test grid_res.strategy == "Grid"
end


results = (ucb_res, ei_res, random_res, grid_res)
y_true = f.(x_grid)

# Прогнозы всех моделей
forecasts = [
    predict_data(result.final_model, x_grid)
    for result in results
]

intervals = [
    1.96 .* sqrt.(forecast.vect_disp_function)
    for forecast in forecasts
]

# Одинаковые границы оси y для всех графиков
y_min = minimum(y_true)
y_max = maximum(y_true)

for i in eachindex(results)
    global y_min = min(
        y_min,
        minimum(results[i].y_coords),
        minimum(forecasts[i].mu .- intervals[i])
    )
    global y_max = max(
        y_max,
        maximum(results[i].y_coords),
        maximum(forecasts[i].mu .+ intervals[i])
    )
end

padding = 0.05 * max(y_max - y_min, 1.0)
panels = []

for i in eachindex(results)
    result = results[i]
    forecast = forecasts[i]

    println(
        "$(result.strategy): ",
        "измерений = $(length(result.y_coords)), ",
        "рекомендация x = $(result.recommended_x), ",
        "прогноз = $(result.recommended_mean), ",
        "истинное значение = $(f(result.recommended_x))"
    )

    panel = plot(
        x_grid, y_true;
        label="Настоящая функция",
        title=result.strategy,
        xlabel="x",
        ylabel="y",
        xlims=(x_min, x_max),
        ylims=(y_min - padding, y_max + padding),
        legend=:bottomleft
    )

    plot!(
        panel, x_grid, forecast.mu;
        ribbon=intervals[i],
        fillalpha=0.2,
        label="Прогноз"
    )

    scatter!(
        panel, result.x_coords, result.y_coords;
        label="Измерения",
        markersize=3
    )

    vline!(
        panel, [result.recommended_x];
        label="Рекомендация",
        linestyle=:dash,
        color=:red
    )

    push!(panels, panel)
end

comparison_plot = plot(
    panels...;
    layout=(2, 2),
    size=(1200, 800)
)

display(comparison_plot)
savefig(
    comparison_plot,
    joinpath(@__DIR__, "methods_comparison.png")
)