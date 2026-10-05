using Random
using Statistics

position = range(0, 10,length = 101)
pos_arr = collect(position)

y_true = 3 .* pos_arr .+ 2

rng = Xoshiro(42) #noise
rnd = Xoshiro(43) #index

samples = randn(rng, 101)
noise = samples.*0.5

y_observed = y_true.+noise


# using Plots


# p = plot(position,y_true)
# scatter!(p, position,y_observed, label="data")
# display(p)
# savefig(p, "graph_0_5.png")

function predict(data, b, a)
    predict = a.*data.+b
    return predict
end


function mse(data,predicted)
    mse = 0
    len = length(data)
    for n in 1:len
        mse += (data[n] - predicted[n])^2
    end
    mse /= len
    return mse
end



index_mas = randperm(rnd, length(pos_arr))
tr= round(Int, length(pos_arr)*0.6)
va = round(Int, length(pos_arr)*0.8)

train_index = index_mas[1:tr]
valid_index = index_mas[tr+1:va]
test_index = index_mas[va+1:end]

function data_form(massive_x_sorted, massive_y_sorted, index_massive)
    massive_x_rand = []
    massive_y_rand = []
    for n in index_massive
        push!(massive_x_rand, massive_x_sorted[n])
        push!(massive_y_rand,massive_y_sorted[n])
    end
    return [massive_x_rand, massive_y_rand]
end

trin_data = data_form(pos_arr, y_observed, train_index)
valid_data = data_form(pos_arr, y_observed, valid_index)
test_data = data_form(pos_arr, y_observed, test_index)




pred = predict(trin_data[1],2,3)
ms = mse(trin_data[2],pred)

# println(pred)
# println(ms)
println(trin_data)
println(valid_data)
println(test_data)
println(ms)
