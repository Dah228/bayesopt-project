using Random
using Statistics

position = range(0, 10,length = 101)
pos_arr = collect(position) #массив отсортированных х
y_true = 3 .* pos_arr .+ 2 #массив идеальных у

rng = Xoshiro(42) #noise
rnd = Xoshiro(43) #index

samples = randn(rng, 101) #рандомный шум
noise = samples.*0.5 #noise with 0.5 std

y_observed = y_true.+noise #real y - observed in experience


#using b and a as linear coefficient predict data
function predict(data, b, a)
    predicted = a.*data.+b
    return predicted
end

#calculates mistake between observed and predicted data
function mse(data,predicted)
    m = 0
    len = length(data)
    for n in 1:len
        m += (data[n] - predicted[n])^2
    end
    m /= len
    return m
end



index_mas = randperm(rnd, length(pos_arr))
tr= round(Int, length(pos_arr)*0.6)
va = round(Int, length(pos_arr)*0.8)

train_index = index_mas[1:tr]
valid_index = index_mas[tr+1:va]
test_index = index_mas[va+1:end]

function data_form(massive_x_sorted, massive_y_sorted, index_massive)
    len = length(index_massive)
    massive_x_rand = Vector{Float64}(undef, len)
    massive_y_rand = Vector{Float64}(undef, len)
    massive_x_rand = massive_x_sorted[index_massive]
    massive_y_rand = massive_y_sorted[index_massive]
    return [massive_x_rand, massive_y_rand]
end

train_data = data_form(pos_arr, y_observed, train_index)
valid_data = data_form(pos_arr, y_observed, valid_index)
test_data = data_form(pos_arr, y_observed, test_index)
avg_train = mean(train_data[2])

#diviation from the average
function study_avg(train_data_massive,avg)
    mistake = 0
    len = length(train_data_massive)
    for i in 1:len
        mistake += (train_data_massive[i]-avg)^2
    end
    mistake/=len
    return mistake
end

pred = predict(train_data[1],2,3)
ms = mse(train_data[2],pred)
ms_av= study_avg(train_data[2],avg_train)
ms_val = study_avg(valid_data[2],avg_train)

function matrix_input_ultimate(massive_data)
    return hcat(ones(length(massive_data)), massive_data)
end

#заполнили первый столбец 1
train_data_x = matrix_input_ultimate(train_data[1])

#получили в переменной theta коэффициенты [b,a]
theta = train_data_x \ train_data[2]

#по тренировочным входам предсказали результат
mnk_predict_data = predict(train_data[1],theta[1],theta[2])

println("a = ", theta[2], "\n", "b = ", theta[1])
println(mse(train_data[2], mnk_predict_data))

# println(pred)
# println(ms)
# println(train_data)
# println(valid_data)
# println(test_data)
# println(ms)
# println(ms_av)
# println(ms_val)


# using Plots


# p = plot(position,y_true)
# scatter!(p, position,y_observed, label="data")
# display(p)
# savefig(p, "graph_0_5.png")

