using Random
using Statistics

position = range(0, 10,length = 101)
pos_arr = collect(position) #массив отсортированных х
pos_arr_z = -1 .+pos_arr./5
y_true = 3 .* pos_arr .+ 2 #массив идеальных у

rng = Xoshiro(42) #noise
rnd = Xoshiro(43) #index

samples = randn(rng, 101) #рандомный шум
noise = samples.*0.5 #noise with 0.5 std

y_observed = y_true.+noise #real y - observed in experience

function matrix_input_ultimate(massive_data)
    return hcat(ones(length(massive_data)), massive_data)
end


function matrix_deg_dep(massive_data_z, deg)
    len = length(massive_data_z)
    matrix_ans = Matrix{Float64}(undef,len,deg+1)
    for i in 1:deg+1
        matrix_ans[:,i] = massive_data_z.^(i-1)
    end
    return matrix_ans
end


#using b and a as linear coefficient predict data
function predict(data, koef_v)
    deg = length(koef_v) -1 
    predicted = matrix_deg_dep(data,deg)*koef_v
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

train_data = data_form(pos_arr_z, y_observed, train_index)
valid_data = data_form(pos_arr_z, y_observed, valid_index)
test_data = data_form(pos_arr_z, y_observed, test_index)


# avg_train = mean(train_data[2])

#diviation from the average
# function study_avg(train_data_massive,avg)
#     mistake = 0
#     len = length(train_data_massive)
#     for i in 1:len
#         mistake += (train_data_massive[i]-avg)^2
#     end
#     mistake/=len
#     return mistake
# end

# pred = predict(train_data[1],2,3)
# ms = mse(train_data[2],pred)
# ms_av= study_avg(train_data[2],avg_train)
# ms_val = study_avg(valid_data[2],avg_train)





#заполнили первый столбец 1
# train_data_x = matrix_input_ultimate(train_data[1])
train_data_z = matrix_deg_dep(train_data[1], 1)
#получили в переменной theta коэффициенты [b,a] (вектор длины n)
theta = train_data_z \ train_data[2]
#по тренировочным входам предсказали результат
mnk_predict_data = predict(train_data[1],theta)
mnk_predict_valid_data = predict(valid_data[1],theta)


train_data_z_2 = matrix_deg_dep(train_data[1], 2)
#получили в переменной theta коэффициенты [b,a] (вектор длины n)
theta_2 = train_data_z_2 \ train_data[2]
#по тренировочным входам предсказали результат
mnk_predict_data_2 = predict(train_data[1],theta_2)
mnk_predict_valid_data_2 = predict(valid_data[1],theta_2)

train_data_z_3 = matrix_deg_dep(train_data[1], 3)
#получили в переменной theta коэффициенты [b,a] (вектор длины n)
theta_3 = train_data_z_3 \ train_data[2]
#по тренировочным входам предсказали результат
mnk_predict_data_3 = predict(train_data[1],theta_3)
mnk_predict_valid_data_3 = predict(valid_data[1],theta_3)

train_data_z_5 = matrix_deg_dep(train_data[1], 5)
#получили в переменной theta коэффициенты [b,a] (вектор длины n)
theta_5 = train_data_z_5 \ train_data[2]
#по тренировочным входам предсказали результат
mnk_predict_data_5 = predict(train_data[1],theta_5)
mnk_predict_valid_data_5 = predict(valid_data[1],theta_5)

train_data_z_10 = matrix_deg_dep(train_data[1], 10)
#получили в переменной theta коэффициенты [b,a] (вектор длины n)
theta_10 = train_data_z_10 \ train_data[2]
#по тренировочным входам предсказали результат
mnk_predict_data_10 = predict(train_data[1],theta_10)
mnk_predict_valid_data_10 = predict(valid_data[1],theta_10)

println(theta)
println("train mse ",mse(train_data[2], mnk_predict_data))
println("valid mse ", mse(valid_data[2],mnk_predict_valid_data))


using Plots
p = plot(pos_arr_z,y_true)
# p = plot!(train_data[1],mnk_predict_data)
# p = plot!(train_data[1], mnk_predict_data_2)
p = plot!(pos_arr_z,predict(pos_arr_z,theta))
scatter!(p,valid_data[1], valid_data[2], label="data_valid_r")
scatter!(p, valid_data[1], mnk_predict_valid_data, label="data_valid_p")
display(p)
savefig(p, "graph_1_test+valid.png")



# println(pred)
# println(ms)
# println(train_data)
# println(valid_data)
# println(test_data)
# println(ms)
# println(ms_av)
# println(ms_val)

