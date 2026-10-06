using Random
using Statistics

position = range(0, 10,length = 101)


pos_arr = collect(position) #массив отсортированных х
pos_arr_z = -1 .+pos_arr./5
y_true = 3 .* pos_arr .+ 2 #массив идеальных у

rng = Xoshiro(42) #noise
rnd = Xoshiro(43) #index

samples = randn(rng, 101) #рандомный шум
noise = samples.*2 #noise with 0.5 std
arr = fill(Float64(5), 101)
mu_arr_result = arr .+noise


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

using Printf
using Random
using Statistics
using Plots

mu = 5
sko = 2
dispersion = sko^2
observe_numb = 100
seed = 42


mu_0 = 0 #aprior average
sko_0 = 3 #aprior sko
dispersion_0 = sko_0^2 #aprior dispersion


arr = fill(Float64(mu),observe_numb)
rnd = Xoshiro(seed)
noise = randn(rnd, observe_numb).*sko

mu_arr_result = arr .+noise


function analitic_baes_update(arr_with_noise, aprior_avg, aprior_sko, noise_dispersion)
    n = length(arr_with_noise)
    aprior_disp = aprior_sko^2
    sum_observed = sum(arr_with_noise)
    aposterior_disper = 1/((1/aprior_disp)+(n/noise_dispersion))
    aposterior_average = aposterior_disper*((aprior_avg/aprior_disp)+(sum_observed/noise_dispersion))
    return aposterior_average, aposterior_disper
end

slice_1 = mu_arr_result[1:1]
# mean_1 = mean(slice_1)
# sko_1 = std(slice_1)
aposterior_average_1 = analitic_baes_update(slice_1,mu_0,sko_0, dispersion)[1]
aposterior_dispersion_1 = analitic_baes_update(slice_1,mu_0,sko_0, dispersion)[2]
boarder_1 = [aposterior_average_1 - 1.96*(sqrt(aposterior_dispersion_1)),aposterior_average_1 + 1.96*(sqrt(aposterior_dispersion_1))]


slice_5 = mu_arr_result[1:5]
# mean_5 = mean(slice_5)
# sko_5 = std(slice_5)
aposterior_average_5 = analitic_baes_update(slice_5,mu_0,sko_0, dispersion)[1]
aposterior_dispersion_5 = analitic_baes_update(slice_5,mu_0,sko_0, dispersion)[2]
boarder_5 = [aposterior_average_5 - 1.96*(sqrt(aposterior_dispersion_5)),aposterior_average_5 + 1.96*(sqrt(aposterior_dispersion_5))]


slice_20 = mu_arr_result[1:20]
# mean_20 = mean(slice_20)
# sko_20 = std(slice_20)
aposterior_average_20 = analitic_baes_update(slice_20,mu_0,sko_0, dispersion)[1]
aposterior_dispersion_20 = analitic_baes_update(slice_20,mu_0,sko_0, dispersion)[2]
boarder_20 = [aposterior_average_20 - 1.96*(sqrt(aposterior_dispersion_20)),aposterior_average_20 + 1.96*(sqrt(aposterior_dispersion_20))]

slice_100 = mu_arr_result[1:100]
# mean_100 = mean(slice_100)
# sko_100 = std(slice_100)
aposterior_average_100 = analitic_baes_update(slice_100,mu_0,sko_0, dispersion)[1]
aposterior_dispersion_100 = analitic_baes_update(slice_100,mu_0,sko_0, dispersion)[2]
boarder_100 = [aposterior_average_100 - 1.96*(sqrt(aposterior_dispersion_100)),aposterior_average_100 + 1.96*(sqrt(aposterior_dispersion_100))]


# println(mean_1, " ", aposterior_average_1, " ", aposterior_dispersion_1, " ", boarder_1)
# println(mean_5, " ", aposterior_average_5, " ", aposterior_dispersion_5, " ", boarder_5)
# println(mean_20, " ", aposterior_average_20, " ", aposterior_dispersion_20, " ", boarder_20)
# println(mean_100, " ", aposterior_average_100, " ", aposterior_dispersion_100, " ", boarder_100)


net_number_t = range(-10, 12, length = 1000)

function count_density(arr_of_numb, m_average, s_sko)
    arr_of_density = Vector{Float64}()
    for t in arr_of_numb
        n = (1/(s_sko*sqrt(2*pi)))*exp((-((t-m_average)^2))/(2*s_sko^2))
        push!(arr_of_density,n)
    end
    return arr_of_density
end

p = plot(net_number_t, count_density(net_number_t,mu_0,sko_0))
plot!(net_number_t, count_density(net_number_t,aposterior_average_1,sqrt(aposterior_dispersion_1)))
plot!(net_number_t, count_density(net_number_t,aposterior_average_5,sqrt(aposterior_dispersion_5)))
plot!(net_number_t, count_density(net_number_t,aposterior_average_20,sqrt(aposterior_dispersion_20)))
plot!(net_number_t, count_density(net_number_t,aposterior_average_100,sqrt(aposterior_dispersion_100)))
vline!([5], label="Настоящее среднее (μ=5)", linestyle=:dash, color=:red)


display(p)
savefig(p, "task_4_8(1).png")



# @printf("%d %f %f\n", length(mu_arr_result), mean(mu_arr_result), std(mu_arr_result))
# println(analitic_baes_update(mu_arr_result,mu_0,sko_0,dispersion))
# println(analitic_baes_update(6,0,3,4))