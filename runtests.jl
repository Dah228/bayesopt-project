using Test
using LinearAlgebra

include("gp.jl")
include("kernel.jl")


x_obser = [0.0,1.0,2.0]
ans = [0.0,1.0,0.0]
kernel_len = 1.0
func_sko = 1.0
noise_disp = 0.01
jitters = 1e-8

rbf_matr = matrix_kor(x_obser, x_obser, kernel_len, func_sko)
@test size(rbf_matr) == (3, 3)
@test transpose(rbf_matr) ≈ rbf_matr
for i in 1:size(rbf_matr)[1]
    @test rbf_matr[i,i] ≈ 1.0
end

@test size(matrix_kor(x_obser,[1.0,4.0], kernel_len, func_sko)) == (3,2)
mode = gauss_process(x_obser,ans,kernel_len,func_sko,noise_disp,jitters)
@test length(mode.alpha) == 3
@test all(isfinite, mode.alpha)
@test rbf_matr[1, 2] > rbf_matr[1, 3]

c = rbf_matr + (0.01+1e-8)*I
@test c*mode.alpha ≈ ans
predict = predict_data(mode,[0.5,1.5])

@test length(predict.mu) == 2
@test length(predict.vect_disp_function) == 2 
@test length(predict.vect_disp_observed) == 2
@test all(isfinite, predict.mu)
@test all(isfinite, predict.vect_disp_function)
@test all(isfinite, predict.vect_disp_observed)
@test all(>=(0),predict.vect_disp_function)

@test all(<=(1),predict.vect_disp_function)
@test isapprox(predict.vect_disp_function.+0.01,predict.vect_disp_observed)





# println(c*mode.alpha)