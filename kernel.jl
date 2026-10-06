function RBF_kernel(x1,x2,len_ker,sko_ker)
    disp_ker = sko_ker^2
    return disp_ker*exp(-((x1-x2)^2)/(2*len_ker^2))
end


function matrix_kor(A,B,len_ker,sko_ker)
    len_A = length(A)
    len_B = length(B)
    matr = Matrix{Float64}(undef,len_A,len_B)
    for i in 1:len_A
        for j in 1:len_B
            matr[i,j]=RBF_kernel(A[i],B[j],len_ker,sko_ker)
        end
    end
    return matr
end