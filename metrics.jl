function mse(data, predicted)
    m = 0.0
    len = length(data)

    for n in 1:len
        m += (data[n] - predicted[n])^2
    end

    return m / len
end