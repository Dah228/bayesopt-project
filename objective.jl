using Random

function f(x)
    return sin(x)+ 0.5*sin(3*x)    
end

function measurment(x, sko_noise, rng)
    return f(x) + randn(rng)*sko_noise    
end