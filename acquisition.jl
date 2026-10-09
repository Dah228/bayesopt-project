using Distributions

function UCB(avg_vect, dispers_vect, ucb_koef)
    if length(avg_vect) != length(dispers_vect) || ucb_koef < 0 
        throw(ArgumentError("длины переданных векторов не совпадают или коэффициент не является положительным!"))
    end

    sko_vect = sqrt.(dispers_vect)
    apprise_vect = Vector{Float64}(undef,length(avg_vect))
    for i in eachindex(avg_vect)
        apprise_vect[i] = avg_vect[i]+ucb_koef*sko_vect[i]
    end
    return apprise_vect
end


function EI(avg_vect, dispers_vect,b_koef)
    if length(avg_vect)!=length(dispers_vect)
        throw(ArgumentError("длины массивов не совпадают"))

    end
    s = sqrt.(dispers_vect)
    d = avg_vect.-b_koef
    Ei = Vector{Float64}(undef,length(avg_vect)) 
    norm = Normal(0,1)
    for i in eachindex(avg_vect)
        if s[i] <= 1e-10
            ei = max(d[i],0)
            Ei[i] = ei
        else
            z = d[i]/s[i]
            ei = d[i]*cdf(norm, z)+s[i]*pdf(norm, z)
            Ei[i] = ei
        end
    end
    return Ei
end
