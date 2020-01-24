module QGA
export minimize

using Printf
#using StatsBase
using Random
using Statistics
using LinearAlgebra

function ent(t, f, fmin, w)
    w .= exp2.(t.*(fmin.-f))
    z = sum(w)
    w .*= t.*(fmin.-f)
    log2(z)-sum(w)/z
end

function selection_entropy!(w, k, t, f, fmin)
    if ent(t, f, fmin, w) > k
        while true
            t = t*1.001
            if ent(t, f, fmin, w) < k
                return t
            end
#             if t > 1e15
#                 @warn "probably infinite loop" f
#             end
        end
    else
        smax = log2(length(f))
        while true
            t = t/1.001
            s = ent(t, f, fmin, w)
            if s > k
                return t
            elseif s >= smax - 1e-2
                @warn "reached maximum entropy $smax"
                return t
            end    
        end
    end
end

function minimize(fitness, x, f, w, r, xm, xnew, s, n,
        FminTarget, sdmin, trace, MaxEvals, mumin; evals = 0)
    fmin, fmini = findmin(f)
    beta = 0.1/std(f)
    sd = 0.0
    dim = size(x, 1)
    local halt
    if trace < Inf
        println("Iteration     Fmin     Beta")
    end
    while true
        beta = selection_entropy!(w, s, beta, f, fmin)
        w .= exp2.(beta.*(fmin.-f))
        z = sum(w)
        z2 = sum(abs2, w)
        fm = mapreduce(*, +, f, w)/z
        if sdmin > 0.0
            sd = mapreduce((f,w)-> (f-fm)*w, +, f, w)/(z - z2/z)
            if sd < sdmin
                halt = :sdmin
                break
            end
        end
        if evals % trace == 0
            @printf("%9i %1.2e %1.2e\n", 
                    evals, fmin, beta)
        end
        if mumin
            xm .= x[:, fmini]
        else
            for i = 1:dim
                xm[i] = 0.0
                for j = 1:n
                    xm[i] += x[i, j]*w[j]
                end
                xm[i] /= z
            end
        end
        
        randn!(r)
        r .= r.*sqrt.(w)/sqrt(z-z2/z)
        mul!(xnew, x, r)
        xnew .+= xm.*(1-sum(r))

        j = argmin(w)
        f[j] = fitness(xnew)
        x[:, j] .= xnew
        evals += 1
        if f[j] < fmin
            fmin = f[j]
            fmini = j
        end
        if fmin < FminTarget
            halt = :FminTarget
            break
        elseif evals >= MaxEvals
            halt = :MaxEvals
            break
        elseif sum(isequal(f[j]), f) > 1
            halt = :DupFit
            break
        end
    end
    Dict(:fmin => fmin, :halt => halt, :xmin => x[:,fmini],
         :n => n, :mumin => mumin, :sd => sd,
         :evals => evals, :s=> s)
end


function minimize(fitness, xm0, xsd0, s; 
        FminTarget = -Inf, sdmin = 1e-16,
        trace = Inf, MaxEvals = 10000*length(xm0), 
        mumin = true,
        n = floor(Int64, exp2(s+1)))
    dim = length(xm0)
    x = randn(dim, n).*xsd0 .+ xm0
    f = vec(mapslices(fitness, x; dims=1))
    w = zeros(n)
    r = zeros(n)
    xm = zeros(dim)
    xnew = zeros(dim)
    minimize(fitness, x, f, w, r, xm, xnew, s, n,
        FminTarget, sdmin, trace, MaxEvals, mumin, evals = n)
end


end