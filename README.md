# Quantitative genetic algorithm

*note this is research, there are no guarantees*

## How to use
```
include("QGA.jl")
`QGA.minimize(f, mu, sd, s)
```
* f: An objective function which takes a vector as input 
* mu: mean of initial distribution (vector)
* sd: standard deviation of initial distribution (vector)
* s: target entropy

Keyword arguments
* Fmintarget: stop when target value of objective reached (default -Inf)
* sdmin: stop when target value of standard deviation of fitness reached (default 0)
* MaxEvals: stop after (number) iterations (default 10000*dim)
* trace: show info every (number) iterations (default Inf)
* n: number of unique variants (default exp2(s+1))

Output is a dictionary with :halt describing reason for stopping. :DupFit means duplicate fitnesses were found and is generally a failure to converge.

## example
```
include("QGA.jl")`
dim = 5
fd = 2 .^(1:dim)
fellipse(x) = sum(fd.*x.^2)
QGA.minimize(fellipse, zeros(dim), ones(dim), sdmin = 0.0, 5.0, n = 512,
            FminTarget = 1e-8, trace = 100*dim)
```