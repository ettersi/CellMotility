__precompile__()

module FEFunctions

###################
# Utility functions

export scalartype

scalartype(::Type{Any}) = throw(MethodError(scalartype, (Any,)))
scalartype(t::DataType) = scalartype(super(t))
scalartype(x) = scalartype(typeof(x))
scalartype{T <: Number}(::Type{T}) = T


###############
# Vec functions

using FixedSizeArrays
scalartype{D,T}(::Type{Vec{D,T}}) = T
mattype{D,T}(::Type{Vec{D,T}}) = Mat{D,D,T}
mattype{T <: Number}(::Type{T}) = T


############
# FEFunction

export FEFunction, nshape, shape, index, func, sample

abstract FEFunction{T}

macro defineFEFunctionType(name, nloc)
    esc(quote
        export $name
        type $name{T} <: FEFunction{T}
            data::Vector{T}
        end

        $name(T::Type, n::Int) = $name(Array{T}($nloc*n))
        $name(args...) = $name(Float64,args...)

        Base.length(u::$name) = div(length(u.data),$nloc)
        index(u::$name, i, s) = mod1($nloc*(i-1)+s,length(u.data))
    end)
end

scalartype{T}(::Type{FEFunction{T}}) = scalartype(T)
Base.eltype{T}(::Type{FEFunction{T}}) = T

Base.vec(u::FEFunction) = reinterpret(scalartype(u), u.data)
func{F <: FEFunction}(::Type{F}, data) = F(reinterpret(eltype(F), data))
func{F <: FEFunction}(::F, data) = func(F,data)

nshape{T <: FEFunction}(::Type{T}) = throw(MethodError(nshape, (T,)))
nshape(x::FEFunction) = nshape(typeof(x))

shape{T <: FEFunction}(::Type{T},args...) = throw(MethodError(shape, (T,args...)))
shape{T <: FEFunction}(::Type{T},s,p) = shape(T,s,p,Val{0})
shape(x::FEFunction,args...) = shape(typeof(x),args...)

Base.getindex(u::FEFunction, i::Int, s::Int) = @inbounds return u.data[index(u,i,s)]
Base.setindex!(u::FEFunction, ui, i::Int, s::Int) = @inbounds u.data[index(u,i,s)] = ui

function Base.getindex{D}(u::FEFunction, i::Int, p, ::Type{Val{D}}) 
    uip = zero(eltype(u))
    for s = 1:nshape(u)
        uip .+= u[i,s]*shape(u,s,p,Val{D})
    end
    return uip
end
Base.getindex(u::FEFunction, i::Int, p) = u[i,p,Val{0}]

function sample{D}(u::FEFunction, n::Int, ::Type{Val{D}}) 
    P = linspace(0,1,n+1)[1:end-1]
    v = LinearFEFunction(eltype(u), length(u)*n)
    for i = 1:length(u)
        for (ip,p) = enumerate(P)
            v[n*(i-1) + ip] = u[i,p,Val{D}]
        end
    end
    return v
end
sample(u::FEFunction, n::Int) = sample(u,n,Val{0})


##################
# LinearFEFunction

@defineFEFunctionType(LinearFEFunction,1)

function LinearFEFunction(T, f, P)
    u = LinearFEFunction(T,length(P)-1)
    for (i,p) = enumerate(P[1:end-1])
        u[i] = f(p)
    end
    return u
end

nshape{T}(::Type{LinearFEFunction{T}}) = 2
function shape{T}(::Type{LinearFEFunction{T}}, s::Int, p, ::Type{Val{0}})
    if s == 1
        return 1 - p
    else
        return p
    end
end
function shape{T}(::Type{LinearFEFunction{T}}, s::Int, p, ::Type{Val{1}})
    if s == 1
        return -one(p)
    else
        return one(p)
    end
end

Base.getindex(x::LinearFEFunction, i::Int) = x[i,1]
Base.setindex!(x::LinearFEFunction, xi, i::Int) = x[i,1] = xi


##################
# CubicFEFunction

@defineFEFunctionType(CubicFEFunction,3)

function CubicFEFunction(T, f, P)
    u = CubicFEFunction(T,length(P)-1)
    for i = 1:length(P)-1
        for (j,p) = enumerate(linspace(P[i],P[i+1],4)[1:end-1])
            u[i,j] = f(p)
        end
    end
    return u
end

nshape{T}(::Type{CubicFEFunction{T}}) = 4
function shape{T}(::Type{CubicFEFunction{T}}, s::Int, p, ::Type{Val{0}})
    if s == 1
        return -9/2*(p-1/3)*(p-2/3)*(p-1)
    elseif s == 2
        return 27/2*p*(p-2/3)*(p-1)
    elseif s == 3
        return -27/2*p*(p-1/3)*(p-1)
    elseif s == 4
        return 9/2*p*(p-1/3)*(p-2/3)
    else
        return convert(typeof(p), NaN)
    end
end
function shape{T}(::Type{CubicFEFunction{T}}, s::Int, p, ::Type{Val{1}})
    if s == 1
        return -9/2*((p-2/3)*(p-1) + (p-1/3)*(p-1) + (p-1/3)*(p-2/3))
    elseif s == 2
        return 27/2*((p-2/3)*(p-1) + p*(p-1) + p*(p-2/3))
    elseif s == 3
        return -27/2*((p-1/3)*(p-1) + p*(p-1) + p*(p-1/3))
    elseif s == 4
        return 9/2*((p-1/3)*(p-2/3) + p*(p-2/3) + p*(p-1/3))
    else
        return convert(typeof(p), NaN)
    end
end


###################
# HermiteFEFunction

@defineFEFunctionType(HermiteFEFunction,2)

function HermiteFEFunction(T::DataType, f, df, P)
    fac = (P[end]-P[1])/(length(P)-1)
    u = HermiteFEFunction(T,length(P)-1)
    for (i,p) = enumerate(P[1:end-1])
        u[i,1] = f(p)
        u[i,2] = fac*df(p)
    end
    return u
end

using ForwardDiff
function HermiteFEFunction(T::DataType, f, P)
    fac = (P[end]-P[1])/(length(P)-1)
    u = HermiteFEFunction(T,length(P)-1)
    for (i,p) = enumerate(P[1:end-1])
        _,fp = derivative(f,p,AllResults)
        u[i,1] = value(fp)
        u[i,2] = fac*derivative(fp)
    end
    return u
end

nshape{T}(::Type{HermiteFEFunction{T}}) = 4
  hermite_f(p) = 3*p.^2 - 2*p.^3
  hermite_df(p) = p.^3 - p.^2
 dhermite_f(p) = 6*p - 6*p.^2
 dhermite_df(p) = 3*p.^2 - 2*p
ddhermite_f(p) = 6 - 12*p
ddhermite_df(p) = 6*p - 2
function shape{T}(::Type{HermiteFEFunction{T}}, s::Int, p, ::Type{Val{0}})
    if s == 1
        return hermite_f(1-p)
    elseif s == 2
        return -hermite_df(1-p)
    elseif s == 3
        return hermite_f(p)
    elseif s == 4
        return hermite_df(p)
    else
        return convert(typeof(p), NaN)
    end
end
function shape{T}(::Type{HermiteFEFunction{T}}, s::Int, p, ::Type{Val{1}})
    if s == 1
        return -dhermite_f(1-p)
    elseif s == 2
        return dhermite_df(1-p)
    elseif s == 3
        return dhermite_f(p)
    elseif s == 4
        return dhermite_df(p)
    else
        return convert(typeof(p), NaN)
    end
end
function shape{T}(::Type{HermiteFEFunction{T}}, s::Int, p, ::Type{Val{2}})
    if s == 1
        return ddhermite_f(1-p)
    elseif s == 2
        return -ddhermite_df(1-p)
    elseif s == 3
        return ddhermite_f(p)
    elseif s == 4
        return ddhermite_df(p)
    else
        return convert(typeof(p), NaN)
    end
end


####################
# Assembly functions

using FastGaussQuadrature

export @assemble_mat
macro assemble_mat(u, nq, preproc, expr)
    return quote
        u = $(esc(u))

        (P,W) = gausslegendre($(esc(nq)))
        P = 0.5*(P+1); W /= 2

        d = div(sizeof(eltype(u)),sizeof(scalartype(u)))
        N = d^2*nshape(u)^2
        I = Array{Int}(length(u)*N)
        J = Array{Int}(length(u)*N)
        V = Array{scalartype(u)}(length(u)*N)
        for i = 1:length(u)
            Mloc = zeros(mattype(eltype(u)),nshape(u),nshape(u))
            for (p,w) = zip(P,W)
                $(esc(:i)) = i
                $(esc(:p)) = p
                $(esc(preproc))

                for s = 1:nshape(u)
                    for ss = 1:nshape(u)
                        $(esc(:s)) = s
                        $(esc(:ss)) = ss
                        Mloc[ss,s] .+= w*$(esc(expr))
                    end
                end
            end

            for s = 1:nshape(u)
                sidx = index(u,i,s)
                for ss = 1:nshape(u)
                    ssidx = index(u,i,ss)

                    for id = 1:d
                        I[N*(i-1)+d^2*nshape(u)*(s-1)+d^2*(ss-1)+d*(id-1)+(1:d)] = d*(ssidx-1) + (1:d)
                        J[N*(i-1)+d^2*nshape(u)*(s-1)+d^2*(ss-1)+d*(id-1)+(1:d)] = d*( sidx-1) + id
                    end
                end
            end
            V[N*(i-1)+(1:N)] = reinterpret(scalartype(u),Mloc, (N,))
        end
        sparse(I,J,V)
    end
end

export @assemble_vec
macro assemble_vec(u, nq, preproc, expr)
    return quote
        u = $(esc(u))

        (P,W) = gausslegendre($(esc(nq)))
        P = 0.5*(P+1); W /= 2

        b = typeof(u)(zeros(u.data))
        for i = 1:length(u)
            for (p,w) = zip(P,W)
                $(esc(:i)) = i
                $(esc(:p)) = p
                $(esc(preproc))

                for s = 1:nshape(u)
                    $(esc(:s)) = s
                    b[i,s] += w*$(esc(expr))
                end
            end
        end
        vec(b)
    end
end

end
