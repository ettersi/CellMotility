__precompile__()

module EvolvingCurves

using FastGaussQuadrature
using FixedSizeArrays
using PyPlot
using FEFunctions


##########
# Plotting

export update

function PyPlot.plot{T}(
    x::LinearFEFunction{Vec{2,T}}, 
    c::LinearFEFunction{T};
    size = 6, 
    vmin = nothing,
    vmax = nothing
) 
    scat = scatter(
        vec(x)[1:2:end], 
        vec(x)[2:2:end]; 
        c = vec(c),
        s = size,
        edgecolors = "face",
        vmin = vmin,
        vmax = vmax
    )
    gca()[:set_aspect]("equal")
    return scat
end
function update{T}(
    scat,
    x::LinearFEFunction{Vec{2,T}}, 
    c::LinearFEFunction{T}
) 
    scat[:set_offsets](reshape(vec(x), (2,length(x)))')
    scat[:set_color](scat[:to_rgba](vec(c)))
end
function PyPlot.plot{T}(
    x::FEFunction{Vec{2,T}}, 
    c::FEFunction{T};
    major_size = 10,
    minor_size = 0.5,
    nsamples = 10,
    vmin = nothing,
    vmax = nothing
) 
    scat_minor = plot(sample(x,nsamples),sample(c,nsamples), size=minor_size, vmin=vmin, vmax=vmax)
    scat_major = plot(sample(x,1),sample(c,1), size=major_size, vmin=vmin, vmax=vmax)
    return nsamples,scat_major,scat_minor
end
function update{T}(
    plotobj,
    x::FEFunction{Vec{2,T}}, 
    c::FEFunction{T}
) 
    nsamples,scat_major,scat_minor = plotobj
    update(scat_major, sample(x,1), sample(c,1))
    update(scat_minor, sample(x,nsamples), sample(c,nsamples))
end

function PyPlot.plot{T}(
    x::LinearFEFunction{Vec{2,T}};
    size = 6,
    color = "k"
) 
    scat = scatter(
        vec(x)[1:2:end], 
        vec(x)[2:2:end]; 
        s = size,
        c = color,
        edgecolors = "face"
    )
    gca()[:set_aspect]("equal")
    return scat
end
function update{T}(
    scat,
    x::LinearFEFunction{Vec{2,T}}
) 
    scat[:set_offsets](reshape(vec(x), (2,length(x)))')
end
function PyPlot.plot{T}(
    x::FEFunction{Vec{2,T}}, 
    major_size = 10,
    minor_size = 0.5,
    color = "k",
    nsamples = 10
) 
    scat_minor = plot(sample(x,nsamples), size=minor_size)
    scat_major = plot(sample(x,1), size=major_size)
    return nsamples,scat_major,scat_minor
end
function update{T}(
    plotobj,
    x::FEFunction{Vec{2,T}}, 
) 
    nsamples,scat_major,scat_minor = plotobj
    update(scat_major, sample(x,1))
    update(scat_minor, sample(x,nsamples))
end


############################
# Geometry utility functions

export perp
perp{T}(v::Vec{2,T}) = Vec{2,T}(-v[2],v[1])

export integrate
function integrate(x,c, nq = 5)
    (P,W) = gausslegendre(nq)
    P = 0.5*(P+1); W /= 2

    I = 0
    for i = 1:length(x)
        for (p,w) in zip(P,W)
            I += w*c[i,p]*norm(x[i,p,Val{1}])
        end
    end
    return I
end


##################
# Comparing curves

using Optim
export project
function project{T}(v::Vec{2,T}, x::FEFunction{Vec{2,T}})
    imin = 0
    pmin = zero(T)
    dmin = convert(T,Inf)
    for i = 1:length(x)
        res = optimize(p -> norm(v - x[i,p]), 0,1, abs_tol=1e-6)
        if dmin > res.f_minimum
            imin = i
            pmin = res.minimum
            dmin = res.f_minimum
        end
    end
    return imin,pmin,dmin
end

export linf_errors
function linf_errors(x,c, xref,cref)
    xlinf = zero(scalartype(x))
    clinf = zero(scalartype(c))
    for iref = 1:length(xref)
        i,p,d = project(xref[iref],x)
        xlinf = max(xlinf,d)
        clinf = max(clinf,abs(c[i,p] - cref[iref]))
    end
    return xlinf, clinf
end


########
# Evolve

export evolve
function evolve(
    x,T;
    α = 0.1, 
    ks = 1.0,
    kb = 1.0,
    conserve_volume = true,
    nq = 5
)
    for it = 2:length(T)
        dt = T[it] - T[it-1]

        Mx = @assemble_mat(x, nq, begin
            xp = x[i,p,Val{1}]
        end,begin
            Mat{2,2,Float64}((
                -perp(xp)*shape(x,s,p)*shape(x,ss,p),
                  α * xp *shape(x,s,p)*shape(x,ss,p)
            ))'
        end)
        Ax = @assemble_mat(x, nq, begin
            xp = x[i,p,Val{1}]
            xpp = x[i,p,Val{2}]
        end, begin
            dt*Mat{2,2,Float64}((
                (
                    # Mean curvature
                    -ks*perp(xp)/norm(xp)^2*shape(x,s,p,Val{2})*shape(x,ss,p,Val{0})
                    # Willmore 
                    +kb*( 
                        perp(xp)/norm(xp)^2*shape(x,s,p,Val{2})*(
                            + shape(x,ss,p,Val{2})/norm(xp)^2
                            - dot(xp,xpp)/norm(xp)^4*shape(x,ss,p,Val{1})
                        ) 
                        +
                        0.5*dot(perp(xp),xpp)^2/norm(xp)^8*perp(xp)*shape(x,s,p,Val{2})*shape(x,ss,p,Val{0})
                    )
                ),
                (xp/norm(xp)^2*shape(x,s,p,Val{2})*shape(x,ss,p,Val{0}))
            ))'
        end)

        # Volume conservation
        if conserve_volume
            ν = @assemble_vec(x,nq, begin
                xp = x[i,p,Val{1}]
            end,begin 
                norm(xp)*Vec{2}(shape(x,s,p),0) 
            end)
            I = @assemble_vec(x,nq, begin
                xp = x[i,p,Val{1}]
            end,begin 
                -perp(xp)*shape(x,s,p) 
            end)
            A = [
                Mx - Ax          sparsevec(ν)
                sparsevec(I)'    spzeros(1,1)
            ]
            b = [
                Mx*vec(x); 
                dot(I,vec(x))
            ]
            xnew = func(x,(A\b)[1:end-1])
        else
            xnew = func(x,(Mx - Ax)\(Mx*vec(x)))
        end
        x = xnew
    end
    return x
end
function evolve(
    x,c,f,T;
    α = 0.1, 
    ks = 1.0,
    kb = 1.0,
    kc = 1.0,
    conserve_volume = true,
    nq = 5
)
    Mc = @assemble_mat(c, nq, begin
        xp = x[i,p,Val{1}]
    end,begin
         shape(c,s,p)*shape(c,ss,p)*norm(xp)
    end)
    for it = 2:length(T)
        dt = T[it] - T[it-1]

        Mx = @assemble_mat(x, nq, begin
            xp = x[i,p,Val{1}]
        end,begin
            Mat{2,2,Float64}((
                -perp(xp)*shape(x,s,p)*shape(x,ss,p),
                  α * xp *shape(x,s,p)*shape(x,ss,p)
            ))'
        end)
        Ax = @assemble_mat(x, nq, begin
            xp = x[i,p,Val{1}]
            xpp = x[i,p,Val{2}]
        end, begin
            dt*Mat{2,2,Float64}((
                (
                    # Mean curvature
                    -ks*perp(xp)/norm(xp)^2*shape(x,s,p,Val{2})*shape(x,ss,p,Val{0})
                    # Willmore 
                    +kb*( 
                        perp(xp)/norm(xp)^2*shape(x,s,p,Val{2})*(
                            + shape(x,ss,p,Val{2})/norm(xp)^2
                            - dot(xp,xpp)/norm(xp)^4*shape(x,ss,p,Val{1})
                        ) 
                        +
                        0.5*dot(perp(xp),xpp)^2/norm(xp)^8*perp(xp)*shape(x,s,p,Val{2})*shape(x,ss,p,Val{0})
                    )
                ),
                (xp/norm(xp)^2*shape(x,s,p,Val{2})*shape(x,ss,p,Val{0}))
            ))'
        end)
        fx = @assemble_vec(x,nq, begin
            xp = x[i,p,Val{1}]
            fci = f(c[i,p])
        end,begin 
            Vec{2}(dt*fci*shape(x,s,p)*norm(xp),0.0) 
        end)

        # Volume conservation
        if conserve_volume
            ν = @assemble_vec(x,nq, begin
                xp = x[i,p,Val{1}]
            end,begin 
                norm(xp)*Vec{2}(shape(x,s,p),0) 
            end)
            I = @assemble_vec(x,nq, begin
                xp = x[i,p,Val{1}]
            end,begin 
                -perp(xp)*shape(x,s,p) 
            end)
            A = [
                Mx - Ax          sparsevec(ν)
                sparsevec(I)'    spzeros(1,1)
            ]
            b = [
                Mx*vec(x) + fx; 
                dot(I,vec(x))
            ]
            xnew = func(x,(A\b)[1:end-1])
        else
            xnew = func(x,(Mx - Ax)\(Mx*vec(x) + fx))
        end

        Mcnew = @assemble_mat(c, nq, begin
            xnewp = xnew[i,p,Val{1}]
        end,begin
            shape(c,s,p)*shape(c,ss,p)*norm(xnewp)
        end)
        Ac = @assemble_mat(c, nq, begin
            xnewp = xnew[i,p,Val{1}]
            xnewi = xnew[i,p]
            xi = x[i,p]
        end,begin
            (
                -dt*kc*shape(c,s,p,Val{1})*shape(c,ss,p,Val{1})/norm(xnewp)
                -dot(xnewp/norm(xnewp), xnewi - xi)*shape(c,s,p)*shape(c,ss,p,Val{1})
            )
        end)
        c = func(c,(Mcnew - Ac)\(Mc*vec(c)))

        x = xnew
        Mc = Mcnew
    end
    return x,c
end

end
