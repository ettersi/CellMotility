using FastGaussQuadrature
using FixedSizeArrays
using FastAnonymous
using PyPlot
using FEFunctions
using EvolvingCurves

clogspace(start::Real, stop::Real,  n::Integer = 50) = return start*(stop/start).^linspace(0,1,n)

iround(x::Real) = convert(Int,round(x))
iround(x::AbstractVector) = [iround(x) for x in x]


# Set up simulation parameters

function parameters(nP)
    assemble_kwargs(;kwargs...) = kwargs

    P = linspace(0,2*π,nP+1)
    x = HermiteFEFunction(Vec{2,Float64}, p -> [cos(p), sin(p)], P)
    c = CubicFEFunction(Float64, p -> 0.5+20*exp(-(2*(p<π?p:p-2π))^2), P)
    f = @anon c -> c^2

    return x,c,f,assemble_kwargs(α = 1e-4, kb = 0.2)#, kb = 0.2, kc = 0.05, ks = 10
end


# Gather convergence data

using JLD
function convergence_run(;reuse = false)
    if !isdir("convergence") mkdir("convergence"); end

    function solve(nP,nT)
        T = linspace(0,2e-3,nT+1)
        x,c,f,kwargs = parameters(nP)
        return evolve(x,c,f,T; kwargs...)
    end

    nPmax = 100
    nTmax = 10000

    nPref = iround(nPmax*10.0^(1/4))
    nTref = nTmax*10
    nP = unique(iround(clogspace(2,nPmax,40)))
    nT = unique(iround(clogspace(10,nTmax,40)))

    println("Computing reference solution")
    if !reuse
        @time xref,cref = solve(nPref, nTref)
        save("convergence/refsol.jld",
            "nPref",nPref,
            "nTref",nTref,
            "xref", xref,
            "cref", cref
        )
    end
    data = load("convergence/refsol.jld")
    xref = data["xref"]
    cref = data["cref"]

    xref = sample(xref,5)
    cref = sample(cref,5)

    xP = Array{Float64}(nP)
    cP = Array{Float64}(nP)
    for i = 1:length(nP)
        println("nP-study: computing errors for nP = ", nP[i])
        @time x,c = solve(nP[i], nTref)
        @time xP[i], cP[i] = linf_errors(x,c, xref,cref)
    end
    println()

    xT = Array{Float64}(nT)
    cT = Array{Float64}(nT)
    for i = 1:length(nT)
        println("nT-study: computing errors for nT = ", nT[i])
        @time x,c = solve(nPref, nT[i])
        @time xT[i], cT[i] = linf_errors(x,c, xref,cref)
    end
    println()

    save("convergence/data.jld",
        "nP",nP,
        "xP",xP,
        "cP",cP,
        "nT",nT,
        "xT",xT,
        "cT",cT
    )
end

# Plot convergence data (run convergence_run() first)

function convergence_plot()
    data = load("convergence/data.jld")
    nP = data["nP"]
    xP = data["xP"]
    cP = data["cP"]
    nT = data["nT"]
    xT = data["xT"]
    cT = data["cT"]

    fig = figure(figsize = (4,3))
    idx = nP .> 20
    loglog(nP[idx],2*cP[end]*nP[end]^4./nP[idx].^4, "k-", lw = 1.5)
    loglog(nP[idx],2*xP[end]*nP[end]^4./nP[idx].^4, "k-", lw = 1.5)
    loglog(nP,xP, "b-o", mec = "b", label="x", ms = 2.5, mfc="w", mew = 1.5, lw = 1.5)
    loglog(nP,cP, "g-D", mec = "g", label="c", ms = 2, mfc="w", mew = 1.5, lw = 1.5)
    xlabel("# mesh points")
    ylabel("Error")
    ylim([1e-6,1e1])
    tight_layout()
    savefig("convergence/convergence_nP.png")
    close(fig)

    fig = figure(figsize = (4,3))
    loglog(nT,2*cT[end]*nT[end]./nT, "k-", lw = 1.5)
    loglog(nT,2*xT[end]*nT[end]./nT, "k-", lw = 1.5)
    loglog(nT,xT, "b-o", mec = "b", label="x", ms = 2.5, mfc="w", mew = 1.5, lw = 1.5)
    loglog(nT,cT, "g-D", mec = "g", label="c", ms = 2, mfc="w", mew = 1.5, lw = 1.5)
    xlabel("# time steps")
    ylabel("Error")
    ylim([1e-6,1e1])
    legend(loc="upper right", fontsize=12)
    tight_layout()
    savefig("convergence/convergence_nT.png")
    close(fig)
end


# Animate evolution

function animate()
    if !isdir("animate") mkdir("animate"); end
    for file in readdir("animate") rm("animate/$file"); end

    nt = 50
    T = unique([linspace(0,1e-2,101); clogspace(1e-2,8,80)]) 
    x,c,f,kwargs = parameters(16)

    fig = figure(figsize=(4,3))
    xlim([-1.2,2.8])
    ylim([-1.2,1.2])
    try
        plotobj = plot(x,c)
        colorbar()

        function saveplot(it,x,c)
            update(plotobj, x,c)
            draw()

            savefig("animate/it$it.png")
        end

        saveplot(1,x,c)
        for it in 2:length(T)
            TT = linspace(T[it-1],T[it], nt)
            x,c = evolve(x,c,f,TT; kwargs...)
            saveplot(it,x,c)
        end
    catch 
        rethrow()
    finally 
        close(fig)
    end
end


# Demonstrate mesh degeneration

function mesh_degeneration()
    if !isdir("mesh_degeneration") mkdir("mesh_degeneration"); end
    for file in readdir("mesh_degeneration") rm("mesh_degeneration/$file"); end

    for (iα,α) = enumerate((1e-4,1e4))
        nt = 50
        T = unique([linspace(0,1e-3,11); clogspace(1e-3,8,91)]) 
        x,c,f,kwargs = parameters(20)

        fig = figure(figsize=(4,3))
        xlim([-1.2,2.8])
        ylim([-1.2,1.2])
        axis("off")
        gca()[:get_xaxis]()[:set_visible](false)
        gca()[:get_yaxis]()[:set_visible](false)
        try
            plotobj = plot(x,c, major_size=40, minor_size=2)

            function saveplot(it,x,c)
                update(plotobj, x,c)
                draw()

                savefig("mesh_degeneration/alpha$(iα)_it$it.png")
            end

            saveplot(1,x,c)

            for it in 2:length(T)
                TT = linspace(T[it-1],T[it],nt)
                x,c = evolve(x,c,f,TT; filter(t -> t[1] != :α, kwargs)..., α = α)
                saveplot(it,x,c)
            end
        catch 
            rethrow()
        finally 
            close(fig)
        end
    end
end

