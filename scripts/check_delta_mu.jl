using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, DataFrames, Statistics

chunk_dir = joinpath(@__DIR__, "..", "data", "sim2_condition_chunks")

println("===== gauss_iso M=11: 30-replicate averages at final generation =====")
for T in [1, 2, 5, 11, 50, 100, 200, 500]
    eco_vals = Float64[]
    full_vals = Float64[]
    risk_vals = Float64[]
    dm_vals = Float64[]
    for rep in 1:30
        fname = "gauss_iso__M11__T" * string(T) * "__rep" * string(rep) * ".jld2"
        fpath = joinpath(chunk_dir, fname)
        isfile(fpath) || continue
        df = load(fpath, "result")
        push!(eco_vals, df.frac_eco_veridical[end])
        push!(full_vals, df.frac_full_veridical[end])
        push!(risk_vals, df.mean_risk[end])
        push!(dm_vals, mean(df.delta_mu))
    end
    println("T=" * lpad(string(T), 3) *
            ": eco_verid=" * string(round(mean(eco_vals), digits=3)) *
            " full_verid=" * string(round(mean(full_vals), digits=3)) *
            " risk=" * string(round(mean(risk_vals), digits=5)) *
            " mean_delta_mu=" * string(round(mean(dm_vals), digits=6)) *
            " (n=" * string(length(eco_vals)) * ")")
end

println()
println("===== gauss_k1000 M=11: 30-replicate averages at final generation =====")
for T in [1, 2, 5, 11, 50, 100, 200, 500]
    eco_vals = Float64[]
    full_vals = Float64[]
    risk_vals = Float64[]
    dm_vals = Float64[]
    for rep in 1:30
        fname = "gauss_k1000__M11__T" * string(T) * "__rep" * string(rep) * ".jld2"
        fpath = joinpath(chunk_dir, fname)
        isfile(fpath) || continue
        df = load(fpath, "result")
        push!(eco_vals, df.frac_eco_veridical[end])
        push!(full_vals, df.frac_full_veridical[end])
        push!(risk_vals, df.mean_risk[end])
        push!(dm_vals, mean(df.delta_mu))
    end
    println("T=" * lpad(string(T), 3) *
            ": eco_verid=" * string(round(mean(eco_vals), digits=3)) *
            " full_verid=" * string(round(mean(full_vals), digits=3)) *
            " risk=" * string(round(mean(risk_vals), digits=5)) *
            " mean_delta_mu=" * string(round(mean(dm_vals), digits=6)) *
            " (n=" * string(length(eco_vals)) * ")")
end

println()
println("===== ALL families M=11: sorted by mean delta_mu =====")
rows = Tuple{String, Int, Float64, Float64, Float64, Float64}[]
for fam in ["gauss_iso", "gauss_k10", "gauss_k100", "gauss_k1000", "beta_narrow", "beta_wide"]
    for T in [1, 5, 11, 50, 100, 500]
        eco_vals = Float64[]
        full_vals = Float64[]
        risk_vals = Float64[]
        dm_vals = Float64[]
        for rep in 1:30
            fname = fam * "__M11__T" * string(T) * "__rep" * string(rep) * ".jld2"
            fpath = joinpath(chunk_dir, fname)
            isfile(fpath) || continue
            df = load(fpath, "result")
            push!(eco_vals, df.frac_eco_veridical[end])
            push!(full_vals, df.frac_full_veridical[end])
            push!(risk_vals, df.mean_risk[end])
            push!(dm_vals, mean(df.delta_mu))
        end
        length(eco_vals) == 0 && continue
        push!(rows, (fam, T, mean(dm_vals), mean(full_vals), mean(eco_vals), mean(risk_vals)))
    end
end

sort!(rows, by=x -> x[3])
println("family              T   mean_delta_mu  full_verid  eco_verid  risk")
for r in rows
    println(rpad(r[1], 20) *
            lpad(string(r[2]), 4) *
            "  " * lpad(string(round(r[3], digits=6)), 12) *
            "  " * lpad(string(round(r[4], digits=3)), 10) *
            "  " * lpad(string(round(r[5], digits=3)), 9) *
            "  " * lpad(string(round(r[6], digits=5)), 9))
end
