# YAAD's Documentations

to build the doc locally, run

```julia
julia --project=doc/ -e 'using Pkg; Pkg.instantiate(); Pkg.add(PackageSpec(path=pwd()))'
julia --project=doc doc/make.jl local
```
