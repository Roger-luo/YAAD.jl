# YAAD.jl

Yet Another Automatic Differentiation package in Julia.

## Installation

```
pkg> add https://github.com/Roger-luo/YAAD.jl.git#master
```

## Introduction

This project aims to provide a similar interface with PyTorch's autograd, while
keeping things simple. The core implementation only contains a straight-forward
200 line of Julia. It is highly inspired by [AutoGrad.jl](https://github.com/denizyuret/AutoGrad.jl)
and [PyTorch](https://github.com/pytorch/pytorch)

Every operation will directly return a `CachedNode`, which constructs a computation
graph dynamically without using a tape.
Although performance is the crucial goal of this AD package. It is actually faster than
several packages and even PyTorch itself.

The simplicity provides us a faster development, while waiting for those Cassette-based
AD packages like [Zygote.jl](https://github.com/FluxML/Zygote.jl)
