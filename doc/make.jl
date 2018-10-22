using Documenter, YAAD

# TODO: use Literate to process examples
# using Literate
# preprocess tutorial scripts

# make documents
makedocs(
    modules = [YAAD],
    clean = false,
    format = :html,
    sitename = "YAAD.jl",
    linkcheck = !("skiplinks" in ARGS),
    analytics = "UA-89508993-1",
    pages = [
        "Home" => "index.md",
    ],
    html_prettyurls = !("local" in ARGS),
    html_canonical = "https://rogerluo.me/YAAD.jl/latest/",
)

deploydocs(
    repo = "github.com/Roger-luo/YAAD.jl.git",
    target = "build",
    julia = "1.0",
    osname = "osx",
    deps = nothing,
    make = nothing,
)
