var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#YAAD.arg",
    "page": "Home",
    "title": "YAAD.arg",
    "category": "function",
    "text": "arg(node, i) -> ArgumentType\n\nReturns the i-th argument of the call in node.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.args",
    "page": "Home",
    "title": "YAAD.args",
    "category": "function",
    "text": "args(node) -> Tuple\n\nReturns the arguments of the call in node.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.backward",
    "page": "Home",
    "title": "YAAD.backward",
    "category": "function",
    "text": "backward(node) -> nothing\n\nBackward evaluation of the comput-graph.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.forward",
    "page": "Home",
    "title": "YAAD.forward",
    "category": "function",
    "text": "forward(node) -> output\n\nForward evaluation of the comput-graph. This method will call the operator in the comput-graph and update the cache.\n\nforward(f, args...) -> output\n\nFor function calls.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.gradient",
    "page": "Home",
    "title": "YAAD.gradient",
    "category": "function",
    "text": "gradient(node, grad)\n\nReturns the gradient.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.operator",
    "page": "Home",
    "title": "YAAD.operator",
    "category": "function",
    "text": "operator(node) -> YAAD.Operator\n\nReturns the operator called in this node.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.register-Tuple{Any,Vararg{Any,N} where N}",
    "page": "Home",
    "title": "YAAD.register",
    "category": "method",
    "text": "register(f, args...; kwargs...)\n\nThis is just a alias for constructing a CachedNode. But notice this function is used for register a node in tape in the global tape version implementation:\n\nhttps://github.com/Roger-luo/YAAD.jl/tree/tape\n\n\n\n\n\n"
},

{
    "location": "#YAAD.value",
    "page": "Home",
    "title": "YAAD.value",
    "category": "function",
    "text": "value(node)\n\nReturns the value when forwarding at current node. value is different than forward method, value only returns what the node contains, it will throw an error, if this node does not contain anything.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.AbstractNode",
    "page": "Home",
    "title": "YAAD.AbstractNode",
    "category": "type",
    "text": "AbstractNode\n\nAbstract type for nodes in computation graph.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.CachedNode",
    "page": "Home",
    "title": "YAAD.CachedNode",
    "category": "type",
    "text": "CachedNode{NT, OutT} <: AbstractNode\n\nStores the cache of output with type OutT from a node of type NT in comput-graph. CachedNode is mutable, its output can be updated by forward.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.LeafNode",
    "page": "Home",
    "title": "YAAD.LeafNode",
    "category": "type",
    "text": "LeafNode <: AbstractNode\n\nAbstract type for leaf nodes in a computation graph.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.Node",
    "page": "Home",
    "title": "YAAD.Node",
    "category": "type",
    "text": "Node{FT, ArgsT} <: AbstractNode\n\nGeneral node in a comput-graph. It stores a callable operator f of type FT and its arguments args in type ArgsT which should be a tuple.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.Variable",
    "page": "Home",
    "title": "YAAD.Variable",
    "category": "type",
    "text": "Variable{T} <: LeafNode\n\nA kind of leaf node. A general type for variables in a comput-graph. Similar to PyTorch\'s Variable, gradient will be accumulated to var.grad.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.PrintTrait",
    "page": "Home",
    "title": "YAAD.PrintTrait",
    "category": "function",
    "text": "PrintTrait(node) -> Trait\n\n\n\n\n\n"
},

{
    "location": "#YAAD.backward_type_assert",
    "page": "Home",
    "title": "YAAD.backward_type_assert",
    "category": "function",
    "text": "backward_type_assert(node, grad)\n\nthrow more readable error msg for backward type check.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.kwargs",
    "page": "Home",
    "title": "YAAD.kwargs",
    "category": "function",
    "text": "kwargs(node) -> NamedTuple\n\nReturns the keyword arguements of the call in node.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.uncat-Tuple{Any,Any,Vararg{Any,N} where N}",
    "page": "Home",
    "title": "YAAD.uncat",
    "category": "method",
    "text": "uncat(dims, cat_output, xs...) -> Vector{SubArray}\n\nThe reverse operation of [Base.cat], it will return corresponding [Base.view] of the inputs of a cat.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.ComputGraphStyle",
    "page": "Home",
    "title": "YAAD.ComputGraphStyle",
    "category": "type",
    "text": "ComputGraphStyle <: Broadcast.BroadcastStyle\n\nThis style of broadcast will forward the broadcast expression to be registered in a computation graph, rather than directly calculate it.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.Operator",
    "page": "Home",
    "title": "YAAD.Operator",
    "category": "type",
    "text": "Operator\n\nAbstract type for operators in the computation graph.\n\n\n\n\n\n"
},

{
    "location": "#Yet-Another-Automatic-Differentiation-1",
    "page": "Home",
    "title": "Yet Another Automatic Differentiation",
    "category": "section",
    "text": "Modules = [YAAD]\nOrder = [:function, :type, :module]"
},

{
    "location": "#YAAD.Batched",
    "page": "Home",
    "title": "YAAD.Batched",
    "category": "module",
    "text": "Batched operation in Julia.\n\nThis module wraps some useful batched operation with a plain for-loop on CPU. All the functions in this module are defined with gradients in YAAD.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.Batched.ScalarIdentity",
    "page": "Home",
    "title": "YAAD.Batched.ScalarIdentity",
    "category": "type",
    "text": "ScalarIdentity{B, K, T} <: AbstractArray{T, 3}\n\nA batch of scalar multiplies a batch of identities, where batch size is B, each identity\'s size is K.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.Batched.Transpose",
    "page": "Home",
    "title": "YAAD.Batched.Transpose",
    "category": "type",
    "text": "Transpose{B, T, AT <: AbstractArray{T, 3}} <: AbstractArray{T, 3}\n\nBatched transpose. Transpose a batch of matrix.\n\n\n\n\n\n"
},

{
    "location": "#Batched-Operations-1",
    "page": "Home",
    "title": "Batched Operations",
    "category": "section",
    "text": "Modules = [YAAD.Batched]"
},

{
    "location": "#YAAD.Trait",
    "page": "Home",
    "title": "YAAD.Trait",
    "category": "module",
    "text": "Trait\n\nThis module contains function traits as a subtype of Operator.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.Trait.Broadcasted",
    "page": "Home",
    "title": "YAAD.Trait.Broadcasted",
    "category": "type",
    "text": "Broadcasted{FT} <: Operator\n\nThis trait wraps a callable object that being broadcasted. It will help to dispatch different gradient methods overloaded for broadcasted operation comparing to Method.\n\n\n\n\n\n"
},

{
    "location": "#YAAD.Trait.Method",
    "page": "Home",
    "title": "YAAD.Trait.Method",
    "category": "type",
    "text": "Method{FT} <: Operator\n\nThis trait wraps a callable object in Julia (usually a Function).\n\n\n\n\n\n"
},

{
    "location": "#Operator-Traits-1",
    "page": "Home",
    "title": "Operator Traits",
    "category": "section",
    "text": "Modules = [YAAD.Trait]"
},

]}
