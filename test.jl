d = WeakKeyDict()

A = rand(2, 2)
function check(A)
    d[A] = 2
end

A = rand(2, 2)
