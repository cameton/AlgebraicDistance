module AlgebraicDistance

using LinearAlgebra
using Random

function sumzero!(X)
    for j in axes(X, 2)
        acc = 0
        @simd for i in axes(X, 1)
            acc += X[i, j]
        end
        acc /= size(X, 1)
        @simd for i in axes(X, 1)
            X[i, j] -= acc
        end
    end
    return X
end

gauss_seidel(L) = LowerTriangular(L)
jacobi(L) = Diagonal(diag(L))

function algdist!(X, R, L, P; numiters=10)
    sumzero!(X)
    for i in 1:numiters
        mul!(R, L, X)
        ldiv!(P, R)
        X += R
        sumzero!(X)
    end
    return X
end

function algdist(B, f, p; numvecs=1, numiters=10, rng=Xoshiro())
    L = B * B'
    P = f(L)
    X = randn(rng, size(B, 1), numvecs)
    R = similar(X)
    algdist!(X, R, L, P; numiters=numiters)
    grads = B' * X
    dists = p == Inf ? maximum(abs, grads; dims=2) : sum(x -> abs(x) ^ p, grads; dims=2) .^ (1 / p)
    return X, reshape(dists, :)
end

export algdist, gauss_seidel, jacobi

end # module
