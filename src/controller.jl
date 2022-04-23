function tvlqr(A,B,Q,R,Qf)
    # Extract some variables
    T = length(A)+1
    n,m = size(B[1])
    P = [zeros(n,n) for k = 1:T]
    K = [zeros(m,n) for k = 1:T-1]

    P[end] .= Qf
    for k = reverse(1:T-1) 
        K[k] .= (R + B[k]'P[k+1]*B[k])\(B[k]'P[k+1]*A[k])
        P[k] .= Q + A[k]'P[k+1]*A[k] - A[k]'P[k+1]*B[k]*K[k]
    end

    return K,P
end

function lqr(A,B,Q,R; max_iters=200, tol=1e-6)
    P = copy(Q)
    n,m = size(B)
    K = zeros(m,n)
    K_prev = copy(K)

    P = copy(Q)
    for k = 1:max_iters
        K .= (R + B'P*B) \ (B'P*A)
        P .= Q + A'P*A - A'P*B*K
        if norm(K-K_prev,Inf) < tol
            println("Converged in $k iterations")
            return K
        end
        K_prev .= K
    end
    return K * NaN
end