"""
    eval_c!(nlp, c, Z)

Evaluate all the constraints
"""
function eval_c!(nlp::HybridNLP, c, Z)
    xi = nlp.xinds
    c[nlp.cinds[1]] .= Z[xi[1]] - nlp.x0
    c[nlp.cinds[2]] .= Z[xi[end]] - nlp.xf
    dynamics_constraint!(nlp, c, Z)
    length_constraint!(nlp, c, Z)
    height_constraint!(nlp, c, Z)
    return c
end

function dynamics_constraint!(nlp::HybridNLP, c, Z)
    X, U = unpackZ(nlp, Z)
    c_init_inds, c_term_inds, c_dyn_inds, c_length_inds, c_height_inds, c_body_inds = nlp.cinds
    modes = nlp.modes
    n = length(nlp.xinds[1])
    for k = 1:N-1
        t = nlp.times[k]
        dt = nlp.times[k+1] - t
        if any(modes[:,k] .!= modes[:,k+1])
            x1 = rk4(nlp.model, X[k], U[k], t, dt, modes[:,k])
            c[c_dyn_inds[(k-1)*n+1:k*n]] = jumpmap(nlp.model, x1, modes[:,k+1]) - X[k+1]
        else
            c[c_dyn_inds[(k-1)*n+1:k*n]] = rk4(nlp.model, X[k], U[k], t, dt, modes[:,k]) - X[k+1]
        end
    end
end

function length_constraint!(nlp::HybridNLP, c, Z)
    model = nlp.model
    X, U = unpackZ(nlp, Z)
    c_init_inds, c_term_inds, c_dyn_inds, c_length_inds, c_height_inds, c_body_inds = nlp.cinds
    s = [model.s1, model.s2, model.s3, model.s4]
    for k = 1:N
        Q = q2Q(X[k][4:7])
        for l = 1:4
            L = X[k][1:3] + Q*s[l] - X[k][3*l+5:3*l+7]
            c[c_length_inds[(k-1)*4+l]] = L'*L
        end
    end
end

function height_constraint!(nlp::HybridNLP, c, Z)
    c_init_inds, c_term_inds, c_dyn_inds, c_length_inds, c_height_inds, c_body_inds = nlp.cinds
    X, U = unpackZ(nlp, Z)
    # Height 4N
    for k = 1:N
        for l = 1:4
            c[c_height_inds[k*4+l-4]] = X[k][3*(l-1)+10] - getHeight(nlp.terrain, X[k][3*(l-1)+8], X[k][3*(l-1)+9])
        end
        c[c_body_inds[k]] = X[k][3] - getHeight(nlp.terrain, X[k][1], X[k][2])
    end
end

"""
    jac_c!(nlp, jac, Z)

Evaluate the constraint Jacobians.
"""
function jac_c!(nlp::HybridNLP{n,m}, jacvec::AbstractVector, Z) where {n,m}
    jac = NonzerosVector(jacvec, nlp.blocks)
    jacvec .= 0
    
    xi,ui = nlp.xinds, nlp.uinds
    model = nlp.model
    N = nlp.N                      # number of time steps
    M = nlp.M                      # time steps per mode
    X, U = unpackZ(nlp, Z)
    modes = nlp.modes
    c_init_inds, c_term_inds, c_dyn_inds, c_length_inds, c_height_inds, c_body_inds = nlp.cinds
    
    # Init/final n, n
    jac[c_init_inds, xi[1]] = I(n)
    jac[c_term_inds, xi[end]] = I(n)
    
    # Dynamics n(N-1)
    for k = 1:N-1
        t = nlp.times[k]
        dt = nlp.times[k+1] - t
        A, B = discrete_jacobian(model, X[k], U[k], t, dt, modes[:,k])
        if any(modes[:,k] .!= modes[:,k+1])
            J = jump_jacobian(modes[:,k+1])
            jac[c_dyn_inds[(k-1)*n+1:k*n], vcat(xi[k], ui[k], xi[k+1])] = hcat(J*A, J*B, -LinearAlgebra.I(n))
        else
            jac[c_dyn_inds[(k-1)*n+1:k*n], vcat(xi[k], ui[k], xi[k+1])] = hcat(A, B, -LinearAlgebra.I(n))
        end
    end
    
    # Length 4N
    s = [model.s1, model.s2, model.s3, model.s4]
    for k = 1:N
        v = zeros(4,n)
        Q = q2Q(X[k][4:7])
        q = Rotations.UnitQuaternion(X[k][4:7])
        for l = 1:4
            L = X[k][1:3] + Q*s[l] - X[k][3*l+5:3*l+7]
            v[l,1:3] = 2*L
            v[l,3*l+5:3*l+7] = -2*L
            v[l,4:7] = 2*L'*Rotations.???rotate(q, s[l])*norm(X[k][4:7])
        end
        jac[c_length_inds[k*4-3:k*4], xi[k]] = v
    end
    
    # Height 4N
    for k = 1:N
        for l = 1:4
            gx, gy = getSlope(nlp.terrain, X[k][3*(l-1)+8], X[k][3*(l-1)+9])
            jac[c_height_inds[k*4+l-4],xi[k][3*(l-1)+8]] = -gx
            jac[c_height_inds[k*4+l-4],xi[k][3*(l-1)+9]] = -gy
            jac[c_height_inds[k*4+l-4],xi[k][3*(l-1)+10]] = 1
        end
        gx, gy = getSlope(nlp.terrain, X[k][1], X[k][2])
        jac[c_body_inds[k], xi[k][3]] = 1
        jac[c_body_inds[k], xi[k][1]] = -gx
        jac[c_body_inds[k], xi[k][2]] = -gy
    end
end
