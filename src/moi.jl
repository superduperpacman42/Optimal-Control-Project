function MOI.eval_objective(prob::HybridNLP, x)
    eval_f(prob, x)
end

function MOI.eval_objective_gradient(prob::HybridNLP, grad_f, x)
    grad_f!(prob, grad_f, x)
    return nothing
end

function MOI.eval_constraint(prob::HybridNLP,g,x)
    eval_c!(prob, g, x)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::HybridNLP, vec, x)
    n_nlp, m_nlp = num_primals(prob), num_duals(prob)
    if prob.use_sparse_jacobian
        jac = vec
    else
        jac = reshape(vec, m_nlp, n_nlp)
    end
    jac_c!(nlp, jac, x)
    return nothing
end

function MOI.features_available(prob::HybridNLP)
    return [:Grad, :Jac]
end

MOI.initialize(prob::HybridNLP, features) = nothing
function MOI.jacobian_structure(nlp::HybridNLP)
    if nlp.use_sparse_jacobian
        initialize_sparsity!(nlp)
        getrc(nlp.blocks)
    else
        vec(Tuple.(CartesianIndices(zeros(num_duals(nlp), num_primals(nlp)))))
    end
end
    
    
function initialize_sparsity!(nlp::HybridNLP{n,m}) where {n,m}
    blocks = nlp.blocks
    
    # Some useful variables
    xi,ui = nlp.xinds, nlp.uinds
    model = nlp.model
    N = nlp.N                      # number of time steps
    M = nlp.M                      # time steps per mode
    Nmodes = nlp.Nmodes            # number of mode sequences (N ÷ M)
    
    Nt = nlp.N
    Nx,Nu = n,m
    dt = nlp.times[2]
    Nm = nlp.M

    
    ic = (1:n) .+ (nlp.cinds[3][1]-1)
    for k = 1:(Nmodes-1)
        for j = 1:(Nm-1)
            s = (k-1)*Nm + j
            zi = [xi[s]; ui[s]]
            setblock!(blocks, ic, zi)
            setblock!(blocks, ic, xi[s+1])
            ic = ic .+ n
        end
        s = k*Nm
        zi = [xi[s]; ui[s]]
        setblock!(blocks, ic, zi)
        setblock!(blocks, ic, xi[s+1])
        ic = ic .+ n
    end
    for j = 1:(Nm-1)
        s = (Nmodes-1)*Nm + j
        zi = [xi[s]; ui[s]]
        setblock!(blocks, ic, zi)
        setblock!(blocks, ic, xi[s+1])

        ic = ic .+ n
    end
    
    setblock!(blocks, nlp.cinds[1], xi[1])
    setblock!(blocks, nlp.cinds[2], xi[end])
    
    t = 1
    for k = 1:nlp.N
        
        # stance constraint
        foot_ind = nlp.modes[k] == 1 ? 4 : 6
        setblock!(blocks, t + nlp.cinds[4][1] - 1, xi[k][foot_ind])
        
        # length constraint
        setblock!(blocks, nlp.cinds[5][1] - 1 + 2*(k-1) .+ (1:2), xi[k])
        t += 1
    end

end


"""
    solve(x0, nlp::HybridNLP; tol, c_tol, max_iter)

Solve the NLP `nlp` using Ipopt via MathOptInterface, providing `x0` as an initial guess.

# Keyword Arguments
The following arguments are sent to Ipopt
* `tol`: overall optimality tolerance
* `c_tol`: constraint feasibility tolerance
* `max_iter`: maximum number of solver iterations
"""
function solve(x0,prob::HybridNLP;
        tol=1.0e-6,c_tol=1.0e-6,max_iter=10000)
    n_nlp, m_nlp = num_primals(prob), num_duals(prob)
    x_l, x_u = fill(-Inf,n_nlp), fill(+Inf,n_nlp)
    c_l, c_u = prob.lb, prob.ub

    println("Creating NLP Block Data...")
    nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
    has_objective = true
    block_data = MOI.NLPBlockData(nlp_bounds, prob, has_objective)

    println("Creating Ipopt...")
    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol

    x = MOI.add_variables(solver, n_nlp)

    println("Adding constraints...")
    for i = 1:n_nlp
        xi = MOI.SingleVariable(x[i])
        MOI.add_constraint(solver, xi, MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, xi, MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    # Solve the problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    println("starting Ipopt Solve...")
    MOI.optimize!(solver)

    # Get the solution
    res = MOI.get(solver, MOI.VariablePrimal(), x)

    return res, solver
end