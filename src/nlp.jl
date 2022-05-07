"""
    HybridNLP{n,m,L,Q}

Represents a (N)on(L)inear (P)rogram of a trajectory optimization problem,
with a dynamics model of type `L`, a quadratic cost function, horizon `T`, 
and initial and final state `x0`, `xf`.

The kth state and control can be extracted from the concatenated state vector `Z` using
`Z[nlp.xinds[k]]`, and `Z[nlp.uinds[k]]`.

# Constructor
    HybridNLP(model, obj, tf, N, M, x0, xf, [integration])

# Basic Methods
    Base.size(nlp)    # returns (n,m,T)
    num_ineq(nlp)     # number of inequality constraints
    num_eq(nlp)       # number of equality constraints
    num_primals(nlp)  # number of primal variables
    num_duals(nlp)    # total number of dual variables
    packZ(nlp, X, U)  # Stacks state `X` and controls `U` into one vector `Z`

# Evaluating the NLP
The NLP supports the following API for evaluating various pieces of the NLP:

    eval_f(nlp, Z)         # evaluate the objective
    grad_f!(nlp, grad, Z)  # gradient of the objective
    eval_c!(nlp, c, Z)     # evaluate the constraints
    jac_c!(nlp, c, Z)      # constraint Jacobian
"""
struct HybridNLP{n,m,L,Q} <: MOI.AbstractNLPEvaluator
    model::L                                 # dynamics model
    obj::Vector{QuadraticCost{n,m,Float64}}  # objective function
    N::Int                                   # number of knot points
    M::Int                                   # number of steps in each mode
    Nmodes::Int                              # number of modes
    tf::Float64                              # total time (sec)
    x0::MVector{n,Float64}                   # initial condition
    xf::MVector{n,Float64}                   # final condition
    times::Vector{Float64}                   # vector of times
    modes::Matrix{Float64}                   # contact sequence (4xN)
    terrain::Terrain                         # terrain heightmap
    xinds::Vector{SVector{n,Int}}            # Z[xinds[k]] gives states for time step k
    uinds::Vector{SVector{m,Int}}            # Z[uinds[k]] gives controls for time step k
    cinds::Vector{UnitRange{Int}}            # indices for each of the constraints
    lb::Vector{Float64}                      # lower bounds on the constraints
    ub::Vector{Float64}                      # upper bounds on the constraints
    zL::Vector{Float64}                      # lower bounds on the primal variables
    zU::Vector{Float64}                      # upper bounds on the primal variables
    rows::Vector{Int}                        # rows for Jacobian sparsity
    cols::Vector{Int}                        # columns for Jacobian sparsity
    use_sparse_jacobian::Bool
    blocks::BlockViews
    function HybridNLP(model, obj::Vector{<:QuadraticCost{n,m}},
            tf::Real, N::Integer, M::Integer, x0::AbstractVector, xf::AbstractVector, modes::Matrix{Float64}, terrain::Terrain,
            integration::Type{<:QuadratureRule}=RK4; use_sparse_jacobian::Bool=false
        ) where {n,m}
        # Create indices
        xinds = [SVector{n}((k-1)*(n+m) .+ (1:n)) for k = 1:N]
        uinds = [SVector{m}((k-1)*(n+m) .+ (n+1:n+m)) for k = 1:N-1]
        times = collect(range(0, tf, length=N))
        
        Nmodes = Int(ceil(N/M))
        
        # specify the constraint indices
        c_init_inds = 1:n                                                  # initial constraint
        c_term_inds = (c_init_inds[end]+1):(c_init_inds[end]+n)            # terminal constraint
        c_dyn_inds = (c_term_inds[end]+1):(c_term_inds[end]+n*(N-1))       # dynamics constraints
        c_length_inds = (c_dyn_inds[end]+1):(c_dyn_inds[end]+(4*N))        # length bounds     (4 per time step)
        c_height_inds = (c_length_inds[end]+1):(c_length_inds[end]+(4*N))  # stance foot height = ground, swing foot height > ground (4 per time step)
        
        m_nlp = c_height_inds.stop # total number of constraints
        
        
        # TODO: specify the bounds on the constraints
        #lb = fill(+Inf,m_nlp)                                                # lower bounds on the constraints
        #ub = fill(-Inf,m_nlp)                                                # upper bounds on the constraints
        
        lb = zeros(m_nlp)
        lb[c_length_inds] .= model.ℓmin^2 # min length
        
        ub = zeros(m_nlp)
        ub[c_length_inds] .= model.ℓmax^2 # max length

        # Swing height limits
        for k = 1:N
            for l = 1:4
                if modes[l,k] == 0
                    ub[c_height_inds[4*(k-1)+l]] = +Inf
                end
            end
        end
        
        # Other initialization
        cinds = [c_init_inds, c_term_inds, c_dyn_inds, c_length_inds, c_height_inds]
        n_nlp = n*N + (N-1)*m
        zL = fill(-Inf, n_nlp)
        zU = fill(+Inf, n_nlp)
        rows = Int[]
        cols = Int[]
        blocks = BlockViews(m_nlp, n_nlp)
        
        new{n,m,typeof(model), integration}(
            model, obj,
            N, M, Nmodes, tf, x0, xf, times, modes, terrain,
            xinds, uinds, cinds, lb, ub, zL, zU, rows, cols, use_sparse_jacobian, blocks
        )
    end
end
Base.size(nlp::HybridNLP{n,m}) where {n,m} = (n,m,nlp.N)
num_primals(nlp::HybridNLP{n,m}) where {n,m} = n*nlp.N + m*(nlp.N-1)
num_duals(nlp::HybridNLP) = nlp.cinds[end][end]

"""
    packZ(nlp, X, U)

Take a vector state vectors `X` and controls `U` and stack them into a single vector Z.
"""
function packZ(nlp, X, U)
    Z = zeros(num_primals(nlp))
    for k = 1:nlp.N-1
        Z[nlp.xinds[k]] = X[k]
        Z[nlp.uinds[k]] = U[k]
    end
    Z[nlp.xinds[end]] = X[end]
    return Z
end

"""
    unpackZ(nlp, Z)

Take a vector of all the states and controls and return a vector of state vectors `X` and
controls `U`.
"""
function unpackZ(nlp, Z)
    X = [Z[xi] for xi in nlp.xinds]
    U = [Z[ui] for ui in nlp.uinds]
    return X, U
end

function TrajOptPlots.visualize!(vis, nlp::HybridNLP, Z)
    TrajOptPlots.visualize!(vis, nlp.model, nlp.tf, unpackZ(nlp, Z)[1])
end