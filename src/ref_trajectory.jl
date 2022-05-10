function reference_trajectory(model::SimpleQuadruped, times, modes, step_length, terrain::Terrain;
        xinit = 0.0,
        xterm = 10.0
    )
    # no global planning, just walking in a straight line right now
    
    # Some useful variables
    n,m = size(model)
    tf = times[end]
    N = length(times)
    
    mb,g = model.mb + 4*model.mf, model.g
    body_width = 2*abs(model.s1[2]) # from FL shoulder y pos wrt COM
    body_length = 2*abs(model.s1[1]) # from FL shoulder x pos wrt COM
    nft = size(modes)[1] # nft lol
    
    body_height = 0.7*model.ℓmax # const body height
    ft_height = 0.25*model.ℓmax # swing foot height
    
    nsteps = sum(modes, dims=2) # number of steps each foot takes 
    dx = (xterm - xinit) ./ nsteps # dx average dist moved for each timestep in swing
    dt_ft = tf ./ nsteps
    
    xs = range(xinit,xterm,length=N) # body xpos
    dt = tf/N
    
    # initialization
    xref = zeros(n,N)
    uref = zeros(m,N)
    
    # add/subtract half body length, for front vs back feet
    xref[8,1] = (body_length/2)
    xref[11,1] = -(body_length/2)
    xref[14,1] = (body_length/2)
    xref[17,1] = -(body_length/2)
    
    
    for k = 1:N-1
          
        xref[1,k] = xs[k] # body x
        xref[2,k] = 0 # body y
        xref[3,k] = body_height # body z
        
        # no body rotation in reference
        xref[4,k] = 1 # no rotation
        
        # foot index: (1,2,3,4) = (FL,BL,FR,BR)
        
        # foot x pos
        # pick up foot when in swing
        xinds = (8,11,14,17)
        for i = 1:nft # num feet
            if modes[i,k] == 0 # in swing
               xref[xinds[i],k+1] = xref[xinds[i],k] + dx[i]
            else
               xref[xinds[i],k+1] = xref[xinds[i],k]
            end
        end
        # would make more sense to lin interp over each swing phase
        # while checking for final swing phase, must ensure you reach xterm in final swing phase
        
        
        # foot y pos
        # keep const
        xref[9,k] = body_width / 2
        xref[12,k] = body_width / 2
        xref[15,k] = -body_width / 2
        xref[18,k] = -body_width / 2
        
        
        # foot z pos
        # pick up foot when in swing
        zinds = (10,13,16,19)
        for i = 1:nft # num feet
            if modes[i,k] == 0 # in swing
               xref[zinds[i],k] = ft_height 
            end
        end
        
        # set velocities
        # keep at 0 for k = 1
        if k > 1
            xref[20,k] = (xref[1,k+1] - xref[1,k])/dt # body x vel
            
            xref[26,k] = (xref[8,k+1] - xref[8,k])/dt_ft[1] # foot1 x vel
            xref[29,k] = (xref[11,k+1] - xref[11,k])/dt_ft[2] # foot2 x vel
            xref[32,k] = (xref[14,k+1] - xref[14,k])/dt_ft[3] # foot3 x vel
            xref[35,k] = (xref[17,k+1] - xref[17,k])/dt_ft[4] # foot4 x vel
            
            xref[28,k] = (xref[10,k+1] - xref[10,k])/dt_ft[1] # foot1 z vel
            xref[31,k] = (xref[13,k+1] - xref[13,k])/dt_ft[2] # foot2 z vel
            xref[34,k] = (xref[16,k+1] - xref[16,k])/dt_ft[3] # foot3 z vel
            xref[37,k] = (xref[19,k+1] - xref[19,k])/dt_ft[4] # foot4 z vel
        end
    end
    
    # end state
    # set terminal positions, all velocities should 0 out
    
    # body pos
    xref[1,N] = xterm
    xref[2,N] = 0
    xref[3,N] = body_height
    
    xref[4,N] = 1 # no rotation
    xref[5,N] = 0 # 5:7 already set to 0, so this is unnecessary
    xref[6,N] = 0
    xref[7,N] = 0
    
    # foot x pos
    xref[8,N] = xterm + (body_length/2)
    xref[11,N] = xterm - (body_length/2)
    xref[14,N] = xterm + (body_length/2)
    xref[17,N] = xterm - (body_length/2)

    # foot y pos
    xref[9,N] = body_width / 2
    xref[12,N] = body_width / 2
    xref[15,N] = -body_width / 2
    xref[18,N] = -body_width / 2
    
    
    # reference trajectory
    uref .= kron(ones(N)', -[0;0;0.5*mb*g; 0;0;0.5*mb*g; 0;0;0.5*mb*g; 0;0;0.5*mb*g]) # 1/2mg for each foot, ignoring foot mass
    for k = 1:N-1
        uref[[3,6,9,12],k] .*= modes[:,k]
    end
    
    # compensate for terrain height
    for k = 1:N
        xref[3,k] += getHeight(terrain, xref[1,k], xref[2,k])
        xref[10,k] += getHeight(terrain, xref[8,k], xref[9,k])
        xref[13,k] += getHeight(terrain, xref[11,k], xref[12,k])
        xref[16,k] += getHeight(terrain, xref[14,k], xref[15,k])
        xref[19,k] += getHeight(terrain, xref[17,k], xref[18,k])
    end
    
    # Convert to a trajectory
    Xref = [SVector{n}(x) for x in eachcol(xref)]
    Uref = [SVector{m}(u) for u in eachcol(uref)]
    return Xref, Uref
end