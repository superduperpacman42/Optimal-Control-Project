function contact_sequence_trot(times, step_length=5)
    # 1 = stance, 0 = swing
    # 4xN matrix encoding contact schedule
    
    # initialize
    contact_schedule = zeros(4,length(times))

    #hardcoded a trot
    for k = 1:N
        if (k-1) % (2*step_length) < step_length
            contact_schedule[:,k] .= [1;0;0;1]
        else
            contact_schedule[:,k] .= [0;1;1;0]
        end
    end
    return contact_schedule
end