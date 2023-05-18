

function replace_value_PN(PN, SS, LBL, I)
    # Find all positions of value I in LBL
    positions = findall(x -> x == I, LBL)

    # Iterate over the positions
    for pos in positions
        row, col, depth = pos.I  # Access CartesianIndex components directly
        println(pos.I)

        # Replace the value in PN with the corresponding value from SS
        PN[row, col, depth] = SS[row, col-1, depth]
    end
    
    return PN
end


function replace_value_PE(PE, SW, LBL, I)
    # Find all positions of value I in LBL
    positions = findall(x -> x == I, LBL)

    # Iterate over the positions
    for pos in positions
        row, col, depth = pos.I  # Access CartesianIndex components directly
        println(pos.I)

        # Replace the value in PN with the corresponding value from SS
        PE[row, col, depth] = SW[row+1, col, depth]
    end
    return PE
end


function replace_value_PE(PS, SN, LBL, I)
    # Find all positions of value I in LBL
    positions = findall(x -> x == I, LBL)

    # Iterate over the positions
    for pos in positions
        row, col, depth = pos.I  # Access CartesianIndex components directly
        println(pos.I)

        # Replace the value in PN with the corresponding value from SS
        PS[row, col, depth] = SN[row, col+1, depth]
    end
    
    return PS
end


function replace_value_PE(PW, SE, LBL, I)
    # Find all positions of value I in LBL
    positions = findall(x -> x == I, LBL)

    # Iterate over the positions
    for pos in positions
        row, col, depth = pos.I  # Access CartesianIndex components directly
        println(pos.I)

        # Replace the value in PN with the corresponding value from SS
        PN[row, col, depth] = SS[row, col-1, depth]
    end
    
    return PN
end

function replace_value_PE(PN, SS, LBL, I)
    # Find all positions of value I in LBL
    positions = findall(x -> x == I, LBL)

    # Iterate over the positions
    for pos in positions
        row, col, depth = pos.I  # Access CartesianIndex components directly
        println(pos.I)

        # Replace the value in PN with the corresponding value from SS
        PN[row, col, depth] = SS[row, col-1, depth]
    end
    
    return PN
end







PN = cat([1 2 3; 5 4 2; 3 5 6],
             [1 5 3; 3 1 7; 8 0 1],
             [1 3 6; 3 7 7; 8 8 2], dims=3)

SS = cat([10 20 30; 50 40 20; 30 50 60],
             [10 50 30; 30 10 70; 80 0 10],
             [10 30 60; 30 70 70; 80 80 20], dims=3)

LBL = cat([1 2 3; 5 4 2; 3 5 6],
             [1 5 3; 3 1 7; 8 0 1],
             [1 3 6; 3 7 7; 8 8 8], dims=3)

PN = replace_values(PN, SS, LBL, 6)
