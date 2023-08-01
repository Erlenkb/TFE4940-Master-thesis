
"""
The following functions are used to alter the TLM matrix
"""


"""
    _real_to_index(val, Δd)

Takes in the real coordinates in the x,y,z planes and return the 
indicies position given the step value `Δd`.
"""
function _real_to_index(val::Array{Float64,1}, Δd)
    ind = [Int(div(val[1],Δd)), Int(div(val[2],Δd)), Int(div(val[3],Δd))]
    return ind
end




"""
    case(n)

Computes the used cased for the labeled_tlm matrix. It takes in a 
sorted integer containing values from 0 to 123456. Returns the 
used and unused numbers, as well as a boolean value 'Diffusor' if the 
case applies to the diffusor or not.
"""
function case(n::Int64)
    Diffusor = false
    if n < 0
        Diffusor = true
        n = abs(n)
    end
    num_str = string(n)
    used_nums = [parse(Int, digit) for digit in num_str]
    all_nums = Set(1:6)
    unused_nums = setdiff(collect(all_nums), used_nums)

    return used_nums, unused_nums, Diffusor
end




"""
    merge(A,B)

Takes in two integer values used for the labaled TLM matrix, and fuse the two together.
"""
function merge(A::Int, B::Int)::Int
    # Concatenate A and B as strings
    str = string(A, B)

    # Remove duplicates and sort the remaining characters
    unique_chars = sort(unique(str))

    # Convert the sorted string back to an integer and return it
    result = 0
    for c in unique_chars
        result = result * 10 + (c - '0')
    end
    return result
end




