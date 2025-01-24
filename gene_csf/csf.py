from itertools import permutations
"""
WORK IN PROGRESS
code to create CSFs using the  genealogical coupling method
"""
def generate_arrangements(n, S):
    # Calculate the number of ones and zeros required to satisfy S
    num_ones = (n + S) // 2
    num_zeros = n - num_ones

    if num_ones < 0 or num_zeros < 0 or (n + S) % 2 != 0:
        raise ValueError("Invalid inputs: n and S do not produce a valid list of 0s and 1s.")

    # Create the input list with the calculated number of zeros and ones
    elements = [0] * num_zeros + [1] * num_ones

    # Use set to remove duplicates as permutations might repeat identical elements
    arrangements = set(permutations(elements))

    # Convert each arrangement back to a list for display or further processing
    return [list(arr) for arr in arrangements]

# Example usage
n = 6  # Total number of elements
S = 0  # Desired sum (+1 for 1, -1 for 0)
arrangements = generate_arrangements(n, S)
for arrangement in arrangements:
    print(arrangement)

def coupling_coeff(S,M,pm,sigma):
    if pm ==0.5:
        return np.sqrt((S+2*sigma*M)/(2*S))
    elif pm == -0.5:
        return np.sqrt((S+1-2*sigma*M)/(2*(S+1)))

def transform_arrangement_det(arrangement):
    alpha_vector = []
    beta_vector = []
    pf = 1.0
    for element in arrangement:
        if element == 1 or element =="+":
            alpha_vector.append(1)
            beta_vector.append(0)
        if element == 0 or element=="-":
            alpha_vector.append(0)
            beta_vector.append(1)
    return [pf] + alpha_vector + beta_vector


def transform_coupling_Svec(coupling):
    Svec = []
    S = 0
    for sign in coupling:
        if sign=="+":
            S += 0.5 
            Svec.append(S)
        elif sign=="-":
            S -= 0.5 
            Svec.append(S)
    return Svec

coupling = "+-+-"
svec = transform_coupling_Svec("+-+-")
det = transform_arrangement_det(coupling)
print(svec)
print(det)



    
def csf(S,M,coupling):


