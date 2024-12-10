import itertools

def combine_beliefs(belief1, belief2):
    combined_belief = {}
    conflict_mass = 0  

    
    for A, B in itertools.product(belief1.items(), belief2.items()):
        A_set, A_mass = A
        B_set, B_mass = B

        intersection = A_set & B_set
        if intersection:
            combined_belief[intersection] = combined_belief.get(intersection, 0) + A_mass * B_mass
        else:
            conflict_mass += A_mass * B_mass

    
    if conflict_mass < 1:
        for subset in combined_belief:
            combined_belief[subset] /= (1 - conflict_mass)
    else:
        for A_set, A_mass in belief1.items():
            combined_belief[A_set] = combined_belief.get(A_set, 0) + 0.8 * A_mass
        for B_set, B_mass in belief2.items():
            combined_belief[B_set] = combined_belief.get(B_set, 0) + 0.2 * B_mass
        print(combined_belief.get(A_set, 0) + 0.8 * A_mass)

    return combined_belief









