def able_body_weight_formula(sex=str, volume=float, height=None):
    print(sex)
    
    if sex == "female":
        sex_weight = 2
    elif sex == "male":
        sex_weight = 1
    else:
        raise ValueError("sex input must be MALE or FEMALE")
    
    if height is None:
        weight = 0.955 * volume - 2.24* sex_weight + 7.15

    else:
        weight = 0.950 * volume  - 1.57 * sex_weight  + 0.057 * height - 2.92
    
    print(f"The estimated weight is: {weight:.2f} kg or {weight*2.20462:.2f} lbs")
    return weight

