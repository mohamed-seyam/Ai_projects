import numpy as np 

def chads_vasc_score(input_c, input_h, input_a2, input_d, input_s2, input_v, input_a, input_sc):
    # congestive heart failure
    coef_c = 1 
    
    # Coefficient for hypertension
    coef_h = 1 
    
    # Coefficient for Age >= 75 years
    coef_a2 = 2
    
    # Coefficient for diabetes mellitus
    coef_d = 1
    
    # Coefficient for stroke
    coef_s2 = 2
    
    # Coefficient for vascular disease
    coef_v = 1
    
    # Coefficient for age 65 to 74 years
    coef_a = 1
    
    # Coefficient for female
    coef_sc = 1
    
    # Calculate the risk score
    risk_score = (input_c * coef_c) +\
                 (input_h * coef_h) +\
                 (input_a2 * coef_a2) +\
                 (input_d * coef_d) +\
                 (input_s2 * coef_s2) +\
                 (input_v * coef_v) +\
                 (input_a * coef_a) +\
                 (input_sc * coef_sc)
    
    return risk_score

def liver_disease_mortality(input_creatine, input_bilirubin, input_inr):
    """
    Calculate the probability of mortality given that the patient has
    liver disease. 
    Parameters:
        Creatine: mg/dL
        Bilirubin: mg/dL
        INR: 
    """
    # Coefficient values
    coef_creatine = 0.957
    coef_bilirubin = 0.378
    coef_inr = 1.12
    intercept = 0.643
    # Calculate the natural logarithm of input variables
    log_cre = np.log(input_creatine)
    log_bil = np.log(input_bilirubin)
    
    # Calculate the natural log of input_inr
    log_inr = np.log(input_inr)
    
    # Compute output
    meld_score = (coef_creatine * log_cre) +\
                 (coef_bilirubin * log_bil ) +\
                 (coef_inr * log_inr) +\
                 intercept
    
    # Multiply meld_score by 10 to get the final risk score
    meld_score = meld_score * 10 
    
    return meld_score

def ascvd(x_age,
          x_cho,
          x_hdl,
          x_sbp,
          x_smo,
          x_dia,
          verbose=False
         ):
    """
    Atherosclerotic Cardiovascular Disease
    (ASCVD) Risk Estimator Plus
    """
    
    # Define the coefficients
    b_age = 17.114
    b_cho = 0.94
    b_hdl = -18.92
    b_age_hdl = 4.475
    b_sbp = 27.82
    b_age_sbp = -6.087
    b_smo = 0.691
    b_dia = 0.874
    
    # Calculate the sum of the products of inputs and coefficients
    sum_prod =  b_age * np.log(x_age) + \
                b_cho * np.log(x_cho) + \
                b_hdl * np.log(x_hdl) + \
                b_age_hdl * np.log(x_age) * np.log(x_hdl) +\
                b_sbp * np.log(x_sbp) +\
                b_age_sbp * np.log(x_age) * np.log(x_sbp) +\
                b_smo * x_smo + \
                b_dia * x_dia
    
    if verbose:
        print(f"np.log(x_age):{np.log(x_age):.2f}")
        print(f"np.log(x_cho):{np.log(x_cho):.2f}")
        print(f"np.log(x_hdl):{np.log(x_hdl):.2f}")
        print(f"np.log(x_age) * np.log(x_hdl):{np.log(x_age) * np.log(x_hdl):.2f}")
        print(f"np.log(x_sbp): {np.log(x_sbp):2f}")
        print(f"np.log(x_age) * np.log(x_sbp): {np.log(x_age) * np.log(x_sbp):.2f}")
        print(f"sum_prod {sum_prod:.2f}")
        
    # Risk Score = 1 - (0.9533^( e^(sum_prod - 86.61) ) )
    risk_score =  1 - (0.9533 ** ( np.e**(sum_prod - 86.61))) 
    
    return risk_score