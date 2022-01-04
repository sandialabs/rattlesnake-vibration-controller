import numpy as np

def pseudoinverse_control(transfer_function,  # Transfer Functions
                          specification, # Specifications
                          buzz_cpsd, # Buzz test in case cross terms are to be computed
                          extra_parameters = '',
                          last_response_cpsd = None, # Last Response for Error Correction
                          last_output_cpsd = None, # Last output for Drive-based control
                          ):
    # Invert the transfer function using the pseudoinverse
    tf_pinv = np.linalg.pinv(transfer_function)
    # Return the least squares solution for the new output CPSD
    return tf_pinv@specification@tf_pinv.conjugate().transpose(0,2,1)

def pseudoinverse_control_generator():
    output_cpsd = None
    while True:
        (transfer_function,specification,buzz_cpsd,extra_parameters,
            last_response_cpsd,last_output_cpsd) = yield output_cpsd
         # Invert the transfer function using the pseudoinverse
        tf_pinv = np.linalg.pinv(transfer_function)
        # Assign the output_cpsd so it is yielded next time through the loop
        output_cpsd = tf_pinv@specification@tf_pinv.conjugate().transpose(0,2,1)