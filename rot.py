# ------------------------------------------------------------------------------
#
#                                  rot: rot1, rot2, rot3
#
#  this function performs rotations about the 1st, 2nd & 3rd axes.
#
#  author        : david vallado                  719-573-2600   27 may 2002
#
#  revisions
#  Python        : doug buettner                                 29 jan 2024
#
#     buettner   - consolidated rots into a single script        29 jan 2024
#
#  inputs          description                    range / units
#    vec         - input vector
#    xval        - angle of rotation              rad
#
#  outputs       :
#    outvec      - vector result
#
#  locals        :
#    c           - cosine of the angle xval
#    s           - sine of the angle xval
#    temp        - temporary extended value
#
#  coupling      :
#    none.
#
# [outvec] = rot1 ( vec, xval ); 1st axis rotation
# [outvec] = rot2 ( vec, xval ); 2nd axis rotation
# [outvec] = rot3 ( vec, xval ); 3rd axis rotation
# ----------------------------------------------------------------------------- }


import numpy as np

def rot1(vec, xval):
    temp = vec[2]
    c = np.cos(xval)
    s = np.sin(xval)

    outvec = np.zeros(3)
    outvec[2] = c * vec[2] - s * vec[1]
    outvec[1] = c * vec[1] + s * temp
    outvec[0] = vec[0]

    return outvec

def rot2(vec, xval):
    temp = vec[2]
    c = np.cos(xval)
    s = np.sin(xval)

    outvec = np.zeros(3)
    outvec[2] = c * vec[2] + s * vec[0]
    outvec[0] = c * vec[0] - s * temp
    outvec[1] = vec[1]

    return outvec

def rot3(vec, xval):
    temp = vec[1]
    c = np.cos(xval)
    s = np.sin(xval)

    outvec = np.zeros(3)
    outvec[1] = c * vec[1] - s * vec[0]
    outvec[0] = c * vec[0] + s * temp
    outvec[2] = vec[2]

    return outvec
