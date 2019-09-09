'''
Test the water cloud model


'''

import pytest
import numpy as np


from ..watercloudmodel  import wcm, wcm_jac, wcm_hess
from ..watercloudmodel import cost, cost_jac, cost_hess


def test_wcm():
    """Test the WCM mode forward operation."""
    A, B, C, V1, V2, sigma_soil = -12,  0.05, 0.02, np.ones(2)*4, np.ones(2)*4, np.ones(2)*0.1
    x = np.r_[A, B, C, V1, V2, sigma_soil]
    retval = wcm(x)
    expected = np.array([-17.67969331, -17.67969331])
    assert np.allclose( retval, expected)



def test_wcm_jac():
    """Test the Jacobian. Remember that the Jacobian returned by wcm_jac is
    incomplete without the V1 and V2 terms, and that we need to add up the
    A, B and C terms. The sigma_soil terms are independent"""
    A, B, C, V1, V2, sigma_soil = -12,  0.05, 0.02, np.ones(2)*4, np.ones(2)*4, np.ones(2)*0.1
    x = np.r_[A, B, C, V1, V2, sigma_soil]
    jj = wcm_jac(x)
    retval = np.array([jj[0].sum(), jj[1].sum(), jj[2].sum(), *(jj[3])])
    expected = np.array([2.959217508268818, -559.9411675286623,
                        1.2601956229327955, 0.63009781, 0.63009781])
    assert np.allclose( retval, expected)

def test_cost():
    Avv, Bvv, Cvv, V1, V2, sigma_soil = -12,  0.05, 0.02, np.ones(2)*4, np.ones(2)*4, np.ones(2)*0.1
    Avh, Bvh, Cvh = -14, 0.01, 0.1
    x = np.r_[Avv, Bvv, Cvv, Avh, Bvh, Cvh, V1, V2, sigma_soil]

    svv = np.array([-17.67969331,-17.67969331])
    svh = np.array([-4.75896306,-4.75896306])
    lai = np.array([4, 4.])
    retval  = cost(x, svh, svv, theta=30)
    assert np.allclose(retval, 0, atol=1e-10)

def test_cost_jac():
    Avv, Bvv, Cvv, V1, V2, sigma_soil = -12,  0.05, 0.02, np.ones(2)*4, np.ones(2)*4, np.ones(2)*0.1
    Avh, Bvh, Cvh = -14, 0.01, 0.1
    x = np.r_[Avv, Bvv, Cvv, Avh, Bvh, Cvh, V1, V2, sigma_soil]

    svv = np.array([-17.67969331,-17.67969331])
    svh = np.array([-4.75896306,-4.75896306])
    lai = np.array([4, 4.])
    ###cost(x, svv, svh, theta=30 )
    # Check the Jacobians
    # 
    expected = np.array([-2.65e-08,  5.01e-06, -1.13e-08,  1.18e-08,
                        -1.57e-05,  3.04e-08,  9.55e-09, 9.55e-09])
    retval = cost_jac(x, svh, svv, theta=30)
    print(retval-expected)
    # Could also test with respect to 0
    assert np.allclose(retval, expected, atol=1e-6)


def test_optimiser():
    import scipy.optimize

    Avv, Bvv, Cvv, V1, V2, sigma_soil = -12,  0.05, 0.02, np.ones(2)*4, np.ones(2)*4, np.ones(2)*0.1
    Avh, Bvh, Cvh = -14, 0.01, 0.1

    x = np.r_[Avv, Bvv, Cvv, Avh, Bvh, Cvh, sigma_soil]#, V1, V2, sigma_soil]

    svv = np.array([-17.67969331,-17.67969331])
    svh = np.array([-4.75896306,-4.75896306])
    svv = np.array([-17.4,-17.9])
    svh = np.array([-4.1,-5.2])

    lai = np.array([4, 4.])
    theta = 30

    def cost_nolai(xx, svh, svv, lai, theta):
        return cost(np.concatenate([xx[:6], lai,lai, x[-2:]]), svh,svv,theta)

    def cost_nolai_jac(xx, svh, svv, lai, theta):
        return cost_jac(np.concatenate([xx[:6], lai,lai, x[-2:]]), svh,svv,theta)


    def cost_nolai_hess(xx, svh, svv, lai, theta):
        return cost_hess(np.concatenate([xx[:6], lai, lai,x[-2:]]), svh,svv,theta)

    x0 = np.array([-10, 0.1, 1, -10, 0.1, 1, 0.2, 0.2])
    
    retval = scipy.optimize.minimize(cost_nolai, x0, args=(svh, svh, lai,theta), 
                                 jac=cost_nolai_jac, hess=cost_nolai_hess,
                                method="Newton-CG")
                                
    assert np.allclose(retval.fun, 2.42, rtol=1e-2 )
    
    


