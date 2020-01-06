""""Implements the SMOOTHN Matlab function of Damien Garcia
(http://www.biomecardio.com/matlab/smoothn_doc.html) in python.
"""

import numpy as np
import numpy.ma as ma
from numpy.linalg import norm
import scipy.optimize.lbfgsb as lbfgsb
from scipy.fftpack.realtransforms import dct, idct

import logging

# Exit codes
EXIT_SUCCESS = 0
EXIT_LIB_NOT_FOUND = -1

W_TOT_DEFAULT = 0

LOG = logging.getLogger(__name__)

def smoothn(y, nS0=10, axis=None, smoothOrder=2.0, sd=None, verbose=False,
            s0=None, z0=None, isrobust=False, w=None, s=None, max_iter=100,
            tol_z=1e-3, weightstr='bisquare'):
    """
    Robust spline smoothing for 1-D to n-D data.

    SMOOTHN provides a fast, automatized and robust discretized smoothing
    spline for data of any dimension.

    Parameters
    ----------
    y : numpy array or numpy masked array
        The data to be smoothed.

    nS0 : int, optional
        The number of samples to use when estimating the smoothing parameter.
        Default value is 10.

    smoothOrder : float, optional
        The polynomial order to smooth the function to.
        Default value is 2.0.

    sd : numpy array, optional
        Weighting of the data points in standard deviation format.
        Deafult is to not weight by standard deviation.

    verbose : { True, False }, optional
        Create extra logging during operation.

    s0 : float, optional
        Initial value of the smoothing parameter.
        Defaults to no value, being instead derived from calculation.

    z0 : float, optional
        Initial estimate of the smoothed data.

    isrobust : { False, True }
        Whether the smoothing applies the robust smoothing algorithm. This
        allows the smoothing to ignore outlier data without creating large
        spikes to fit the data.

    w : numpy array, optional
        Linear wighting to apply to the data.
        Default is to assume no linear weighting.

    s : float
        Initial smoothing parameter.
        Default is to calculate a value.

    max_iter : int, optional
        The maximum number of iterations to attempt the smoothing.
        Default is 100 iterations.

    tol_z: float, optional
        Tolerance at which the smoothing will be considered converged.
        Default value is 1e-3

    weightstr : { 'bisquare', 'cauchy', 'talworth'}, optional
        The type of weighting applied to the data when performing robust smoothing.

    Returns
    -------

    (z, s, exitflag)
        A tuple of the returned results.
    z : numpy array
        The smoothed data.
    s : float
        The value of the smoothing parameter used to perform this smoothing.
    exitflag : {0, -1}
        A return flag of 0 indicates successfuly execution, -1 an error
        (see the log).

    Notes
    -----

    Z = SMOOTHN(Y) automatically smoothes the uniformly-sampled array Y. Y
    can be any n-D noisy array (time series, images, 3D data,...). Non
    finite data (NaN or Inf) are treated as missing values.

    Z = SMOOTHN(Y,S) smoothes the array Y using the smoothing parameter S.
    S must be a real positive scalar. The larger S is, the smoother the
    output will be. If the smoothing parameter S is omitted (see previous
    option) or empty (i.e. S = []), it is automatically determined using
    the generalized cross-validation (GCV) method.

    Z = SMOOTHN(Y,w) or Z = SMOOTHN(Y,w,S) specifies a weighting array w of
    real positive values, that must have the same size as Y. Note that a
    nil weight corresponds to a missing value.

    Robust smoothing
    ----------------
    Z = SMOOTHN(...,'robust') carries out a robust smoothing that minimizes
    the influence of outlying data.

    [Z,S] = SMOOTHN(...) also returns the calculated value for S so that
    you can fine-tune the smoothing subsequently if needed.

    An iteration process is used in the presence of weighted and/or missing
    values. Z = SMOOTHN(...,OPTION_NAME,OPTION_VALUE) smoothes with the
    termination parameters specified by OPTION_NAME and OPTION_VALUE. They
    can contain the following criteria:
        -----------------
        tol_z:       Termination tolerance on Z (default = 1e-3)
                    tol_z must be in ]0,1[
        max_iter:    Maximum number of iterations allowed (default = 100)
        Initial:    Initial value for the iterative process (default =
                    original data)
        -----------------
    Syntax: [Z,...] = SMOOTHN(...,'max_iter',500,'tol_z',1e-4,'Initial',Z0);

    [Z,S,EXITFLAG] = SMOOTHN(...) returns a boolean value EXITFLAG that
    describes the exit condition of SMOOTHN:
        1       SMOOTHN converged.
        0       Maximum number of iterations was reached.

    Class Support
    -------------
    Input array can be numeric or logical. The returned array is of class
    double.

    Notes
    -----
    The n-D (inverse) discrete cosine transform functions <a
    href="matlab:web('http://www.biomecardio.com/matlab/dctn.html')"
    >DCTN</a> and <a
    href="matlab:web('http://www.biomecardio.com/matlab/idctn.html')"
    >IDCTN</a> are required.

    To be made
    ----------
    Estimate the confidence bands (see Wahba 1983, Nychka 1988).

    Reference
    ---------
    Garcia D, Robust smoothing of gridded data in one and higher dimensions
    with missing values. Computational Statistics & Data Analysis, 2010
    <a
    href="matlab:web('http://www.biomecardio.com/pageshtm/publi/csda10.pdf')"
    >PDF download</a>

    Examples:
    --------
    # 1-D example
    x = linspace(0,100,2**8);
    y = cos(x/10)+(x/50)**2 + randn(size(x))/10;
    y[[70, 75, 80]] = [5.5, 5, 6];
    z = smoothn(y); # Regular smoothing
    zr = smoothn(y,'robust'); # Robust smoothing
    subplot(121), plot(x,y,'r.',x,z,'k','LineWidth',2)
    axis square, title('Regular smoothing')
    subplot(122), plot(x,y,'r.',x,zr,'k','LineWidth',2)
    axis square, title('Robust smoothing')

    # 2-D example
    xp = 0:.02:1;
    [x,y] = meshgrid(xp);
    f = exp(x+y) + sin((x-2*y)*3);
    fn = f + randn(size(f))*0.5;
    fs = smoothn(fn);
    subplot(121), surf(xp,xp,fn), zlim([0 8]), axis square
    subplot(122), surf(xp,xp,fs), zlim([0 8]), axis square

    # 2-D example with missing data
    n = 256;
    y0 = peaks(n);
    y = y0 + rand(size(y0))*2;
    I = randperm(n^2);
    y(I(1:n^2*0.5)) = NaN; # lose 1/2 of data
    y(40:90,140:190) = NaN; # create a hole
    z = smoothn(y); # smooth data
    subplot(2,2,1:2), imagesc(y), axis equal off
    title('Noisy corrupt data')
    subplot(223), imagesc(z), axis equal off
    title('Recovered data ...')
    subplot(224), imagesc(y0), axis equal off
    title('... compared with original data')

    # 3-D example
    [x,y,z] = meshgrid(-2:.2:2);
    xslice = [-0.8,1]; yslice = 2; zslice = [-2,0];
    vn = x.*exp(-x.^2-y.^2-z.^2) + randn(size(x))*0.06;
    subplot(121), slice(x,y,z,vn,xslice,yslice,zslice,'cubic')
    title('Noisy data')
    v = smoothn(vn);
    subplot(122), slice(x,y,z,v,xslice,yslice,zslice,'cubic')
    title('Smoothed data')

    # Cardioid
    t = linspace(0,2*pi,1000);
    x = 2*cos(t).*(1-cos(t)) + randn(size(t))*0.1;
    y = 2*sin(t).*(1-cos(t)) + randn(size(t))*0.1;
    z = smoothn(complex(x,y));
    plot(x,y,'r.',real(z),imag(z),'k','linewidth',2)
    axis equal tight

    # Cellular vortical flow
    [x,y] = meshgrid(linspace(0,1,24));
    Vx = cos(2*pi*x+pi/2).*cos(2*pi*y);
    Vy = sin(2*pi*x+pi/2).*sin(2*pi*y);
    Vx = Vx + sqrt(0.05)*randn(24,24); # adding Gaussian noise
    Vy = Vy + sqrt(0.05)*randn(24,24); # adding Gaussian noise
    I = randperm(numel(Vx));
    Vx(I(1:30)) = (rand(30,1)-0.5)*5; # adding outliers
    Vy(I(1:30)) = (rand(30,1)-0.5)*5; # adding outliers
    Vx(I(31:60)) = NaN; # missing values
    Vy(I(31:60)) = NaN; # missing values
    Vs = smoothn(complex(Vx,Vy),'robust'); # automatic smoothing
    subplot(121), quiver(x,y,Vx,Vy,2.5), axis square
    title('Noisy velocity field')
    subplot(122), quiver(x,y,real(Vs),imag(Vs)), axis square
    title('Smoothed velocity field')

    See also SMOOTH, SMOOTH3, DCTN, IDCTN.

    -- Damien Garcia -- 2009/03, revised 2010/11
    Visit my <a
    href="matlab:web('http://www.biomecardio.com/matlab/smoothn.html')
    >website</a> for more details about SMOOTHN

    # Check input arguments
    error(nargchk(1,12,nargin));

    z0=None,w=None,s=None,max_iter=100,tol_z=1e-3
    """

    (y, w) = preprocessing(y, w, sd)

    sizy = y.shape

    # sort axis
    if axis is None:
        axis = tuple(np.arange(y.ndim))

    noe = y.size  # number of elements
    if noe < 2:
        return y, s, EXIT_SUCCESS, W_TOT_DEFAULT

    # ---
    # "Weighting function" criterion
    weightstr = weightstr.lower()
    # ---
    # Weights. Zero weights are assigned to not finite values (Inf or NaN),
    # (Inf/NaN values = missing data).
    is_finite = np.isfinite(y)
    nof = np.sum(is_finite)  # number of finite elements
    # ---
    # Weighted or missing data?
    isweighted = np.any(w != 1)
    # ---
    # Automatic smoothing?
    isauto = not s

    # Creation of the Lambda tensor
    lambda_ = define_lambda(y, axis)

    #  Upper and lower bound for the smoothness parameter
    s_min_bnd, s_max_bnd = smoothness_bounds(y)

    #  Initialize before iterating
    y_tensor_rank = np.sum(np.array(sizy) != 1)  # tensor rank of the y-array
    # ---
    w_tot = w
    # --- Initial conditions for z
    z = initial_z(y, z0, isweighted)
    # ---
    z0 = z
    y[~is_finite] = 0  # arbitrary values for missing y-data
    # ---
    tol = 1.0
    robust_iterative_process = True
    robust_step = 1
    nit = 0
    # --- Error on p. Smoothness parameter s = 10^p
    errp = 0.1
    # opt = optimset('TolX',errp);
    # --- Relaxation factor relaxation_factor: to speedup convergence
    relaxation_factor = 1 + 0.75 * isweighted
    # ??
    #  Main iterative process
    # ---
    xpost = init_xpost(s, s_min_bnd, s_max_bnd, isauto)

    while robust_iterative_process:
        # --- "amount" of weights (see the function GCVscore)
        aow = np.sum(w_tot) / noe  # 0 < aow <= 1
        # ---
        while tol > tol_z and nit < max_iter:
            if verbose:
                LOG.info(f"tol {tol:s} nit {nit:s}")
            nit = nit + 1
            dct_y = dctND(w_tot * (y - z) + z, f=dct)
            if isauto and not np.remainder(np.log2(nit), 1):
                # ---
                # The generalized cross-validation (GCV) method is used.
                # We seek the smoothing parameter s that minimizes the GCV
                # score i.e. s = Argmin(GCVscore).
                # Because this process is time-consuming, it is performed from
                # time to time (when nit is a power of 2)
                # ---
                # errp in here somewhere

                # bounds = [(log10(s_min_bnd),log10(s_max_bnd))]
                # args = (lambda_, aow,dct_y,is_finite,w_tot,y,nof,noe)
                # xpost,f,d = lbfgsb.fmin_l_bfgs_b(gcv,xpost,fprime=None,
                # factr=10., approx_grad=True,bounds=,bounds\
                #   args=args)

                # if we have no clue what value of s to use, better span the
                # possible range to get a reasonable starting point ...
                # only need to do it once though. nS0 is the number of samples
                # used
                if not s0:
                    ss = np.arange(nS0) * (1.0 / (nS0 - 1.0)) * (
                        np.log10(s_max_bnd) - np.log10(s_min_bnd)
                        ) + np.log10(s_min_bnd)
                    g = np.zeros_like(ss)
                    for i, p in enumerate(ss):
                        g[i] = gcv(p, lambda_, aow, dct_y, is_finite, w_tot,
                                   y, nof, noe, smoothOrder)
                    xpost = [ss[g == g.min()]]
                else:
                    xpost = [s0]
                bounds = [(np.log10(s_min_bnd), np.log10(s_max_bnd))]
                args = (lambda_, aow, dct_y, is_finite,
                        w_tot, y, nof, noe, smoothOrder)
                xpost, _, _ = lbfgsb.fmin_l_bfgs_b(gcv, xpost,
                                                   fprime=None,
                                                   factr=1e7,
                                                   approx_grad=True,
                                                   bounds=bounds,
                                                   args=args)
            s = 10 ** xpost[0]
            # update the value we use for the initial s estimate
            s0 = xpost[0]

            gamma = gamma_from_lambda(lambda_, s, smoothOrder)

            z = relaxation_factor * dctND(gamma*dct_y, f=idct) +\
                (1 - relaxation_factor) * z
            # if no weighted/missing data => tol=0 (no iteration)
            tol = isweighted * norm(z0 - z) / norm(z)

            z0 = z  # re-initialization
        exitflag = nit < max_iter

        if isrobust:  # -- Robust Smoothing: iteratively re-weighted process
            # --- average leverage
            h = np.sqrt(1 + 16.0 * s)
            h = np.sqrt(1 + h) / np.sqrt(2) / h
            h = h ** y_tensor_rank
            # --- take robust weights into account
            w_tot = w*robust_weights(y - z, is_finite, h, weightstr)
            # --- re-initialize for another iterative weighted process
            isweighted = True
            tol = 1
            nit = 0
            # ---
            robust_step = robust_step + 1
            # 3 robust steps are enough.
            robust_iterative_process = robust_step < 3
        else:
            robust_iterative_process = False  # stop the whole process

    #  Warning messages
    # ---
    if isauto:
        limit = ""
        if np.abs(np.log10(s) - np.log10(s_min_bnd)) < errp:
            limit = "lower"
        elif np.abs(np.log10(s) - np.log10(s_max_bnd)) < errp:
            limit = "upper"
        warning(
            f"smoothn:S{limit.capitalize()}Bound",
            [f"s = {s:.3f}: the {limit} bound for s has been reached. " +
             "Put s as an input variable if required."]
        )

    return z, s, exitflag, w_tot


def warning(s1, s2):
    """Combine warnings.

    Parameters
    ----------
    s1 : str
        The first half of the warning.
    s2 : array-like collection of strings
        An array of explanations, from which the first is used
    """
    LOG.warning(s1)
    LOG.warning(s2[0])


def weights_from_sd(sd):
    """Take the standard deviation values and produce a set of weights.

    Parameters
    ----------
    sd : numpy array
        array of the standard deivations of each data point.

    Returns
    -------
    numpy array
        weighting equivalents of the provide standard deviations.
    """
    sd = np.array(sd)
    mask = sd > 0.0
    w = np.zeros_like(sd)
    w[mask] = sd[mask] ** (-2)
    sd = None
    return w


def unmask_array(m, w):
    """Convert a masked numpy array (m) into an unmasked array.

    Masked points are dealt with by assigning zero
    weighting on the masked points.

    Parameters
    ----------
    m : numpy masked array
        The data array as a masked array.
    w : numpy array
        Current data weights.

    Returns
    -------
    numpy array
        The data as a regular numpy array.
    numpy array
        The updated weightings.
    """
    mask = m.mask
    y = np.array(m)
    if np.any(w is not None):
        w = np.array(w)
    w[mask] = 0.0
    y[mask] = np.nan
    return (y, w)


def preprocessing(y, w, sd):
    """Condition the algorithm inputs.

    Condition the inputs to return data and weight arrays ready to be used
    by the main algorithm.

    Parameters
    ----------
    y : numpy array
        The data array.
    w : numpy array
        The data weightings.
    sd : numpy array
        The standard deviations of data points.

    Returns
    -------
    numpy array
        The updated data array.
    numpy array
        The updated data weightings array.
    """

    if np.any(sd is not None):
        w = weights_from_sd(sd)

    if isinstance(y, ma.core.MaskedArray):
        (y, w) = unmask_array(y, w)

    # Normalize weights to a maximum of 1
    if np.any(w is not None):
        w = w/w.max()

    sizy = y.shape

    # Set the weights to unity if there are none defined
    if np.all(w is None):
        w = np.ones(sizy)

    # Unweight non-finite values
    is_finite = np.isfinite(y)
    #    nof = is_finite.sum() # number of finite elements
    w = w * is_finite
    if np.any(w < 0):
        raise ValueError("smoothn: Weights must all be >=0")

    return (y, w)


def define_lambda(y, axis):
    """Creation of the Lambda tensor.

    Lambda contains the eingenvalues of the difference matrix used in this
    penalized least squares process.

    Parameters
    ----------
    y : numpy array
        The data array.
    axis : int
        The index of the axis along with to calculate the tensor.

    Return
    ------
    numpy array
        The lambda tensor derived from the data.
    """
    axis_tuple = tuple(np.array(axis).flatten())

    lam = np.zeros(y.shape)
    for i in axis_tuple:
        siz0 = np.ones((1, y.ndim))[0].astype(int)
        siz0[i] = y.shape[i]
        lam = lam + (np.cos(np.pi*(np.arange(1, y.shape[i] + 1) - 1) /
                            y.shape[i]).reshape(siz0))

    lam = -2.0 * (len(axis_tuple) - lam)

    return lam


def smoothness_bounds(y):
    """Upper and lower bound for the smoothness parameter

    The average leverage (h) is by definition in [0 1]. Weak smoothing occurs
    if h is close to 1, while over-smoothing appears when h is near 0. Upper
    and lower bounds for h are given to avoid under- or over-smoothing. See
    equation relating h to the smoothness parameter (Equation #12 in the
    referenced CSDA paper).

    Parameters
    ----------
    y : numpy array
        The data array.

    Returns
    -------
    float
        Lower smoothness bound
    float
        Upper smoothness bound
    """
    rnk = np.sum(np.array(y.shape) != 1)
    h_min = 1e-6
    h_max = 0.99

    try:
        s_min_bnd = bound_calc(h_max, rnk)
        s_max_bnd = bound_calc(h_min, rnk)
    except ValueError:
        s_min_bnd = None
        s_max_bnd = None

    return s_min_bnd, s_max_bnd


def bound_calc(h_limit, rank):
    """Calculate smooth bound from leverage bound.

    (h/rnk)**2 = (1 + a)/( 2 a)
    a = 1/(2 (h/rnk)**2 -1)
    where a = sqrt(1 + 16 s)
    (a**2 -1)/16

    Parameters
    ----------
    h_limit : float
        Limit of step size, h.
    rank : int
        Rank of the data being smoothed.

    Return
    ------
    float
        Smoothness bound.
    """
    return np.sqrt((
                ((1 + np.sqrt(1 + 8 * h_limit ** (2.0 / rank))
                  ) / 4. / h_limit ** (2.0 / rank)
                ) ** 2 - 1) / 16.)


def initial_z(y, z0, is_weighted):
    """An initial conditions for z.

    With weighted/missing data, an initial guess is provided to ensure faster
    convergence. For that purpose, a nearest neighbor interpolation followed
    by a coarse smoothing are performed.

    For unweighted data, the initial guess is zero.
    """
    if is_weighted:
        if z0 is not None:  # an initial guess (z0) has been provided
            z = z0
        else:
            z = np.where(np.isfinite(y), y, 0.0)  # initial_guess(y,IsFinite);
    else:
        z = np.zeros_like(y)

    return z


def init_xpost(s, s_min_bnd, s_max_bnd, is_auto):
    """Calculate xpost based on the smoothing and smoothing bounds values.

    Parameters
    ----------
    s : float
        Current smoothness.
    s_min_bnd : float
        Lower smoothness bound.
    s_max_bnd : float
        Upper smoothness bound.
    is_auto : boolean
        Auto smoothing is enabled.

    Returns
    -------
    numpy array of one element
        xpost
    """
    if is_auto:
        try:
            return np.array([(0.9 * np.log10(s_min_bnd) +
                              np.log10(s_max_bnd) * 0.1)])
        except ValueError:
            return np.array([100.0])
    else:
        return np.array([np.log10(s)])


#  GCV score
# ---
# function GCVscore = gcv(p)
def gcv(p, lambda_v, aow, dct_y, is_finite, w_tot, y, nof, noe, smooth_order):
    """Search the smoothing parameter s that minimizes the GCV score.

    Parameters
    ----------
    p : float
        p value.
    lambda_v : numpy array
        Lambda eigenvalue tensor.
    aow : float
        Variation in the values of the weights.
    dct_y : numpy array
        No clue
    is_finite : array-like of booleans
        Array denoting where the data values are finite.
    w_tot : float
        Total of all weights.
    y : numpy array
        Data array.
    nof : int
        Number of fs.
    noe : int
        Number of es.
    smooth_order : int
        Smoothing order

    Returns
    -------
    float
        gcv score for the current smoothing.
    """
    s = 10 ** p
    gamma = gamma_from_lambda(lambda_v, s, smooth_order)
    # --- rss = Residual sum-of-squares
    if aow > 0.9:  # aow = 1 means that all of the data are equally weighted
        # very much faster: does not require any inverse DCT
        rss = norm(dct_y * (gamma - 1.0)) ** 2
    else:
        # take account of the weights to calculate rss:
        yhat = dctND(gamma*dct_y, f=idct)
        rss = norm(np.sqrt(w_tot[is_finite]) *
                   (y[is_finite] - yhat[is_finite])) ** 2
    # ---
    tr_h = np.sum(gamma)
    gcv_score = rss / float(nof) / (1.0 - tr_h / float(noe)) ** 2
    return gcv_score


#  Robust weights
# function W = robust_weights(r,I,h,wstr)
def robust_weights(r, i, h, wstr):
    """Recalculate the weights for robust smoothing.

    Parameters
    ----------
    r : numpy array
        r
    i : int
        index of r
    h : numpy array
        step size
    wstr : {"bisquare", "cauchy", "talworth"}
        name of the method used to recalculate the weights

    Returns
    -------
    numpy array
        Recalculated weights

    References
    ----------
    Peter J. Rousseeuw & Christophe Croux (1993) Alternatives to the Median
    Absolute Deviation, Journal of the American Statistical Association,
    88:424, 1273-1283, DOI: 10.1080/01621459.1993.10476408

    Richard M. Heiberger & Richard A. Becker (1992) Design of an S Function
    for Robust Regression Using Iteratively Reweighted Least Squares, Journal
    of Computational and Graphical Statistics, 1:3, 181-196,
    DOI: 10.1080/10618600.1992.10474580


    """
    b_stddev = 1.4826 # mad to standard deviation multiplier (Rousseeuw & Croux, 1991)
    mad = np.median(np.abs(r[i] - np.median(r[i])))  # median absolute deviation
    u = np.abs(r / (b_stddev * mad) / np.sqrt(1 - h))  # studentized residuals
    if wstr == "cauchy":
        c = 2.385 # Cauchy weighting coefficient (Heiberger & Becker, 1992)
        w = 1.0 / (1 + (u / c) ** 2)  # Cauchy weights
    elif wstr == "talworth":
        c = 2.795 # Talworth weighting coefficient (Heiberger & Becker, 1992)
        w = u < c  # Talworth weights
    else:
        c = 4.685 # Bisquare weighting coefficient (Heiberger & Becker, 1992)
        w = (1 - (u / c) ** 2) ** 2.0 * ((u / c) < 1)  # bisquare weights

    w[np.isnan(w)] = 0
    return w


#  Initial Guess with weighted/missing data
# function z = initial_guess(y,i)
def initial_guess(y, i):
    """Generate an initial guess using nearest neighbour interpolation.

    Parameters
    ----------
    y : numpy array
        Data array.
    i : numpy boolean array
        Boolean array indicating finite data values.

    Returns
    -------
    numpy array
        Initial guess for the smoothed data value.
    """
    # -- nearest neighbor interpolation (in case of missing values)
    if np.any(~i):
        try:
            from scipy.ndimage.morphology import distance_transform_edt

            # if license('test','image_toolbox')
            # [z,ell] = bwdist(i);
            ell = distance_transform_edt(1 - i)
            z = y
            z[~i] = y[ell[~i]]
        except ArithmeticError:
            # If BWDIST does not exist, NaN values are all replaced with the
            # same scalar. The initial guess is not optimal and a warning
            # message thus appears.
            z = y
            z[~i] = np.mean(y[i])
    else:
        z = y
    # coarse fast smoothing
    z = dctND(z, f=dct)
    k = np.array(z.shape)
    m = np.ceil(k / 10) + 1
    d = []
    for j, kj in enumerate(k):
        d.append(np.arange(m[j], kj))
    d = np.array(d).astype(int)
    z[d] = 0.0
    z = dctND(z, f=idct)
    return z


def dctND(data, f=dct):
    """One to four dimensional application of the function f.

    The function f defaults to scipy.fftpack.realtransforms.dct.
    The dimensionality of the function is easy to that of data.

    Parameters
    ----------
    data : numpy array
        The data to apply the function to.
    f : function-like, optional
        The function to appply to the data.

    Returns
    -------
    numpy array
        The data post-application of the function.
    """
    nd = len(data.shape)
    if nd == 1:
        return f(data, norm="ortho", type=2)
    if nd == 2:
        return f(f(data, norm="ortho", type=2).T, norm="ortho", type=2).T
    if nd == 3:
        return f(f(f(data, norm="ortho", type=2, axis=0),
                   norm="ortho", type=2, axis=1),
                 norm="ortho", type=2, axis=2)
    if nd == 4:
        return f(f(f(f(data, norm="ortho", type=2, axis=0),
                     norm="ortho", type=2, axis=1),
                   norm="ortho", type=2, axis=2),
                 norm="ortho", type=2, axis=3)


def gamma_from_lambda(lambda_, s, smooth_order):
    """Calculate the gamma value.
    
    Calculate the gamma value from the lambda tensor, the current smoothing
    parameter and the smoothing order.

    Parameters
    ----------
    lambda_ : numpy array
        The lambda tensor derived from the input data.
    s : float
        The current smoothing parameter.
    smooth_order : int
        The smoothing order to apply to the data.

    Returns
    -------
    float
        The gamma parameter for the current smoothing.
    """
    return 1.0 / (1 + (s * np.abs(lambda_)) ** smooth_order)