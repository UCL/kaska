import datetime as dt

from .kaska import KaSKA, define_temporal_grid
from .s2_observations import Sentinel2Observations

def run_process():
    import pkgutil
    from io import BytesIO

    start_date = dt.datetime(2017, 5, 1)
    end_date = dt.datetime(2017, 9, 1)
    temporal_grid_space = 5
    temporal_grid = define_temporal_grid(start_date, end_date,
                                        temporal_grid_space)
    nn_inverter = pkgutil.get_data("kaska",
                    "inverters/prosail_2NN.npz")
    s2_obs = Sentinel2Observations(
        "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/s2_obs/",
        BytesIO(nn_inverter),
        "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif",
        band_prob_threshold=20,
        chunk=None,
        time_grid=temporal_grid,
    )
    state_mask = "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif"
    approx_inverter = BytesIO(pkgutil.get_data("kaska",
                    "inverters/Prosail_5_paras.h5"))
    kaska = KaSKA(s2_obs, temporal_grid, state_mask, approx_inverter,
                     "/tmp/")
    slai, scab, scbrown = kaska.run_retrieval()
