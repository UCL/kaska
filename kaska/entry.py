from .kaska import KaSKA, define_temporal_grid
from .s2_observations import Sentinel2Observations

def run_process(start_date, end_date, temporal_grid_space, parent_folder, state_mask):
    import pkgutil
    from io import BytesIO

    temporal_grid = define_temporal_grid(start_date, end_date,
                                        temporal_grid_space)
    nn_inverter = pkgutil.get_data("kaska",
                    "inverters/prosail_2NN.npz")
    s2_obs = Sentinel2Observations(
        parent_folder,
        BytesIO(nn_inverter),
        state_mask,
        band_prob_threshold=20,
        chunk=None,
        time_grid=temporal_grid,
    )
    approx_inverter = BytesIO(pkgutil.get_data("kaska",
                    "inverters/Prosail_5_paras.h5"))
    kaska = KaSKA(s2_obs, temporal_grid, state_mask, approx_inverter,
                     "/tmp/")
    slai, scab, scbrown = kaska.run_retrieval()
