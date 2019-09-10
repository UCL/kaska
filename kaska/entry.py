import datetime as dt
from .kaska import KaSKA, define_temporal_grid
from .inference_runner import kaska_runner
from .logger import create_logger
from .inverters import get_emulator, get_inverter


def run_process(start_date, end_date, temporal_grid_space, s2_folder, 
                state_mask, output_folder, debug=True, logfile=None):
    import pkgutil
    from io import BytesIO

    if logfile is None:
        logfile = f"./KaSKA_{int(dt.datetime.now().timestamp()):d}.log"
    LOG = create_logger(debug=debug, fname=logfile)
    LOG.info("Running KaSKA...")
    temporal_grid = define_temporal_grid(start_date, end_date,
                                        temporal_grid_space)
    s2_emulator = get_emulator("prosail", "Sentinel2")
    approx_inverter = get_inverter("prosail_5paras", "Sentinel2")
    
    kaska_runner(start_date, end_date, temporal_grid_space, state_mask,
                s2_folder, approx_inverter, s2_emulator, output_folder)


