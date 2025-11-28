from pathlib import Path
import pandas as pd
from tpx3awkward._utils import raw_as_numpy, ingest_raw_data

data_dir = Path('/home/hades/projects/TimeVaryingDiffuser/Experiment_controll/data_timepix/Measurement_Nov_21_2025_16h20m39s/raw')
#data_dir = Path('/home/hades/projects/TimeVaryingDiffuser/Experiment_controll/data_timepix/Measurement_Nov_21_2025_15h53m29s/raw')

filename_sufix = 'fGKd_000000'

tmp = []

# csv = data_dir / f'{filename_sufix}.csv'
# tmp.append(pd.read_csv(csv))
# ref = pd.concat(tmp).reset_index()

fname = data_dir / f'{filename_sufix}.tpx3'

d = raw_as_numpy(fname)
test = pd.DataFrame(ingest_raw_data(d)).reset_index()


x=1


# ref_sorted = ref.sort_values(['#ToA'])
#
#
# assert (test_sorted['x'].values == ref_sorted['#Col'].values).all()
# assert (test_sorted['y'].values == ref_sorted['#Row'].values).all()
# assert (test_sorted['ToT'].values == ref_sorted['#ToT[arb]'].values).all()
# assert (test_sorted['timestamp'].values == ref_sorted['#ToA'].values).all()