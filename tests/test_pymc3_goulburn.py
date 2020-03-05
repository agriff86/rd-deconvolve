import sys
import os
import datetime
import numpy as np

PARALLEL = False
if PARALLEL:
    from dask.distributed import Client

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(
    PROJECT_DIR, "examples", "goulburn-deconvolution", "raw-data"
)

sys.path.append(PROJECT_DIR)

import rddeconv


def main(dask_client):


    #
    # ... load/munge data
    #
    fname_raw_data = os.path.join(
        RAW_DATA_DIR, "Goulburn_Nov_2011_Internal_DB_v01_raw.csv"
    )
    df = rddeconv.load_standard_csv(fname_raw_data)
    # drop problematic first value (lld=1)
    df = df.dropna(subset=["lld"])
    df["lld"].iloc[0] = np.NaN
    df = df.dropna(subset=["lld"])

    # drop the bad data at the end of the record
    df = df.loc[: datetime.datetime(2011, 11, 10, 12)]

    # add time-varying detector parameters, in correct (SI) units
    ## background is just a nominal value - 1 count per minute
    df["background_rate"] = 60 / 3600
    df["Q_external"] = df.exflow / 1000 / 60.0

    # internal flow, convert from velocity (m/sec) to volumetric flow rate m3/sec
    # the inflow parameter is in units of m/sec, pipe diameter is 100mm
    pipe_area = np.pi * (100 / 1000 / 2.0) ** 2
    df["Q"] = df.inflow * pipe_area

    df["total_efficiency"] = 0.154

    print(df[['lld','uld']].head(10))

    parameters = {}
    parameters.update(rddeconv.standard_parameters_700L)
    parameters.update(
        dict(
            Q=0.0122,
            rs=0.95,
            lamp=1 / 180.0,
            Q_external=40.0 / 60.0 / 1000.0,
            V_delay=200.0 / 1000.0,
            V_tank=750.0 / 1000.0,
            expected_change_std=1.25,
            total_efficiency=0.154,
        )
    )

    chunksize = 48 * 2
    overlap = 12

    rddeconv.deconvolve_dataframe_in_chunks(
        df, parameters, chunksize=chunksize, Noverlap=overlap, joblib_tasks=10
    )




if __name__ == "__main__":
    try:
        if PARALLEL:
            dask_client = Client()
        else:
            dask_client = None

        main(dask_client)
    
    finally:
        if PARALLEL:
            dask_client.shutdown()
