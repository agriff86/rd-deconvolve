"""
Main deconvolution routine
"""
import logzero
import pandas as pd
import pymc3 as pm

from . import pymc3_deconvolve

logger = logzero.logger


def overlapping_chunk_dataframe_iterator(df, chunksize, overlap=0, minsize=1):
    """
    A generator which produces an iterator over a dataframe with overlapping chunks
    """
    ix0 = 0
    ixstart = ix0 + overlap
    ixend = ixstart + chunksize
    ix1 = ixend + overlap
    while ix1 <= len(df):
        yield df.iloc[ix0:ix1]
        ix0 += chunksize
        ixstart += chunksize
        ixend += chunksize
        ix1 += chunksize
    # return the end chunk - likely shorter than the others
    ix1 = len(df)
    if ix1 > ix0 + minsize:
        yield df.iloc[ix0:ix1]
    return


def get_overlapping_chunks(df: pd.DataFrame, chunksize: int, overlap: int = 0):
    dfl = list(
        overlapping_chunk_dataframe_iterator(df, chunksize=chunksize, overlap=overlap)
    )

    # for ii, df in enumerate(dfl):
    #    # add diagnostic columns
    #    df = df.copy()
    #    df["chunk_number"] = ii
    #    df["overlap_region"] = False
    #    colidx = df.columns.get_loc("overlap_region")
    #    df.iloc[:overlap, colidx] = True
    #    df.iloc[-overlap:, colidx] = True

    return dfl


def add_chunk_metadata(ds, chunk_id, Noverlap):
    overlap_data = np.zeros(len(ds.time), dtype=np.int8)
    overlap_data[:Noverlap] = 1
    overlap_data[-Noverlap:] = 1
    ds["overlap_flag"] = xr.DataArray(overlap_data, dims=["time"])
    ds["overlap_flag"].attrs["long_name"] = "Flag for overlap with adjacent chunk"
    ds["chunk_id"] = chunk_id
    ds = ds.expand_dims(dim={"chunk_id": 1})
    return ds


def deconvolve_dataframe_in_chunks(
    df,
    detector_params,
    mcmc_backend="pymc3",
    chunksize=None,
    Noverlap=0,
    dask_client=None,
    fname_base="./deconvolution_result",
    njobs=None,
    joblib_tasks=1
):

    dfl = get_overlapping_chunks(df, chunksize, Noverlap)
    logger.info(f'Dataset divided into {len(dfl)} chunks')

    joblib = joblib_tasks > 1
    if joblib:
        logger.info(f'Running in parallel using joblib with {joblib_tasks} tasks')
    if joblib and dask_client is not None:
        raise ValueError("Can't have joblib True when dask_client is not None")

    if njobs is None:
        # by default, run sampler in series under dask or joblib
        if dask_client is None:
            njobs = 4
        else:
            njobs = 1
        
    if joblib:
        # joblib is used by pymc3, needs to be suppressed if we're using joblib here
        njobs = 1
    
    logger.info(f'MCMC sampling will be carried out with {njobs} jobs')
    constant_kwargs = {"detector_params": detector_params, "mcmc_backend": mcmc_backend, "njobs":njobs}


    fnames = []
    def run_task(dfss, chunk_id):
        ds_trace = deconvolve_dataframe(dfss, **constant_kwargs)
        ds_output = add_chunk_metadata(ds_trace, chunk_id, Noverlap)
        fname = f"{fname_base}_chunk{chunk_id:04}.nc"
        logger.info(f'Writing MCMC samples to {fname}')
        ds_output.to_netcdf(fname)
        return fname
       
    fnames = []
    for chunk_id, dfss in enumerate(dfl):
        if joblib:
            from joblib import delayed
            fname = delayed(run_task)(dfss, chunk_id)
        elif dask_client is None:
            fname = run_task(dfss, chunk_id)
        elif dask_client is not None:
            fname = dask_client.submit(run_task, dfss, chunk_id)
        else:
            assert(False)

        fnames.append(fname)
    
    if joblib:
        from joblib import Parallel
        fnames = Parallel(n_jobs=joblib_tasks)(fnames)

    if dask_client is not None:
        fnames = dask_client.gather(fnames)
    
    return fnames


def deconvolve_dataframe(df, detector_params, mcmc_backend="pymc3", Nsamples=1000, njobs=None):

    logger.info(f"Processing data from {df.index[0]} to {df.index[-1]}")
    if mcmc_backend == "pymc3":
        # TODO: validate detector parameters

        dp = {}
        dp.update(detector_params)

        time = (df.index - df.index.values[0]).total_seconds().values
        time += time[1]
        counts = (df.lld - df.uld).values

        # extract parameters from data frame, if provided
        dataframe_vars = ["background_rate", "Q", "Q_external", "total_efficiency"]
        for vn in dataframe_vars:
            if vn in df.columns:
                dp[vn] = df[vn].mean()
                logger.info(f"Using {vn} = {dp[vn]} from timeseries input")

        deconv_result = pymc3_deconvolve.fit_model_to_obs(
            time, counts, detector_params=dp, Nsamples=Nsamples, njobs=njobs
        )
        raw_trace = deconv_result['trace']
        map_estimate = deconv_results['map_estimate']
        num_divergences = raw_trace.get_sampler_stats('diverging').sum()
        if num_divergences > 0:
            logger.error(f'There are {num_divergences} divergences in the samples')

        trace = pymc3_deconvolve.trace_as_xarray(
            index_time=df.index, trace=raw_trace, Noverlap=Noverlap
        )
        # TODO: add map_estimate to trace?
        # TODO: add summary to trace?

    else:
        raise NotImplementedError(f"no backend for `mcmc_backend={mcmc_backend}`")
    
    return trace