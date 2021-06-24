import numpy as np
import pandas as pd


def HIV_NRTI(drug='3TC', 
             standardize=True, 
             datafile=None,
             min_occurrences=11):
    """
    Download 
        http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt
    and return the data set for a given NRTI drug.
    The response is an in vitro measurement of log-fold change 
    for a given virus to that specific drug.

    Parameters
    ----------
    drug : str (optional)
        One of ['3TC', 'ABC', 'AZT', 'D4T', 'DDI', 'TDF']
    standardize : bool (optional)
        If True, center and scale design X and center response Y.
    datafile : str (optional)
        A copy of NRTI_DATA above.
    min_occurrences : int (optional)
        Only keep positions that appear
        at least a minimum number of times.
        
    """

    if datafile is None:
        datafile = "http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt"
    NRTI = pd.read_table(datafile, na_values="NA")

    NRTI_specific = []
    NRTI_muts = []
    for i in range(1,241):
        d = NRTI['P%d' % i]
        for mut in np.unique(d):
            if mut not in ['-','.'] and len(mut) == 1:
                test = np.equal(d, mut)
                if test.sum() >= min_occurrences:
                    NRTI_specific.append(np.array(np.equal(d, mut))) 
                    NRTI_muts.append("P%d%s" % (i,mut))

    NRTI_specific = NRTI.from_records(np.array(NRTI_specific).T, columns=NRTI_muts)

    X_NRTI = np.array(NRTI_specific, np.float)
    Y = NRTI[drug] # shorthand
    keep = ~np.isnan(np.array(Y)).astype(np.bool)
    X_NRTI = X_NRTI[np.nonzero(keep)]; Y=Y[keep]
    Y = np.array(np.log(Y), np.float); 

    if standardize:
        Y -= Y.mean()
        X_NRTI -= X_NRTI.mean(0)[None, :]; X_NRTI /= X_NRTI.std(0)[None,:]
    return X_NRTI, Y, np.array(NRTI_muts)
