import numpy as np
import pandas as pd
import functools



class diversity_functions:
        

    def __init__(self, metric, counts, validate=True, **kwargs):
        """ Compute alpha diversity for one or more samples

        Parameters
        ----------
        metric : str, callable
            The alpha diversity metric to apply to the sample(s). Passing metric as
            a string is preferable as this often results in an optimized version of
            the metric being used.
        counts : 1D or 2D array_like of ints or floats
            Vector or matrix containing count/abundance data. If a matrix, each row
            should contain counts of OTUs in a given sample.
        validate: bool, optional
            If `False`, validation of the input won't be performed. This step can
            be slow, so if validation is run elsewhere it can be disabled here.
            However, invalid input data can lead to invalid results or error
            messages that are hard to interpret, so this step should not be
            bypassed if you're not certain that your input data are valid. See
            :mod:`skbio.diversity` for the description of what validation entails
            so you can determine if you can safely disable validation.
        kwargs : kwargs, optional
            Metric-specific parameters.

        Returns
        -------
        pd.Series
            Values of ``metric`` for all vectors provided in ``counts``. The index
            will be ``ids``, if provided.

        Raises
        ------
        ValueError, MissingNodeError, DuplicateNodeError
            If validation fails. Exact error will depend on what was invalid.
        TypeError
            If invalid method-specific parameters are provided.

        """


        self.counts = counts

        metric_map = self._get_alpha_diversity_metric_map()

        if validate:
            counts = self._validate_counts_matrix(counts)
        
        if metric in metric_map:
            metric = functools.partial(metric_map[metric], **kwargs)
        else:
            raise ValueError('Unknown metric provided: %r.' % metric)

        # kwargs is provided here so an error is raised on extra kwargs
        results = [metric(c, **kwargs) for c in counts]
        return pd.Series(results, index=ids)


    def _get_alpha_diversity_metric_map(self):
        return {
            'observed_otus': self.observed_otus,
            'shannon': self.shannon,
            'simpson': self.simpson}
    

    def _validate_counts_vector(self):
        """Validate and convert input to an acceptable counts vector type.

        Note: may not always return a copy of `counts`!

        """
        counts = np.asarray(self.counts)
        try:
            if not np.all(np.isreal(counts)):
                raise Exception
        except Exception:
            raise ValueError("Counts vector must contain real-valued entries.")
        if counts.ndim != 1:
            raise ValueError("Only 1-D vectors are supported.")
        elif (counts < 0).any():
            raise ValueError("Counts vector cannot contain negative values.")

        return counts


    def observed_otus(self, counts):
        """Calculate the number of distinct OTUs.

        Parameters
        ----------
        counts : 1-D array_like, int
            Vector of counts.

        Returns
        -------
        int
            Distinct OTU count.

        """
        counts = self._validate_counts_vector(counts)
        return (counts != 0).sum()

    def shannon(self, base=2):
        r"""Calculate Shannon entropy of counts, default in bits.

        Shannon-Wiener diversity index is defined as:

        .. math::

        H = -\sum_{i=1}^s\left(p_i\log_2 p_i\right)

        where :math:`s` is the number of OTUs and :math:`p_i` is the proportion of
        the community represented by OTU :math:`i`.

        Parameters
        ----------
        counts : 1-D array_like, int
            Vector of counts.
        base : scalar, optional
            Logarithm base to use in the calculations.

        Returns
        -------
        double
            Shannon diversity index H.

        Notes
        -----
        The implementation here is based on the description given in the SDR-IV
        online manual [1]_ except that the default logarithm base used here is 2
        instead of :math:`e`.

        References
        ----------
        .. [1] http://www.pisces-conservation.com/sdrhelp/index.html

        """
        counts = self._validate_counts_vector(self.counts)
        freqs = counts / counts.sum()
        nonzero_freqs = freqs[freqs.nonzero()]
        return -(nonzero_freqs * np.log(nonzero_freqs)).sum() / np.log(base)


    def dominance(self):
        r"""Calculate dominance.

        Dominance is defined as

        .. math::

        \sum{p_i^2}

        where :math:`p_i` is the proportion of the entire community that OTU
        :math:`i` represents.

        Dominance can also be defined as 1 - Simpson's index. It ranges between
        0 and 1.

        Parameters
        ----------
        counts : 1-D array_like, int
            Vector of counts.

        Returns
        -------
        double
            Dominance.

        See Also
        --------
        simpson

        Notes
        -----
        The implementation here is based on the description given in [1]_.

        References
        ----------
        .. [1] http://folk.uio.no/ohammer/past/diversity.html

        """
        counts = self._validate_counts_vector(self.counts)
        freqs = counts / counts.sum()
        return (freqs * freqs).sum()




    def simpson(self):
        r"""Calculate Simpson's index.

        Simpson's index is defined as ``1 - dominance``:

        .. math::

        1 - \sum{p_i^2}

        where :math:`p_i` is the proportion of the community represented by OTU
        :math:`i`.

        Parameters
        ----------
        counts : 1-D array_like, int
            Vector of counts.

        Returns
        -------
        double
            Simpson's index.

        See Also
        --------
        dominance

        Notes
        -----
        The implementation here is ``1 - dominance`` as described in [1]_. Other
        references (such as [2]_) define Simpson's index as ``1 / dominance``.

        References
        ----------
        .. [1] http://folk.uio.no/ohammer/past/diversity.html
        .. [2] http://www.pisces-conservation.com/sdrhelp/index.html

        """
        counts = self._validate_counts_vector(self.counts)
        return 1 - self.dominance(counts)