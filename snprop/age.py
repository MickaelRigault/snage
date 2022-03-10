""" Prompt vs. Delayed model of the SN population """
import pandas
import numpy as np

from scipy import stats
from .tools import asym_gaussian


class PrompDelayModel(object):

    def __init__(self):
        """ """

    # ====================== #
    #     Methods            #
    # ====================== #
    @staticmethod
    def deltaz(z, k=0.87, phi=2.8):
        """ fraction of young SNeIa as a function of redshift.
        from Rigault et al. 2018 (LsSFR paper)

        Parameters:
        -----------
        z: [float array of]
        redshifts

        k: [float] -optional-
        normalisation. 0.87 means that 50% of SNeIa are prompt at z\approx 0.05
        (anchored by SNfactory)

        phi: [float] -optional-
        power law redshift dependency.

        Returns:
        --------
        array
        """
        return (k**(-1)*(1+z)**(-phi)+1)**(-1)

    # -------- #
    #  SETTER  #
    # -------- #
    # STRETCH
    def set_distprop_stretch(self, mu1=0.37, sigma1=0.61,
                             mu2=-1.22, sigma2=0.56, a=0.51):
        """ Set the parameters of the SNe Ia stretch distribution.

        Following Nicolas, Rigault et al. 2020, the model is the following:
        - prompt SNeIa are single moded, "Mode 1"
        - delayed SNeIa are bimodal, a*"Model 1" + (1-a)*"Mode 2"

        Parameters
        ----------
        mu1, sigma1: [float, float]
            Mean and std of the mode 1

        mu2, sigma2: [float, float]
            Mean and std of the mode 2

        a: [float between 0 and 1]
            The relative weight of mode 1 over mode 2 (0.5 means equal weight)


        """
        if a < 0 or a > 1:
            raise ValueError(f"a must be between 0 and 1, {a} given")

        self._distprop_stretch = \
            {"mode1": {"loc": mu1, "scale": sigma1},
             "mode2": {"loc": mu2, "scale": sigma2},
             "a": a}

    # COLOR
    def set_distprop_color(self, mu=-0.030, sigmaup=0.086, sigmadown=0.052,
                           mu_delayed=None, sigmaup_delayed=None,
                           sigmadown_delayed=None):
        """
        Set the parameters of the SNe Ia color distribution,
        modeled as asymetric gaussians.

        If *_delayed are not provided, prompt and delayed are assumed
        similar (tested on mu_delayed)
        """
        if mu_delayed is None:
            mu_delayed = mu
            sigmaup_delayed = sigmaup
            sigmadown_delayed = sigmadown

        self._distprop_color = \
            {"prompt":
             {"mu": mu, "sigmaup": sigmaup, "sigmadown": sigmadown},
             "delayed":
             {"mu": mu_delayed, "sigmaup": sigmaup_delayed,
              "sigmadown": sigmadown_delayed}}

    # HOST MASS
    def set_distprop_mass(self,
                          mu_prompt=9.23,
                          sigmaup_prompt=0.96, sigmadown_prompt=0.47,
                          mu_delayed=10.61,
                          sigmaup_delayed=0.44, sigmadown_delayed=0.40):
        """
        Set the parameters of the SNe Ia mass distribution,
        modeled as one Gaussian for the prompt,
        and a Gaussian mixture for the delayed.
        """
        self._distprop_mass = \
            {"prompt":
             {"mu": mu_prompt, "sigmaup": sigmaup_prompt,
              "sigmadown": sigmadown_prompt},
             "delayed":
             {"mu": mu_delayed, "sigmaup": sigmaup_delayed,
              "sigmadown": sigmadown_delayed}}

    # HUBBLE RESIDUALS
    def set_distprop_hr(self, mean_prompt=0.075, sigma_prompt=0.1,
                        mean_delayed=-0.075, sigma_delayed=0.1):
        """
        Normal distribution for each age sample. (assuming 0.15 mag step).
        """
        self._distprop_hr = \
            {"prompt":
             {"mean": mean_prompt, "sigma": sigma_prompt},
             "delayed":
             {"mean": mean_delayed, "sigma": sigma_delayed}}

    # - Distortion of what is in Nature

    # -------- #
    # GETTER   #
    # -------- #
    def get_frac_prompt(self, z):
        """
        Get the expected fraction of prompt SNe Ia as the given redshift(s)
        """
        if len(np.atleast_1d(z)) > 1:
            return self.deltaz(np.asarray(z)[:, None])
        return self.deltaz(z)

    # - Stretch
    def get_distpdf_stretch(self, x, fprompt, dx=None, **kwargs):
        """
        Get the pdf of the stretch distribution at the given values.

        Parameters
        ----------
        x: [1d array]
            values where you want to estimate the pdf

        fprompt: [float between 0 and 1]
            Fraction of prompt. 0(1) means pure delayed(prompt)
            Could be a list.

        dx: [1d array] -optional-
            measurement error added in quadrature to the model's std.

        **kwargs goes to set_distprop_stretch()

        Returns
        -------
        pdf values (or list of)
        """
        self.set_distprop_stretch(**kwargs)
        if dx is None:
            dx = 0

        mode1_pdf = stats.norm.pdf(x,
                                   loc=self.distprop_stretch["mode1"]["loc"],
                                   scale=np.sqrt(self.distprop_stretch["mode1"]
                                                 ["scale"]**2 + dx**2))
        mode2_pdf = stats.norm.pdf(x,
                                   loc=self.distprop_stretch["mode2"]["loc"],
                                   scale=np.sqrt(self.distprop_stretch["mode2"]
                                                 ["scale"]**2 + dx**2))

        return(fprompt * mode1_pdf +
               (1-fprompt) * (self.distprop_stretch["a"]*mode1_pdf +
                              (1-self.distprop_stretch["a"])*mode2_pdf))

    # - Color
    def get_distpdf_color(self, x, fprompt, dx=None, **kwargs):
        """
        Get the pdf of the color distribution at the given values.

        Parameters
        ----------
        x: [1d array]
            values where you want to estimate the pdf

        fprompt: [float between 0 and 1]
            Fraction of prompt. 0(1) means pure delayed(prompt)
            Could be a list.

        dx: [1d array] -optional-
            measurement error added in quadrature to the model's std.

        **kwargs goes to set_distprop_color()

        Returns
        -------
        pdf values (or list of)
        """
        self.set_distprop_color(**kwargs)
        if dx is None:
            dx = 0

        prompt = asym_gaussian(
            x, *list(self.distprop_color["prompt"].values()), dx=dx)
        delayed = asym_gaussian(
            x, *list(self.distprop_color["delayed"].values()), dx=dx)
        return fprompt*prompt + (1-fprompt) * delayed

    # - Mass
    def get_distpdf_mass(self, x, fprompt, dx=None, z=None, **kwargs):
        """ get the pdf of the mass distribution at the given values.

        Parameters
        ----------
        x: [1d array]
            values where you want to estimate the pdf

        fprompt: [float between 0 and 1]
            Fraction of prompt. 0(1) means pure delayed(prompt)
            Could be a list.

        dx: [1d array] -optional-
            measurement error added in quadrature to the model's std.

        z: [float] -optional-
            NOT IMPLEMENTED YET
            the redshift at which the prompt/delayed - mass association is made

        **kwargs goes to set_distprop_mass()

        Returns
        -------
        pdf values (or list of)
        """
        self.set_distprop_mass(**kwargs)
        if dx is None:
            dx = 0

        if z is not None:
            raise NotImplementedError(
                "No redshift dependency implemented for get_distpdf_mass()." +
                "Set z=None")

        prompt = asym_gaussian(
            x, *list(self.distprop_mass["prompt"].values()), dx=dx)
        delayed = asym_gaussian(
            x, *list(self.distprop_mass["delayed"].values()), dx=dx)
        return fprompt*prompt + (1-fprompt) * delayed

    # - HR
    def get_distpdf_hr(self, x, fprompt, dx=None, **kwargs):
        """
        Get the pdf of the standardised Hubble Residual distribution at the
        given values.

        Parameters
        ----------
        x: [1d array]
            values where you want to estimate the pdf

        fprompt: [float between 0 and 1]
            Fraction of prompt. 0(1) means pure delayed(prompt)
            Could be a list.

        dx: [1d array] -optional-
            measurement error added in quadrature to the model's std.

        **kwargs goes to set_distprop_hr()

        Returns
        -------
        pdf values (or list of)
        """
        self.set_distprop_hr(**kwargs)
        if dx is None:
            dx = 0

        prompt = stats.norm.pdf(x,
                                loc=self.distprop_hr["prompt"]["mean"],
                                scale=np.sqrt(self.distprop_hr["prompt"]
                                              ["sigma"]**2+dx**2))
        delayed = stats.norm.pdf(x,
                                 loc=self.distprop_hr["delayed"]["mean"],
                                 scale=np.sqrt(self.distprop_hr["delayed"]
                                               ["sigma"]**2+dx**2))
        return fprompt*prompt + (1-fprompt) * delayed

    # ---------- #
    #  Draw      #
    # ---------- #
    def _draw_(self, a, pdf, size=None):
        """ """
        if len(np.shape(pdf)) == 1:
            return np.random.choice(a, size=size, p=pdf)
        elif len(np.shape(pdf)) == 2:
            return(np.asarray([np.random.choice(mm, size=size, p=pdf_)
                               for pdf_ in pdf]))
        raise ValueError("pdf shape must be 1 or 2.")

    def draw_age(self, fprompt=None, z=None, size=1):
        """ """
        fprompt = self._read_fprompt_z_(fprompt=fprompt, z=z)
        s = np.random.random(size=[len(np.atleast_1d(fprompt)), size])
        flag_p = s < fprompt
        young = np.zeros(s.shape)
        young[flag_p] = 1
        return young

    def draw_property(self, which, nprompt, ndelayed, concat=True):
        """ get a random realisation of the SN Ia property you want

        Parameters
        ----------
        which: [string]
            Property you want:
            - stretch
            - color
            - hr
            - mass
            - age

        nprompt, ndelayed: [ints]
            Number of prompt and delayed in the sample respectively.

        concat: [bool] -optional-
            do you want a unique list or first prompt then first delayed.

        Returns
        -------
        list of properties (see concat)
        """
        if which in ["age"]:
            if len(np.atleast_1d(nprompt)) == 1:
                return np.concatenate([np.ones(int(nprompt)),
                                       np.zeros(int(ndelayed))], axis=0)
            else:
                return [np.concatenate([np.ones(int(p_)),
                                        np.zeros(int(d_))], axis=0)
                        for p_, d_ in zip(nprompt, ndelayed)]

        xx_ = self.property_range[which]
        prompt_pdf = getattr(self, f"get_distpdf_{which}")(xx_, 1)
        delayed_pdf = getattr(self, f"get_distpdf_{which}")(xx_, 0)

        if len(np.atleast_1d(nprompt)) == 1:
            prompt_prop = self._draw_(
                xx_,
                prompt_pdf/np.sum(prompt_pdf, axis=0),
                size=int(nprompt))
            delayed_prop = self._draw_(
                xx_,
                delayed_pdf/np.sum(delayed_pdf, axis=0),
                size=int(ndelayed))
            return(np.concatenate([prompt_prop, delayed_prop], axis=0)
                   if concat else [prompt_prop, delayed_prop])
        else:
            prompt_prop = [self._draw_(xx_,
                                       prompt_pdf/np.sum(prompt_pdf, axis=0),
                                       size=int(nprompt_))
                           for nprompt_ in np.atleast_1d(nprompt)]
            delayed_prop = [self._draw_(xx_,
                                        delayed_pdf /
                                        np.sum(delayed_pdf, axis=0),
                                        size=int(ndelayed_))
                            for ndelayed_ in np.atleast_1d(ndelayed)]
            return([np.concatenate([p_, d_], axis=0)
                    if concat else [p_, d_] for p_, d_ in zip(prompt_prop,
                                                              delayed_prop)])

    def draw_sample(self, fprompt=None, z=None, size=1):
        """ draw a random realisation of a sample.
        It will be stored as self.sample (pandas.DataFrame)

        Parameters
        ----------
        fprompt: [0<=float<=1 or list of] -optional-
            Fraction of prompt in the sample
            = requested if z is not given =

        z: [float or list of] -optional-
            Redshift(s) of the SNe Ia
            = requested if fprompt is not given =

        // z and fprompt cannot be given together //

        size: [int] -optional-
            size of the sample.
            If fprompt or z are list, this will be the size per element.

        Returns
        -------
        Void (sets self.sample)

        Usage:
        -----
        >self.draw_sample(z=[0.1,0.5,1], size=1000)
        >self.sample is in that case a dataframe with the length of 3000
        1000 per redshift)
        """
        z = np.atleast_1d(z)
        ages = self.draw_age(fprompt=fprompt, z=z, size=size)
        nprompt = np.sum(ages, axis=1)
        ndelayed = size - nprompt
        data = {k: (np.concatenate(self.draw_property(k, nprompt, ndelayed))
                if len(nprompt) > 1
                else self.draw_property(k,
                                        nprompt,
                                        ndelayed))
                for k in ["color", "stretch", "age", "mass", "hr"]}
        data["z"] = np.concatenate((np.ones((len(z), size)).T*z).T)\
            if z is not None else None
        # - Color
        self._sample = pandas.DataFrame(data)

    def get_subsample(self, size, index_pdf=None):
        """Get the subsample (DataFrame) of the main self.sample given the pdf
        of each index.

        Parameters
        ----------
        size: [int]
            size of the new subsample

        index_pdf: [array/None] -optional-
             array of float of the same size if self.sample
             None means equal weight to all indexes.

        Returns
        -------
        DataFrame
        """
        if index_pdf is not None:
            index_pdf = index_pdf/np.sum(index_pdf, axis=0)

        subsample_index = np.random.choice(
            self.sample.index, p=index_pdf, size=size, replace=False)
        return self.sample[self.sample.index.isin(subsample_index)]

    def show_pdf(self, which, fprompt=None, z=None, detailed=False, ax=None,
                 cmap="coolwarm", zmax=2, **kwargs):
        """ Show the figure of the PDF distribution of the given SN property

        Parameters
        ----------
         which: [string]
            Property you want:
            - stretch
            - color
            - hr
            - mass

        fprompt: [0<=float<=1 or list of] -optional-
            Fraction of prompt in the sample
            = requested if z is not given =

        z: [float or list of] -optional-
            Redshift(s) of the SNe Ia
            = requested if fprompt is not given =

        // z and fprompt cannot be given together //

        detailed: Not Implemented yet

        ax: [matplotlib Axes] -optional-
            ax where the figure will be displayed

        cmap: [string ; matplotlib colormap] -optional-
            colormap. The value will be the SN redshift.

        zmax: [float] -optional-
            upper limit of the colormap


        **kwargs goes to ax.plot()
        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as mpl

        fprompt = self._read_fprompt_z_(fprompt=fprompt, z=z)

        # - Data
        xx = self.property_range[which]
        if detailed:
            print("detailed not implemented")
            prompt_pdf = getattr(self, f"get_distpdf_{which}")(xx, 1)
            delayed_pdf = getattr(self, f"get_distpdf_{which}")(xx, 0)
            pdf = fprompt * prompt_pdf + (1-fprompt) * delayed_pdf
        else:
            pdf = getattr(self, f"get_distpdf_{which}")(xx, fprompt)

        # - Axes
        if ax is None:
            fig = mpl.figure(figsize=[6, 4])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        # - Plot
        z = np.atleast_1d(z)
        if len(z) == 1:
            pdf = [pdf]
        for pdf_, z_ in zip(pdf, z):
            ax.plot(xx, pdf_, color=mpl.cm.get_cmap(cmap)(
                z_/zmax) if z_ is not None else "k", **kwargs)

        ax.set_xlabel(which, fontsize="large")
        return fig

    def show_scatter(self, xkey, ykey, colorkey="age", ax=None, **kwargs):
        """ Show the scatter plot of the sample parameters

        Parameters
        ----------
        xkey, ykey, colorkey: [string]
            self.sample entries used as x, y and color values

        ax: [matplotlib Axes] -optional-
            ax where the figure will be displayed

        **kwargs goes to ax.scatter()

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as mpl
        # - Axes
        if ax is None:
            fig = mpl.figure(figsize=[6, 4])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        ax.scatter(self.sample[xkey], self.sample[ykey],
                   c=self.sample[colorkey], **kwargs)

        ax.set_xlabel(xkey, fontsize="large")
        ax.set_ylabel(ykey, fontsize="large")
        return fig

    # ---------------- #
    #   Internal       #
    # ---------------- #
    def _read_fprompt_z_(self, fprompt=None, z=None):
        """ """
        if fprompt is None and z is None:
            raise ValueError("z or fprompt must be given.")

        elif fprompt is None:
            fprompt = self.get_frac_prompt(z)
        elif z is not None:
            raise ValueError("complict: either fprompt or z must be given.")

        return fprompt
    # ====================== #
    #    Properties          #
    # ====================== #

    @property
    def distprop_stretch(self):
        """dict of the stretch distribution parameters """
        if not hasattr(self, "_distprop_stretch")\
                or self._distprop_stretch is None:
            self.set_distprop_stretch()
        return self._distprop_stretch

    @property
    def distprop_color(self):
        """dict of the color distribution parameters"""
        if not hasattr(self, "_distprop_color")\
                or self._distprop_color is None:
            self.set_distprop_color()
        return self._distprop_color

    @property
    def distprop_mass(self):
        """dict of the host mass distribution parameters"""
        if not hasattr(self, "_distprop_mass")\
                or self._distprop_mass is None:
            self.set_distprop_mass()
        return self._distprop_mass

    @property
    def distprop_hr(self):
        """dict of the standardized hubble residuals distribution parameters"""
        if not hasattr(self, "_distprop_hr")\
                or self._distprop_hr is None:
            self.set_distprop_hr()
        return self._distprop_hr

    @property
    def sample(self):
        """pandas.DataFrame of the randomly draw sample parameters
        (see self.draw_sample()) """
        if not self.has_sample():
            raise AttributeError("No sample drawn. See self.draw_sample()")
        return self._sample

    def has_sample(self):
        """Test if you loaded a sample already (True means yes) """
        return hasattr(self, "_sample") and self._sample is not None

    @property
    def property_range(self):
        """Extent of the SN properties """
        if not hasattr(self, "_property_range")\
                or self._property_range is None:
            self._property_range = {"color": np.linspace(-0.4, 0.5, 1000),
                                    "stretch": np.linspace(-5, 5, 1000),
                                    "mass": np.linspace(6, 13, 1000),
                                    "hr": np.linspace(-1, +1, 1000)
                                    }
        return self._property_range
