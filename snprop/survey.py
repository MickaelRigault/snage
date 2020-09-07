"""Prompt vs. Delayed model of the SN population """
import pandas
import numpy as np
from scipy import stats
import matplotlib.pyplot as mpl

from .tools import asym_gaussian


# =========================================================================== #
#                                                                             #
#                                SURVEY CLASS                                 #
#                                                                             #
# =========================================================================== #

class survey(object):

    # =================================================================== #
    #                              Variables                              #
    # =================================================================== #

    surveys = ['SNF', 'SDSS', 'PS1', 'SNLS', 'HST']

    # Based on SK16's C11 model for SDSS and SNLS,
    #          SK18's C11 model for PS1,
    #          NR20's     model for SNF and HST
    all_cparams = {'SNF':
                   {'mu': 1, 'sigmadown': 1, 'sigmaup': 1},
                   'SDSS':  # SK16 table 1 C11
                   {'mu': -0.061, 'sigmadown': 0.023, 'sigmaup': 0.083},
                   'PS1':  # SK18 table 3 C11
                   {'mu': -0.100, 'sigmadown': 0.003, 'sigmaup': 0.134},
                   'SNLS':  # SK16 table 1 C11
                   {'mu': -0.112, 'sigmadown': 0.003, 'sigmaup': 0.144},
                   'HST':
                   {'mu': 1, 'sigmadown': 1, 'sigmaup': 1}}

    all_xparams = {'SNF':  # NR20 table 4
                   {'mu': 0.68, 'sigmadown': 1.34, 'sigmaup': 0.41},
                   'SDSS':  # SK16 table 1 C11
                   {'mu': 1.142, 'sigmadown': 1.652, 'sigmaup': 0.104},
                   'PS1':  # SK18 table 3 C11
                   {'mu': 0.384, 'sigmadown': 0.987, 'sigmaup': 0.505},
                   'SNLS':  # SK16 table 1 C11
                   {'mu': 0.974, 'sigmadown': 1.236, 'sigmaup': 0.283},
                   'HST':  # NR20 table 4
                   {'mu': 0.964, 'sigmadown': 1.467, 'sigmaup': 0.235}}

    all_mparams = {'SNF':
                   {'mu': 1, 'sigmadown': 1, 'sigmaup': 1},
                   'SDSS':
                   {'mu': 1, 'sigmadown': 1, 'sigmaup': 1},
                   'PS1':
                   {'mu': 1, 'sigmadown': 1, 'sigmaup': 1},
                   'SNLS':
                   {'mu': 1, 'sigmadown': 1, 'sigmaup': 1},
                   'HST':
                   {'mu': 1, 'sigmadown': 1, 'sigmaup': 1}}

    # =================================================================== #
    #                               Initial                               #
    # =================================================================== #

    def __init__(self, surveyname, new=False):
        """Sets the class parameters to given arguments"""
        if (surveyname not in self.surveys) and not new:
            raise KeyError(f"Survey must be in {self.surveys}, " +
                           "{surveyname} given. Set new=True to create one")
        self._surveyname = surveyname
        if not new:
            self._distprop_color = self.all_cparams[surveyname]
            self._distprop_stretch = self.all_xparams[surveyname]
            self._distprop_mass = self.all_mparams[surveyname]

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               EXTFUNC                               #
    # ------------------------------------------------------------------- #

    @staticmethod
    def deltaz(z, k=0.87, phi=2.8):
        """Fraction of young SNeIa as a function of redshift.
        from Rigault et al. 2018 (LsSFR paper)

        Parameters:
        -----------
        z: [float array of]
        redshifts

        k: [float] -optional-
        normalisation. 0.87 means that 50% of SNeIa are prompt at z
        \approx 0.05 (anchored by SNfactory)

        phi: [float] -optional-
        power law redshift dependency.

        Returns:
        --------
        array
        """
        return (k**(-1)*(1+z)**(-phi)+1)**(-1)

    # ------------------------------------------------------------------- #
    #                               SETTER                                #
    # ------------------------------------------------------------------- #

        # ----------------------------------------------------------- #
        #                           STRETCH                           #
        # ----------------------------------------------------------- #

    def set_distprop_color(self, mu=1, sigmaup=1, sigmadown=1):
        """Set the parameters of the SNe Ia color distribution, modeled as
        asymmetric gaussians."""

        self._distprop_color = {"mu": mu,
                                "sigmadown": sigmadown,
                                "sigmaup": sigmaup}

        # ----------------------------------------------------------- #
        #                            COLOR                            #
        # ----------------------------------------------------------- #

    def set_distprop_stretch(self, mu=1, sigmaup=1, sigmadown=1):
        """Set the parameters of the SNe Ia stretch distribution, modeled as
        asymmetric gaussians."""

        self._distprop_stretch = {"mu": mu,
                                  "sigmadown": sigmadown,
                                  "sigmaup": sigmaup}

        # ----------------------------------------------------------- #
        #                          HOST MASS                          #
        # ----------------------------------------------------------- #

    def set_distprop_mass(self, mu=1, sigmaup=1, sigmadown=1):
        """Set the parameters of the SNe Ia host mass distribution, modeled as
        asymmetric gaussians.
        """
        self._distprop_mass = {"mu": mu,
                               "sigmadown": sigmadown,
                               "sigmaup": sigmaup}

        # ----------------------------------------------------------- #
        #                       HUBBLE RESIDUAL                       #
        # ----------------------------------------------------------- #

    def set_distprop_hr(self, mean_prompt=0.075, sigma_prompt=0.1,
                        mean_delayed=-0.075, sigma_delayed=0.1):
        """Normal distribution for each age sample. (assuming 0.15 mag step).
        """
        self._distprop_hr = {"prompt": {"mean": mean_prompt,
                                        "sigma": sigma_prompt},
                             "delayed": {"mean": mean_delayed,
                                         "sigma": sigma_delayed}
                             }

    # - Distortion of what is in Nature

    # ------------------------------------------------------------------- #
    #                               GETTER                                #
    # ------------------------------------------------------------------- #

    def get_frac_prompt(self, z):
        """get the expected fraction of prompt SNe Ia as the given
        redshift(s) """
        if len(np.atleast_1d(z)) > 1:
            return self.deltaz(np.asarray(z)[:, None])
        return self.deltaz(z)

        # ----------------------------------------------------------- #
        #                           STRETCH                           #
        # ----------------------------------------------------------- #

    def get_distpdf_stretch(self, x, dx=None, **kwargs):
        """get the pdf of the stretch distribution at the given values.

        Parameters
        ----------
        x: [1d array]
            values where you want to estimate the pdf

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

        mode = asym_gaussian(x,
                             *list(self.distprop_stretch.values()),
                             dx=dx)
        return mode

        # ----------------------------------------------------------- #
        #                            COLOR                            #
        # ----------------------------------------------------------- #

    def get_distpdf_color(self, c, dc=None, **kwargs):
        """get the pdf of the color distribution at the given values.

        Parameters
        ----------
        c: [1d array]
            values where you want to estimate the pdf

        dc: [1d array] -optional-
            measurement error added in quadrature to the model's std.

        **kwargs goes to set_distprop_color()

        Returns
        -------
        pdf values (or list of)
        """

        self.set_distprop_color(**kwargs)
        if dc is None:
            dc = 0

        mode = asym_gaussian(c,
                             *list(self.distprop_color.values()),
                             dx=dc)
        return mode

        # ----------------------------------------------------------- #
        #                          HOST MASS                          #
        # ----------------------------------------------------------- #

    def get_distpdf_mass(self, M, dM=None, z=None, **kwargs):
        """get the pdf of the mass distribution at the given values.

        Parameters
        ----------
        M: [1d array]
            values where you want to estimate the pdf

        dM: [1d array] -optional-
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
        if dM is None:
            dM = 0

        if z is not None:
            raise NotImplementedError(
                "No redshift dependency implemented for get_distpdf_mass()." +
                "Set z=None")

        mode = asym_gaussian(M,
                             *list(self.distprop_mass.values()),
                             dx=dM)
        return mode

        # ----------------------------------------------------------- #
        #                       HUBBLE RESIDUAL                       #
        # ----------------------------------------------------------- #

    def get_distpdf_hr(self, x, fprompt, dx=None, **kwargs):
        """ get the pdf of the standardised Hubble Residual distribution at
        the given values.

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

    # ------------------------------------------------------------------- #
    #                               PLOTTER                               #
    # ------------------------------------------------------------------- #

        # ----------------------------------------------------------- #
        #                            TOOLS                            #
        # ----------------------------------------------------------- #

    def _draw_(self, a, pdf, size=None):
        """"""
        if len(np.shape(pdf)) == 1:
            return np.random.choice(a, size=size, p=pdf)
        elif len(np.shape(pdf)) == 2:
            return np.asarray([np.random.choice(mm, size=size, p=pdf)
                               for pdf_ in pdf])
        raise ValueError("pdf size must be 1 or 2.")

    def _read_fprompt_z_(self, fprompt=None, z=None):
        """ """
        if fprompt is None and z is None:
            raise ValueError("z or fprompt must be given.")

        elif fprompt is None:
            fprompt = self.get_frac_prompt(z)
        elif z is not None:
            raise ValueError("complict: either fprompt or z must be given.")

        return fprompt

        # ----------------------------------------------------------- #
        #                            DRAWS                            #
        # ----------------------------------------------------------- #

    def draw_property(self, which, nprompt, ndelayed, concat=True):
        """get a random realisation of the SN Ia property you want

        Parameters
        ----------
        which: [string]
            Property you want:
            - stretch
            - color
            - hr
            - mass

        nprompt, ndelayed: [ints]
            Number of prompt and delayed in the sample respectively.

        concat: [bool] -optional-
            do you want a unique list or first prompt then first delayed.

        Returns
        -------
        list of properties (see concat)
        """

        xx_ = self.property_range[which]
        prompt_pdf = getattr(self, f"get_distpdf_{which}")(xx_, 1)
        delayed_pdf = getattr(self, f"get_distpdf_{which}")(xx_, 0)

        prompt_prop = self._draw_(
            xx_, prompt_pdf/np.sum(prompt_pdf, axis=0), size=nprompt)
        delayed_prop = self._draw_(
            xx_, delayed_pdf/np.sum(delayed_pdf, axis=0), size=ndelayed)
        return(np.concatenate([prompt_prop, delayed_prop], axis=0) if concat
               else [prompt_prop, delayed_prop])

    def draw_sample(self, fprompt=None, z=None, size=None):
        """draw a random realisation of a sample.
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
        """
        fprompt = self._read_fprompt_z_(fprompt=fprompt, z=z)

        nprompt = int(size*fprompt)
        ndelayed = size-nprompt

        # - Color
        self._sample = pandas.DataFrame(
            {"color": self.draw_property("color", nprompt, ndelayed),
             "stretch": self.draw_property("stretch", nprompt, ndelayed),
             "mass": self.draw_property("mass", nprompt, ndelayed),
             "hr": self.draw_property("hr", nprompt, ndelayed),
             "prompt": np.concatenate([np.ones(nprompt), np.zeros(ndelayed)],
                                      axis=0),
             "redshift": z})

    def show_pdf(self, which, fprompt=None, z=None, detailed=False, ax=None,
                 cmap="coolwarm", zmax=2, **kwargs):
        """Show the figure of the PDF distribution of the given SN property

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

    def show_scatter(self, xkey, ykey, colorkey="prompt", ax=None, **kwargs):
        """Show the scatter plot of the sample parameters

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

    # =================================================================== #
    #                              Properties                             #
    # =================================================================== #

    @property
    def surveyname(self):
        """Dict of the color parameters for the selected survey"""
        return self._surveyname

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

    @property
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
