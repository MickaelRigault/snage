from scipy import stats

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================================================================== #
#                                                                             #
#                                 PRELIMINARY                                 #
#                                                                             #
# =========================================================================== #

# =========================================================================== #
#                                  Colourmap                                  #
# =========================================================================== #

# create a colormap that consists of
# - 1/5 : custom colormap, ranging from white to the 1st color of the colormap
# - 4/5 : existing colormap

# set upper part: 4 * 256/4 entries
upper = mpl.cm.turbo(np.arange(256))

# set lower part: 1 * 256/4 entries
# - initialize all entries to 1
#   to make sure that the alpha channel (4th column) is 1
lower = np.ones((int(256/4), 4))
# - modify the first three columns (RGB):
#   range linearly between white (1,1,1)
#   and the first color of the upper colormap
for i in range(3):
    lower[:, i] = np.linspace(1, upper[0, i], lower.shape[0])

# combine parts of colormap
cmap = np.vstack((lower, upper))

# convert to matplotlib colormap
cmap_tpw = mpl.colors.ListedColormap(cmap, name='turbopw', N=cmap.shape[0])

# =========================================================================== #
#                                  Asymgauss                                  #
# =========================================================================== #


def asym_gaussian(x, mu, sigmaup, sigmadown, dx=None):
    """ """
    if dx is None:
        varup = sigmaup**2
        vardown = sigmadown**2
    else:
        varup = sigmaup**2 + dx**2
        vardown = sigmadown**2 + dx**2

    pdf = np.exp(-(x-mu)**2 / (2*varup))  # up
    pdf[x < mu] = np.exp(-(x[x < mu]-mu)**2 / (2*vardown))
    norm = np.sqrt(2*np.pi * (0.5*varup+0.5*vardown))
    return pdf/norm


# =========================================================================== #
#                                                                             #
#                                   CLASSES                                   #
#                                                                             #
# =========================================================================== #

# =========================================================================== #
#                                  Asymgauss                                  #
# =========================================================================== #

class AsymGaussian(object):
    """ """

    # =================================================================== #
    #                               Initial                               #
    # =================================================================== #

    @classmethod
    def from_data(cls, data, error=None, weights=None):
        """ """
        this = cls()
        fitout = this.fit(data, error=error, weights=weights)
        print(fitout)
        return this

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               SETTER                                #
    # ------------------------------------------------------------------- #

    def set_param(self, mu, sigmaup, sigmadown):
        """ """
        self._param = {"mu": mu, "sigmaup": sigmaup, "sigmadown": sigmadown}

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def pdf(self, x, dx=None, param=None):
        """ """
        if param is not None:
            self.set_param(*param)
        return asym_gaussian(x, dx=dx, **self.param)

    def fit(self, data, error=None, weights=None, guess=None):
        """ """
        from scipy import optimize
        if weights is None:
            weights = np.ones(len(data))

        def get_loglikelihood(param):
            self.set_param(*param)
            return np.sum(-2*np.log(weights*self.pdf(data, dx=error)))

        if guess is None:
            guess = [np.nanmean(data), np.nanstd(data)/2, np.nanstd(data)/2]

        return optimize.fmin(get_loglikelihood, guess)

    # ------------------------------------------------------------------- #
    #                               PLOTTER                               #
    # ------------------------------------------------------------------- #

    def show(self, ax=None, data=None, error=None,
             show_legend=True,
             dataprop={}, **kwargs):
        """ """
        if ax is None:
            fig = plt.figure(figsize=[6, 4])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        xx = np.linspace(self.mu-8*self.sigmadown,
                         self.mu+8*self.sigmadown, 1000)

        ax.plot(xx, self.pdf(xx), label="model", **kwargs)
        if data is not None:
            if error is None:
                ax.hist(data, normed=True, label="data", **dataprop)
            else:
                g = stats.norm.pdf(xx[:, None], loc=data, scale=error)
                ax.plot(xx, np.sum(g, axis=1)/len(data),
                        label="data", **dataprop)

        if show_legend:
            ax.legend(loc="best", frameon=False)
        return fig

    # =================================================================== #
    #                              Properties                             #
    # =================================================================== #

    @property
    def param(self):
        """ """
        if not hasattr(self, "_param"):
            raise AttributeError("no param set. see self.set_param()")
        return self._param

    @property
    def mu(self):
        """ """
        return self.param["mu"]

    @property
    def sigmaup(self):
        """ """
        return self.param["sigmaup"]

    @property
    def sigmadown(self):
        """ """
        return self.param["sigmadown"]


# =========================================================================== #
#                                   Checker                                   #
# =========================================================================== #

class Checker(object):
    '''Checks a SNANA simulation in various ways.
    Usage
    -----
    checkit = tools.Checker(SNANA_sims,
                            pantheon_data,
                            simulation_name)
    # To create the kernel from the simulation:
        checkit.set_kernel('mass', 'stretch', save=True)
        checkit.fit('mass_stretch')
    # To compute the fit of another kernel on the data:
        checkit.fit(kernel, abs_name, ord_name)

    # Optional:
    # If saved kernel from `checkit.set_kernel`:
        ax = checkit.show_kernel('mass_stretch', aspect=.5)
    # To give another kernel:
        ax = checkit.show_kernel(kernel, abs_name, ord_name, aspect=.5)
    checkit.show_scatter('mass', 'stretch', ax=ax, **kwargs)'''

    # =================================================================== #
    #                              Variables                              #
    # =================================================================== #

    # Dictionary of saved kernel fits
    fits = dict()
    # Dictionary of saved kernels
    kernels = dict()
    # Dictionary of saved imshow of kernel
    kernels_show = dict()
    # Gives "IDSURVEY" corresponding to Pantheon name
    find_id = {'SDSS': 1,
               'PS1': 15,
               'SNLS': 4}
    # Gives column name of both the SNANA simulation and Pantheon dataframes
    # and errors depending on which astrophysical parameter is asked
    find_name = {'mass':     ['HOST_LOGMASS', 'hostmass',
                              'HOST_LOGMASS_ERR', 'hostmass_err'],
                 'stretch':  ['x1', 'stretchs',
                              'x1ERR', 'stretchs_err'],
                 'redshift': ['zCMB', 'redshifts',
                              'zCMBERR', 0],
                 'color':    ['c', 'colors',
                              'cERR', 'colors_err']}

    # =================================================================== #
    #                               Initial                               #
    # =================================================================== #

    def __init__(self, sims_data, act_data, name):
        '''Save data'''
        self.sims_data = sims_data
        self.act_data = act_data
        self.name = name

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               Kernels                               #
    # ------------------------------------------------------------------- #

    def set_kernel(self, abs_name, ord_name, save=False):
        '''Creates a Gaussian KDE kernel from the sims.
        Parameters
        ----------
        abs_name, ord_name: strings
            names of the `sims_data` columns on which computing the kernel. See
            self.find_name.keys() for accepted terms.
        save: optional, bool
            bool to save the kernel and its imshow in the `self.kernels`
            and `self.kernels_show` dictionaries respectively, with key
            `abs_name + '_' + ord_name`.

        Returns
        ----------
        kernel: scipy.stats.kde.gaussian_kde
            the Gaussian KDE of the simulated SNANA on the specified axes'''
        if (abs_name == 'mass') or (ord_name == 'mass'):
            sims_data = self.sims_data[
                self.sims_data[self.find_name['mass'][0]] > 7]
        else:
            sims_data = self.sims_data

        abs_sims = self.find_name[abs_name][0]
        ord_sims = self.find_name[ord_name][0]
        m1 = sims_data[abs_sims]
        m2 = sims_data[ord_sims]
        values = np.vstack([m1, m2])
        kernel = stats.gaussian_kde(values)

        if save is False:
            return kernel
        else:
            kernel_name = abs_name + '_' + ord_name
            xmin = m1.min()
            xmax = m1.max()
            ymin = m2.min()
            ymax = m2.max()

            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions), X.shape)

            self.kernels[kernel_name] = kernel
            self.kernels_show[kernel_name] = Z

            return kernel

    def fit(self,
            kernel_name=None,
            kernel=None, abs_name=None, ord_name=None):
        '''Gives the value of how data corresponds to interpolated simulations
        Parameters
        ----------
        kernel_name: string
            name of the saved kernel in `self.kernels`
        kernel: gaussian_kde
            given Gaussian KDE on which to compute the sum of probabilities.
            Either a kernel_name or a kernel + absciss and ordinate can be
            input
        abs_name, ord_name: strings
            name of the `act_data` columns on which applying the kernel. See
            self.find_name.keys() for accepted terms.

        Returns
        -------
        `self.fits[kernel_name]` or `np.sum(kernel(data))`: float
            the sum of the probabilities of all datapoints using previously
            saved or given kernel.'''
        if (kernel_name is None) and (kernel is None):
            raise NameError('Either `kernel_name` or `kernel` must be given')
        if (kernel_name is not None) and (kernel is not None):
            raise NameError("`kernel_name` and `kernel` can't both be given")
        if (kernel_name is not None) and (kernel is None):
            abs_name, ord_name = kernel_name.split('_')
        if (kernel_name is None) and (kernel is not None):
            pass

        if (abs_name == 'mass') or (ord_name == 'mass'):
            act_data = self.act_data[
                self.act_data[self.find_name['mass'][1]] > 7]
        else:
            act_data = self.act_data
        abs_act = self.find_name[abs_name][1]
        ord_act = self.find_name[ord_name][1]
        d1 = act_data[abs_act]
        d2 = act_data[ord_act]
        data = np.vstack([d1, d2])
        if (kernel_name is not None) and (kernel is None):
            self.fits[kernel_name] = np.sum(self.kernels[kernel_name](data))
            return self.fits[kernel_name]
        if (kernel_name is None) and (kernel is not None):
            return np.sum(kernel(data))

    # ------------------------------------------------------------------- #
    #                               PLOTTER                               #
    # ------------------------------------------------------------------- #

        # ----------------------------------------------------------- #
        #                           Kernels                           #
        # ----------------------------------------------------------- #

    def show_kernel(self,
                    kernel_name=None,
                    kernel=None, abs_name=None, ord_name=None,
                    ax=None, show_cb=True, cax=None,
                    aspect=1, alpha_kernel=1,
                    ticksize='x-large', fsize='x-large'):
        '''Represents the interpolated kernel as an imshow'''
        if ax is None:
            fig = plt.figure(figsize=[7, 5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        if (kernel_name is None) and (kernel is None):
            raise NameError('Either `kernel_name` or `kernel` must be given')
        if (kernel_name is not None) and (kernel is not None):
            raise NameError("`kernel_name` and `kernel` can't both be given")
        if (kernel_name is not None) and (kernel is None):
            xmin = self.kernels[kernel_name].dataset[0].min()
            xmax = self.kernels[kernel_name].dataset[0].max()
            ymin = self.kernels[kernel_name].dataset[1].min()
            ymax = self.kernels[kernel_name].dataset[1].max()
            Z = self.kernels_show[kernel_name]
            abs_name, ord_name = kernel_name.split('_')
        if (kernel_name is None) and (kernel is not None):
            xmin = kernel.dataset[0].min()
            xmax = kernel.dataset[0].max()
            ymin = kernel.dataset[1].min()
            ymax = kernel.dataset[1].max()

            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions), X.shape)

        ims = ax.imshow(np.rot90(Z), cmap=cmap_tpw,
                        extent=[xmin, xmax, ymin, ymax],
                        aspect=aspect, alpha=alpha_kernel)

        if show_cb is True:
            if cax is None:
                cb = ax.figure.colorbar(ims, ax=ax)
            else:
                cb = ax.figure.colorbar(ims, cax=cax)
            cb.ax.tick_params(labelsize=ticksize)
        else:
            pass

        ax.tick_params(labelsize=ticksize)
        ax.set_xlabel(f'{abs_name}', fontsize=fsize)
        ax.set_ylabel(f'{ord_name}', fontsize=fsize)

        return ax

        # ----------------------------------------------------------- #
        #                           Histoer                           #
        # ----------------------------------------------------------- #

    def show_hist(self,
                  abs_name, survey='all',
                  ax=None,
                  alpha=.5, nbbins=15,
                  ht_sims='stepfilled', lbl_sims='Sims',
                  ec_sims=None, fc_sims=cmap_tpw(0.35),
                  ht_data='step', lbl_data='Data',
                  lw=2, fc_data='C0',
                  ticksize='x-large', fsize='x-large',
                  show_leg=False):
        '''Simple histogram plotting'''
        if ax is None:
            fig = plt.figure(figsize=[7, 5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        if abs_name == 'mass':
            sims_data = self.sims_data[
                self.sims_data[self.find_name['mass'][0]] > 7]
            act_data = self.act_data[
                self.act_data[self.find_name['mass'][1]] > 7]
        else:
            sims_data = self.sims_data
            act_data = self.act_data

        if survey != 'all':
            sims_data = sims_data[sims_data['IDSURVEY'].isin([
                self.find_id.get(key) for key in survey])]
            act_data = act_data[act_data['survey'].isin(survey)]

        prophist = dict(alpha=alpha, density=True)

        abs_sims, abs_act = self.find_name[abs_name][0:2]

        _, binsv, _ = ax.hist(sims_data[abs_sims],
                              histtype=ht_sims,
                              bins=nbbins,
                              facecolor=fc_sims,
                              edgecolor=ec_sims,
                              label=lbl_sims,
                              **prophist)

        ax.hist(act_data[abs_act],
                bins=binsv,
                histtype=ht_data, lw=lw,
                color=fc_data,
                label=lbl_data,
                **prophist)

        ax.tick_params(labelsize=ticksize)
        ax.set_xlabel(f'{abs_name}', fontsize=fsize)

        if show_leg is True:
            ax.legend(ncol=1, loc='upper left')
        else:
            pass

        return ax

        # ----------------------------------------------------------- #
        #                           Scatter                           #
        # ----------------------------------------------------------- #

    def show_scatter(self,
                     abs_name, ord_name, survey='all',
                     ax=None, show_cb=True, cax=None,
                     alpha_sims=.5, gsize=30, cmap=cmap_tpw,
                     xsimscale='linear', ysimscale='linear',
                     lbl_sims='Sims',
                     mk_data='o', s_data=50, lbl_data='Data',
                     lw=1, alpha_data=.7, fc_data='C0',
                     ticksize='x-large', fsize='x-large'):
        '''Scatter of sims and data'''
        if ax is None:
            fig = plt.figure(figsize=[8, 7])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        prop_hex = dict(alpha=alpha_sims, gridsize=gsize, cmap=cmap)

        if (abs_name == 'mass') or (ord_name == 'mass'):
            sims_data = self.sims_data[
                self.sims_data[self.find_name['mass'][0]] > 7]
            act_data = self.act_data[
                self.act_data[self.find_name['mass'][1]] > 7]
        else:
            sims_data = self.sims_data
            act_data = self.act_data

        if survey != 'all':
            sims_data = sims_data[sims_data['IDSURVEY'].isin([
                self.find_id.get(key) for key in survey])]
            act_data = act_data[act_data['survey'].isin(survey)]

        abs_sims, abs_act = self.find_name[abs_name][0:2]
        ord_sims, ord_act = self.find_name[ord_name][0:2]

        hb = ax.hexbin(sims_data[abs_sims],
                       sims_data[ord_sims],
                       xscale=xsimscale, yscale=ysimscale,
                       **prop_hex)

        prop = dict(marker=mk_data, s=s_data,
                    lw=lw, alpha=alpha_data, color=fc_data)

        ax.scatter(act_data[abs_act],
                   act_data[ord_act],
                   label=lbl_data, **prop)

        if show_cb is True:
            if cax is None:
                cb = ax.figure.colorbar(hb, ax=ax)
            else:
                cb = ax.figure.colorbar(hb, cax=cax)
            cb.ax.tick_params(labelsize=ticksize)
        else:
            pass

        ax.tick_params(labelsize=ticksize)
        ax.set_xlabel(f'{abs_name}', fontsize=fsize)
        ax.set_ylabel(f'{ord_name}', fontsize=fsize)

        return ax

        # ----------------------------------------------------------- #
        #                           Complet                           #
        # ----------------------------------------------------------- #

    def show_all(self,
                 survey='all',
                 nbbins=8, fc_sims=cmap_tpw(0.35), fc_data='C0'):
        '''5 plots with hists and scatters'''
        fig = plt.figure(figsize=[20, 15])

        width_plot_cb = 0.35
        space_cb = 0.025
        width_cb = 0.0125
        xmin_bottom = 0.075
        ymin_bottom = 0.05
        height_plot_cb = 0.40

        xmin_top = 0.02
        ymin_top = 0.15 + height_plot_cb
        width_plot = 0.30
        space_plot = 0.03
        height_plot = 0.30

        ax4 = fig.add_axes([xmin_top, ymin_top,
                            width_plot, height_plot])

        ax1 = fig.add_axes([ax4.get_position().get_points()[1][0]
                            + space_plot, ymin_top,
                            width_plot, height_plot])

        ax5 = fig.add_axes([ax1.get_position().get_points()[1][0]
                            + space_plot, ymin_top,
                            width_plot, height_plot])

        ax2 = fig.add_axes([xmin_bottom, ymin_bottom,
                            width_plot_cb, height_plot_cb])
        axb = fig.add_axes([ax2.get_position().get_points()[1][0]
                            + space_cb, ymin_bottom,
                            width_cb, height_plot_cb])

        ax3 = fig.add_axes([axb.get_position().get_points()[1][0]
                            + 3*space_cb, ymin_bottom,
                            width_plot_cb, height_plot_cb])
        axc = fig.add_axes([ax3.get_position().get_points()[1][0]
                            + space_cb, ymin_bottom,
                            width_cb, height_plot_cb])

        self.show_hist(abs_name='redshift',
                       survey=survey, lbl_data=str(survey),
                       nbbins=nbbins, fc_sims=fc_sims, fc_data=fc_data,
                       ax=ax1, ticksize=20, fsize=20)

        self.show_hist(abs_name='stretch',
                       survey=survey,
                       nbbins=nbbins, fc_sims=fc_sims, fc_data=fc_data,
                       ax=ax4, ticksize=20, fsize=20)

        self.show_hist(abs_name='color',
                       survey=survey,
                       nbbins=nbbins, fc_sims=fc_sims, fc_data=fc_data,
                       ax=ax5, ticksize=20, fsize=20)

        self.show_scatter(abs_name='redshift', ord_name='stretch',
                          survey=survey,
                          fc_data=fc_data,
                          ax=ax2, cax=axb,
                          xsimscale='log',
                          ticksize=20, fsize=20)

        self.show_scatter(abs_name='mass', ord_name='stretch',
                          survey=survey,
                          fc_data=fc_data,
                          ax=ax3, cax=axc,
                          ticksize=20, fsize=20)

        ax1.legend(fontsize=20, ncol=2,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 1.0, 0.0, 0.25))

        fig.suptitle(self.name, fontsize=20)

        return fig
