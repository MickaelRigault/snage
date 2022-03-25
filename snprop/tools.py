from astropy.cosmology import Planck15 as Planck15
from astropy import units
from scipy import stats

import re
import glob
import numpy as np
import pandas as pd
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
    from snprop import tools
    checkit = tools.Checker(SNANA_folder,
                            pantheon_data)
    # If more than one FITOPT or MUOPT (i.e. FITOPT001 instead of 000):
    checkit = tools.Checker(SNANA_folder,
                            pantheon_data,
                            ['001', '000'])
    # To create the kernel from the simulation:
        checkit.set_kernel('mass', 'stretch', save=True)
        checkit.fit('mass_stretch')
    # To compute the fit of another kernel on the data:
        checkit.fit(kernel, abs_name, ord_name)

    # Optional:
    # If saved kernel from `checkit.set_kernel`:
        ax = checkit.show_kernel('mass_stretch')
    # To give another kernel:
        ax = checkit.show_kernel(kernel, abs_name, ord_name)
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
    find_id = {'LOWZ': [5, 64, 65, 66],
               'SDSS':  [1],
               'PS1':   [15],
               'SNLS':  [4]}
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

    def __init__(self, sims_data, act_data,
                 name, params=None, between_fitres=False):
        self.between_fitres = between_fitres
        self.sims_data = sims_data
        self.act_data = act_data
        self.name = name
        if params is not None:
            self.modif, self.wfit, self.wall, self.gab = params

    @classmethod
    def from_biascor(cls, sims_folder, act_data,
                     ssize=False,
                     folnum=['000', '000'],
                     sample_sims=True, name=None,
                     cols=['CID', 'IDSURVEY', 'zCMB', 'zCMBERR',
                           'HOST_LOGMASS', 'HOST_LOGMASS_ERR',
                           'x1', 'x1ERR', 'MU', 'MUERR', 'MURES']):
        '''Loads all the relevant information from a SNANA BIASCOR folder'''
        fitnum, munum = folnum
        if name is None:
            sept = [sims_folder.split('/')[i].split('_')
                    for i in range(len(sims_folder.split('/')))]
            name = '_'.join([sept[5][-1]] + sept[7][-2:])
        else:
            pass
        sims_prefolder = '%s' % ('/'.join(sims_folder.split('/')[:-2]))
        if ~ssize:
            sims_data = cls.read(sims_folder +
                                 '/FITOPT' + fitnum + '_' +
                                 'MUOPT' + munum + '.FITRES')
            modif = cls.read(sims_folder +
                             '/FITOPT' + fitnum + '_' +
                             'MUOPT' + munum + '.M0DIF')
            wfit = cls.read(sims_folder +
                            '/wfit_' +
                            'FITOPT' + fitnum + '_' +
                            'MUOPT' + munum + '.YAML')
            wfit.rename(inplace=True,
                        columns={'wsig_marg': 'w_err',
                                 'OMsig_marg': 'OM_err'})
            gab = cls.read(sims_folder +
                           '/FITOPT' + fitnum + '_' +
                           'MUOPT' + munum + '.FITRES',
                           True)
            wall = cls.read(sims_prefolder + '/BBC_SUMMARY_wfit.FITRES')
            params = [modif, wfit, wall, gab]
        else:
            # Take all the subfolders's FITRES
            sims_path = glob.glob(sims_folder[:-1] +
                                  '-*/FITOPT' + fitnum + '_' +
                                  'MUOPT' + munum + '.FITRES')
            # Take all the numbers
            sims_numbers = [f.split('-')[-1].split('/')[0] for f in sims_path]
            # Concatenate all of these
            sims_data = pd.concat([cls.read(f) for f in sims_path],
                                  keys=sims_numbers)
            # Repeat for M0DIF
            modif_path = glob.glob(sims_folder[:-1] +
                                   '-*/FITOPT' + fitnum + '_' +
                                   'MUOPT' + munum + '.M0DIF')
            modif = pd.concat([cls.read(f) for f in modif_path],
                              keys=sims_numbers)
            # Make a dataframe using the mean of all the others and append it
            # to the full one with index 0000
            temp = modif.loc['0001'].copy()
            temp['z'] = modif.groupby(level=1)['z'].mean()
            temp['MUDIF'] = modif.groupby(level=1)['MUDIF'].mean()
            temp['MUDIFERR'] = modif.groupby(level=1)['MUDIF'].std() /\
                len(modif.index.levels[0])
            temp['NFIT'] = modif.groupby(level=1)['NFIT'].sum()
            modif = modif.append(
                temp.assign(k='0000').set_index('k',
                                                append=True).swaplevel(0, 1))
            # Repeat for wfit
            wfit_path = glob.glob(sims_folder[:-1] +
                                  '-*/FITOPT' + fitnum + '_' +
                                  'MUOPT' + munum + '.YAML')
            wfit = pd.concat([cls.read(f).rename(
                inplace=True,
                columns={'wsig_marg': 'w_err',
                         'OMsig_marg': 'OM_err'})
                for f in wfit_path],
                keys=sims_numbers)
            # Repeat for gab
            gab_path = glob.glob(sims_folder[:-1] +
                                 '-*/FITOPT' + fitnum + '_' +
                                 'MUOPT' + munum + '.FITRES',
                                 True)
            gab = pd.concat([cls.read(f) for f in gab_path],
                            keys=sims_numbers)
            # Don't repeat for wall, by definition of wall
            wall = cls.read(sims_prefolder + '/BBC_SUMMARY_wfit.FITRES')

        if sample_sims:
            sims_data = cls.sample_sims(sims_data, act_data)

        if cols is None:
            return cls(sims_data, act_data,
                       name, params, between_fitres=False)
        else:
            return cls(sims_data[cols], act_data,
                       name, params, between_fitres=False)

    @ classmethod
    def from_input(cls, sims_data, act_data, between_fitres=False,
                   name=None, sample_sims=True,
                   cols=['CID', 'IDSURVEY', 'zCMB', 'zCMBERR',
                         'HOST_LOGMASS', 'HOST_LOGMASS_ERR',
                         'x1', 'x1ERR', 'MU', 'MUERR', 'MURES']):
        sims_data = cls.sample_sims(sims_data, act_data)
        if cols is None:
            return cls(sims_data, act_data,
                       name, between_fitres=between_fitres)
        else:
            return cls(sims_data[cols], act_data,
                       name, between_fitres=between_fitres)

    # =================================================================== #
    #                               Extfunc                               #
    # =================================================================== #

    @staticmethod
    def get_numbers(folder, opt=True):
        '''Returns the possible FITOPT and MUOPT numbers from a SNANA output
        folder if "opt" is True, or the BBCFIT numbers otherwise.
        Parameter
        ---------
        folder: string
            The path to the wanted SNANA output folder where files of the type
            FITOPT###_MUOPT### exist with different numbers.

        Returns
        -------
        Array of the combinations of number that exist in the folder.
        '''
        folnums = []
        if opt:
            for file in glob.glob(folder + '/FITOPT*'):
                folnums.append([file.split('/')[-1].split('_')[0][-3:],
                                file.split('/')[-1].split('.')[-2][-3:]])
            return np.unique(folnums, axis=0)
        else:
            folstrs = []
            for fd in glob.glob(folder + 'OUTPUT_BBCFIT*'):
                folstrs.append(fd.split('-')[-1])
                folnums.append(int(fd.split('-')[-1]))
            return folstrs, folnums

    @staticmethod
    def read(filename, gab=False, ind=None):
        '''Read relevant data depending on the extension of said file.
        Parameter
        ---------
        filename: string
            The name of the SNANA output file that one wants to have as a
            usable pandad DataFrame: can be a `wfit[...].YAML` file, a
            `FITOPT###_MUOPT###.FITRES` or `FITOPT###_MUOPT###.M0DIF` file.

        Returns
        -------
        A DataFrame containing the asked file's data in a readable way.
        '''
        if gab:
            with open(filename, 'r') as f:
                gab_lines = [line for line in f if
                             ('alpha0' in line) or
                             ('beta0' in line) or
                             ('gamma0' in line)]
                if len(gab_lines) == 0:
                    return 'Asked file has fixed parameters'
                gab_lines = gab_lines[:len(gab_lines)//2]
            pnames = [re.split('0',
                               re.split('=\s+',
                                        line)[0][3:])[0]
                      for line in gab_lines]
            for p in [1, 3, 5][:len(pnames)]:
                pnames.insert(p, pnames[p-1]+'_err')
            pvals = [[float(re.split('\s+\\n',
                                     re.split('\s\+-\s+',
                                              re.split('=\s+',
                                                       line)
                                              [-1])
                                     [k])
                            [0])
                      for k in range(2)]
                     for line in gab_lines]
            pvals = [item for sublist in pvals for item in sublist]
            gab_val = pd.DataFrame({name: pvals[n]
                                    for n, name in enumerate(pnames)},
                                   index=[0] if ind is None else [ind])
            return gab_val
        else:
            # Get the filetype
            ftype = filename.split('.')[-1].lower()
            # If it's a `wfit[...].YAML` file, we want a DataFrame whose
            # columns are the parameters and the one line is the values; that
            # way it'll be concatenable with other results
            if ftype == 'yaml':
                return pd.read_csv(filename,
                                   sep=':\\s+', engine='python',
                                   header=None, index_col=0).transpose()
            # Otherwise, we want to get rid of the commented lines, get the
            # column names that are after `VARNAMES`, and each line is either a
            # `SN:` or `ROW:` line.
            else:
                # todel = {'fitres': 'SN', 'm0dif': 'ROW'}
                with open(filename) as f:
                    for i, line in enumerate(f):
                        if line.startswith('VARNAMES:'):
                            line = line.replace(',', ' ')
                            line = line.replace('\n', '')
                            names = line.split()
                        elif line.startswith(('SN', 'ROW')):
                            startrows = i
                            break
                pandas = pd.read_csv(filename, header=None,
                                     names=names, skiprows=startrows,
                                     delim_whitespace=True, comment='#')
                return pandas.drop('VARNAMES:', axis=1)

    @ staticmethod
    def sample_sims(fitres, data, between_fitres=False):
        '''Draws simulated data from the fitres to mimic the density of actual
        data.
        Parameters
        ----------
        fitres, data: pandas DataFrame
            SNANA FITRES file passed through `Checker.read(filename)` function;
            Currently Pantheon dataset from the `variaIa.stretchevol.tools`.

        Returns
        -------
        Returns a modified DataFrame of the fitres whose survey ratio mimics
        that of the data.'''
        if not between_fitres:
            # Extract the surveys' names from the data
            surveys = np.unique(data['survey'])
            # Take the corresponding ids using previous `find_id` dict
            ids = [Checker.find_id[survey] for survey in surveys]
            # Compute the size of each survey in the data
            surveys_size = {survey:
                            len(data[data['survey'] == survey])
                            for survey in surveys}
            # And the same for the fitres
            fitres_size = {survey:
                           len(fitres[fitres['IDSURVEY'].isin(idsurvey)])
                           for survey, idsurvey in zip(surveys, ids)}
        else:
            # Suppose all surveys are there
            surveys = Checker.find_id.keys()
            # Make the list of ids the same way
            ids = [Checker.find_id[survey] for survey in surveys]
            # Compute the size of each survey in the data
            surveys_size = {survey:
                            len(data[data['IDSURVEY'].isin(idsurvey)])
                            for survey, idsurvey in zip(surveys, ids)}
            # Same for fitres
            fitres_size = {survey:
                           len(fitres[fitres['IDSURVEY'].isin(idsurvey)])
                           for survey, idsurvey in zip(surveys, ids)}
        # Get which survey has the lowest size in the data
        lowest = min(surveys_size, key=surveys_size.get)
        # Make the list of all others surveys to loop on them
        survothers = [survey for survey in surveys if survey is not lowest]
        # To compute how many SNe have to be drawn from the sims overall, the
        # best is to take the lowest value of the lowest survey from the sims
        # that will allow the ratio of the others wrt to the lowest to exist
        # (i.e. the numbers for the non-lowest can't be higher than their
        # current value); if the resulting lowest sample is higher than what
        # is available in the fitres, take the max of the fitres then.
        lowsug = np.min([fitres_size[other]*surveys_size[lowest]
                         / surveys_size[other]
                         for other in survothers])
        todraw = {lowest: int(np.min([lowsug, fitres_size[lowest]]))}
        # Loop on the others
        for other in survothers:
            todraw[other] = int(todraw[lowest]*surveys_size[other]
                                / surveys_size[lowest])
        # Use `sample` to draw from each survey in the sims
        fitres_sub = [fitres[fitres['IDSURVEY'].isin(
            idsurvey)].sample(todraw[survey])
            for survey, idsurvey in zip(surveys, ids)]
        # Concatenate
        return pd.concat(fitres_sub)

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
        abs_sims = self.find_name[abs_name][0]
        ord_sims = self.find_name[ord_name][0]

        if (abs_name == 'mass') or (ord_name == 'mass'):
            sims_data = self.sims_data[
                self.sims_data[self.find_name['mass'][0]] > 7]
        else:
            sims_data = self.sims_data

        if (abs_name == 'redshift'):
            m1 = np.log10(sims_data[abs_sims])
        else:
            m1 = sims_data[abs_sims]
        if (ord_name == 'redshift'):
            m2 = np.log10(sims_data[ord_sims])
        else:
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

            X, Y = np.mgrid[xmin:xmax:100j,
                            ymin:ymax:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions), X.shape)

            self.kernels[kernel_name] = kernel
            if (abs_name == 'redshift') and (ord_name != 'redshift'):
                self.kernels_show[kernel_name] = [np.power(10, X), Y, Z]
            if (ord_name == 'redshift') and (abs_name != 'redshift'):
                self.kernels_show[kernel_name] = [X, np.power(10, Y), Z]
            if (abs_name != 'redshift') and (ord_name != 'redshift'):
                self.kernels_show[kernel_name] = [X, Y, Z]

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
            if self.between_fitres:
                act_data = self.act_data[
                    self.act_data[self.find_name['mass'][0]] > 7]
            else:
                act_data = self.act_data[
                    self.act_data[self.find_name['mass'][1]] > 7]
        else:
            act_data = self.act_data

        if self.between_fitres:
            abs_act = self.find_name[abs_name][0]
            ord_act = self.find_name[ord_name][0]
        else:
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
                    kernel_name=None, abs_name=None, ord_name=None,
                    kernel=None, xscale='linear', yscale='linear',
                    ax=None, show_cb=True, cax=None,
                    alpha_kernel=1,
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
            X, Y, Z = self.kernels_show[kernel_name]
            if (abs_name is None) and (ord_name is None):
                abs_name, ord_name = kernel_name.split('_')
        if (kernel_name is None) and (kernel is not None):
            xmin = kernel.dataset[0].min()
            xmax = kernel.dataset[0].max()
            ymin = kernel.dataset[1].min()
            ymax = kernel.dataset[1].max()

            if (xscale == 'linear') and (yscale == 'linear'):
                X, Y = np.mgrid[xmin:xmax:100j,
                                ymin:ymax:100j]
            if (xscale == 'log') and (yscale == 'linear'):
                X, Y = np.mgrid[np.log10(xmin):np.log10(xmax):100j,
                                ymin:ymax:100j]
                X = 10**X
            if (yscale == 'log') and (xscale == 'linear'):
                X, Y = np.mgrid[xmin:xmax:100j,
                                np.log10(ymin):np.log10(ymax):100j]
                Y = 10**Y
            if (xscale == 'log') and (yscale == 'log'):
                X, Y = np.mgrid[np.log10(xmin):np.log10(xmax):100j,
                                np.log10(ymin):np.log10(ymax):100j]
            else:
                raise NameError('Scales must be `"log"` or `"linear"`')
            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions), X.shape)

        ims = ax.pcolor(X, Y, Z, cmap=cmap_tpw,
                        shading='auto', alpha=alpha_kernel)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

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
            if self.between_fitres:
                act_data = self.act_data[
                    self.act_data[self.find_name['mass'][0]] > 7]
            else:
                act_data = self.act_data[
                    self.act_data[self.find_name['mass'][1]] > 7]
        else:
            sims_data = self.sims_data
            act_data = self.act_data

        if survey != 'all':
            sims_data = sims_data[sims_data['IDSURVEY'].isin(
                self.find_id[survey])]
            if self.between_fitres:
                act_data = act_data[act_data['IDSURVEY'].isin(
                    self.find_id[survey])]
            else:
                act_data = act_data[act_data['survey'].isin(survey)]

        prophist = dict(alpha=alpha, density=True)

        if self.between_fitres:
            abs_sims = self.find_name[abs_name][0]
            abs_act = abs_sims
        else:
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
            if self.between_fitres:
                act_data = self.act_data[
                    self.act_data[self.find_name['mass'][0]] > 7]
            else:
                act_data = self.act_data[
                    self.act_data[self.find_name['mass'][1]] > 7]
        else:
            sims_data = self.sims_data
            act_data = self.act_data

        if survey != 'all':
            sims_data = sims_data[sims_data['IDSURVEY'].isin(
                self.find_id[survey])]
            if self.between_fitres:
                act_data = act_data[act_data['IDSURVEY'].isin(
                    self.find_id[survey])]
            else:
                act_data = act_data[act_data['survey'].isin(survey)]

        if self.between_fitres:
            abs_sims = self.find_name[abs_name][0]
            abs_act = abs_sims
            ord_sims = self.find_name[ord_name][0]
            ord_act = ord_sims
        else:
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

        # ----------------------------------------------------------- #
        #                           HResidu                           #
        # ----------------------------------------------------------- #

    def show_hr(self,
                ax1=None, ax2=None,
                cosmo=Planck15,
                sims_label='FITRES', show_leg=True,
                bins=20,
                fmt_sims='o', alpha_sims=0.2, fc_sims=cmap_tpw(0.2),
                ls_model='-', alpha_model=1, fc_model='C1',
                ticksize='x-large', fsize='x-large'):
        '''Plots the results of the FITRES and wfit on a HR plot with residu'''
        z_lin = np.linspace(np.min(self.sims_data['zCMB']),
                            np.max(self.sims_data['zCMB']),
                            1000)
        mu_cosmo = cosmo.distmod(z_lin)/units.mag

        if bins is None:
            if ax1 is not None and ax2 is None:
                ax1.errorbar(self.sims_data.zCMB, self.sims_data.MU,
                             xerr=self.sims_data.zCMBERR,
                             yerr=self.sims_data.MUERR,
                             fmt=fmt_sims, alpha=alpha_sims, color=fc_sims,
                             label='Fitres data')
                ax1.plot(z_lin, mu_cosmo,
                         ls=ls_model, alpha=alpha_model, color=fc_model,
                         label=cosmo.name)
                ax1.tick_params(labelsize=ticksize)
                ax1.set_xlabel('redshift', fontsize=fsize)
                ax1.set_ylabel(r'$\mu$', fontsize=fsize)

                if show_leg is True:
                    ax1.legend(ncol=1, loc='upper left')
                else:
                    pass

                return ax1
            if ax1 is None and ax2 is not None:
                ax2.errorbar(self.sims_data.zCMB, self.sims_data.MURES,
                             xerr=self.sims_data.zCMBERR,
                             yerr=self.sims_data.MUERR,
                             fmt=fmt_sims, alpha=alpha_sims, color=fc_sims,
                             label=sims_label)
                ax2.axhline(0, ls=ls_model, color=fc_model)
                ax2.tick_params(labelsize=ticksize)
                ax2.set_xlabel('redshift', fontsize=fsize)
                ax2.set_ylabel(r'$\Delta\mu$', fontsize=fsize)

                if show_leg is True:
                    ax2.legend(ncol=1, loc='upper left')
                else:
                    pass

                return ax2
            if ax1 is None and ax2 is None:
                fig, (ax1, ax2) = plt.subplots(2, 1,
                                               figsize=[10, 8], sharex=True)

            else:
                pass

            ax1.errorbar(self.sims_data.zCMB, self.sims_data.MU,
                         xerr=self.sims_data.zCMBERR,
                         yerr=self.sims_data.MUERR,
                         fmt=fmt_sims, alpha=alpha_sims, color=fc_sims,
                         label=sims_label)
            ax1.plot(z_lin, mu_cosmo,
                     color=fc_model, label=cosmo.name)
            ax2.errorbar(self.sims_data.zCMB, self.sims_data.MURES,
                         xerr=self.sims_data.zCMBERR,
                         yerr=self.sims_data.MUERR,
                         fmt=fmt_sims, alpha=alpha_sims, color=fc_sims)
            ax2.axhline(0, color=fc_model)

        else:
            sims_plot = self.sims_data.copy()
            sims_plot['zBIN'] = pd.cut(sims_plot.zCMB, bins)
            sims_plotg = sims_plot.groupby('zBIN').mean()
            sims_plotg['MUERR'] = sims_plot.groupby('zBIN')['MU'].std() /\
                sims_plot.groupby('zBIN')['MU'].size()
            sims_plotg['MURESERR'] = sims_plot.groupby('zBIN')['MURES'].std() /\
                sims_plot.groupby('zBIN')['MURES'].size()

            if ax1 is not None and ax2 is None:
                ax1.errorbar(sims_plotg.zCMB, sims_plotg.MU,
                             xerr=sims_plotg.zCMBERR,
                             yerr=sims_plotg.MUERR,
                             fmt=fmt_sims, alpha=alpha_sims, color=fc_sims,
                             label=sims_label)
                ax1.plot(z_lin, mu_cosmo,
                         ls=ls_model, alpha=alpha_model, color=fc_model,
                         label=cosmo.name)
                ax1.tick_params(labelsize=ticksize)
                ax1.set_xlabel('redshift', fontsize=fsize)

                if show_leg is True:
                    ax1.legend(ncol=1, loc='upper left')
                else:
                    pass

                return ax1
            if ax1 is None and ax2 is not None:
                ax2.errorbar(sims_plotg.zCMB, sims_plotg.MURES,
                             xerr=sims_plotg.zCMBERR,
                             yerr=sims_plotg.MURESERR,
                             fmt=fmt_sims, alpha=alpha_sims, color=fc_sims,
                             label=sims_label)
                ax2.axhline(0, ls=ls_model, color=fc_model)
                ax2.tick_params(labelsize=ticksize)
                ax2.set_xlabel('redshift', fontsize=fsize)

                if show_leg is True:
                    ax2.legend(ncol=1, loc='upper left')
                else:
                    pass

                return ax2
            if ax1 is None and ax2 is None:
                fig, (ax1, ax2) = plt.subplots(2, 1,
                                               figsize=[10, 8], sharex=True)

            else:
                pass

            ax1.errorbar(sims_plotg.zCMB, sims_plotg.MU,
                         xerr=sims_plotg.zCMBERR,
                         yerr=sims_plotg.MUERR,
                         fmt=fmt_sims, alpha=alpha_sims, color=fc_sims,
                         label=sims_label)
            ax1.plot(z_lin, mu_cosmo,
                     color=fc_model, label=cosmo.name)
            ax2.errorbar(sims_plotg.zCMB, sims_plotg.MURES,
                         xerr=sims_plotg.zCMBERR,
                         yerr=sims_plotg.MURESERR,
                         fmt=fmt_sims, alpha=alpha_sims, color=fc_sims)
            ax2.axhline(0, color=fc_model)

            ax1.tick_params(labelsize=ticksize)
            ax2.tick_params(labelsize=ticksize)
            ax2.set_xlabel('redshift', fontsize=fsize)
            ax1.set_ylabel(r'$\mu$', fontsize=fsize)
            ax2.set_ylabel(r'$\Delta\mu$', fontsize=fsize)

            if show_leg is True:
                ax1.legend(ncol=1, loc='upper left')
            else:
                pass

        return (ax1, ax2)

    def show_modif(self, ax=None, drop=14,
                   modif_label='Mudif', show_leg=True,
                   fmt_sims='o', color_sims='k'):
        if ax is None:
            fig = plt.figure(figsize=[8, 7])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])
        if drop is not None:
            modif_plot = self.modif.drop([drop])
        else:
            modif_plot = self.modif
        ax.errorbar(modif_plot['z'], modif_plot['MUDIF'],
                    yerr=modif_plot['MUDIFERR'],
                    fmt=fmt_sims, color=color_sims,
                    label=modif_label)
        if show_leg is True:
            ax.legend(ncol=1, loc='upper left')
