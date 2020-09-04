import numpy as np
from scipy import stats
def asym_gaussian(x, mu, sigmaup, sigmadown, dx=None):
    """ """
    if dx is None:
        varup   = sigmaup**2
        vardown = sigmadown**2
    else:
        varup   = sigmaup**2 + dx**2
        vardown = sigmadown**2 + dx**2
        
    pdf       = np.exp( -(x-mu)**2 / (2*varup) )  # up
    pdf[x<mu] = np.exp( -(x[x<mu]-mu)**2 / (2*vardown)  )
    norm = np.sqrt(2*np.pi * (0.5*varup+0.5*vardown) )
    return pdf/norm
                    
    
class AsymGaussian( object ):
    """ """
    
    @classmethod
    def from_data(cls, data, error=None, weights=None):
        """ """
        this = cls()
        fitout = this.fit(data, error=error, weights=weights)
        print(fitout)
        return this
    
    # ======= #
    # Methods #
    # ======= #
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
            return np.sum( -2*np.log(weights*self.pdf(data, dx=error)) )
        
        if guess is None:
            guess = [np.nanmean(data), np.nanstd(data)/2,np.nanstd(data)/2]
            
        return optimize.fmin(get_loglikelihood, guess)
        
    def set_param(self, mu, sigmaup, sigmadown):
        """ """
        self._param = {"mu":mu, "sigmaup":sigmaup, "sigmadown":sigmadown}
        
    def show(self, ax=None, data=None, error=None, 
             show_legend=True,
             dataprop={}, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[6,4])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
           
        xx = np.linspace(self.mu-8*self.sigmadown, self.mu+8*self.sigmadown, 1000)
        
        ax.plot(xx, self.pdf(xx), label="model", **kwargs)
        if data is not None:
            if error is None:
                ax.hist(data, normed=True, label="data", **dataprop)
            else:
                g = stats.norm.pdf(xx[:,None], loc=data, scale=error)
                ax.plot(xx, np.sum(g, axis=1)/len(data), label="data", **dataprop)
            
        if show_legend:
            ax.legend(loc="best", frameon=False)
        return fig
    # =========== #
    #  Properties #
    # =========== #
    @property
    def param(self):
        """ """
        if not hasattr(self,"_param"):
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
    
