# snage
Type Ia supernova prompt vs delayed underlying population

# Installation

using git

# Concept

The package is made to generate random sample realisation of SN sample as they are in nature (i.e. prior selection function). 
Conceptually, SNe Ia are assumed to be sample mixture of underlying populations that each have their own intrinsic distribution of properties. These may be redshift dependent as well as the mixture composition.

Currently, only the Prompt and Delayed dichotomy has been implemented based on [Rigault et al. 2018](https://ui.adsabs.harvard.edu/abs/2018arXiv180603849R/abstract) and [Nicolas et al. 2O20](https://ui.adsabs.harvard.edu/abs/2020arXiv200509441N/abstract). 

Underlying distributions of SN properties such as stretch, color or host mass have been modelled for both prompt and delayed populations. Gaussian, Gaussian mixture or asymetric Gaussians are currently used for this modelling and default distribution parameters are provided. One can then draw random sample realisations assuming a fraction of prompt SNeIa. This way the underlying correlation between all the SN properties are accurately handled, assuming they as solely due to their correlation with the age population.

# Current limits

The prompt and delayed modeling of the host mass step is not satisfying yet:
 - It is not redshift dependent while it must
 - Even at z~0 it is not great yet.
 
# Usage

Load an instance of the prompt and delayed object (`PrompDelayModel`)

```python
from snprop import age
pdmodel = age.PrompDelayModel()
```

To vizualise the underlying distribution of say, stretch, at _z=0.05_, _z=0.5_ and _z=1_ (the color is the redshift (blue to red) up to, in that case, `zmax=1`.

```python
fig = pdmodel.show_pdf("stretch", z=[0.05, 0.5, 1], zmax=1)
```

<p align="left">
  <img src="figures/snstretch_pdfs.png" width="350" title="hover text">
</p>

Then to draw a random realisation of a sample of 300 SNeIa, made of 40% of prompt ones, simply do:

```python
pdmodel.draw_sample(0.4, size=300)
```
the sample is stored as a pandas datafraom in 
```python
pdmodel.sample
```
```
|     |        color |    stretch |     mass |          hr |   prompt | redshift   |
|----:|-------------:|-----------:|---------:|------------:|---------:|:-----------|
|   0 | -0.0747748   | -0.765766  |  9.11812 | -0.039039   |        1 |            |
|   1 |  0.0675676   |  0.555556  |  9.37037 |  0.107107   |        1 |            |
|   2 | -0.0225225   | -0.405405  |  9.22322 |  0.045045   |        1 |            |
|   3 | -0.0441441   |  0.975976  |  8.73273 |  0.049049   |        1 |            |
|   4 |  0.0189189   |  0.725726  | 10.036   |  0.129129   |        1 |            |
...
| 295 |  0.0216216   |  0.495495  | 11.0941  |  0.011011   |        0 |            |
| 296 | -0.0459459   | -1.42643   | 11.2482  | -0.0630631  |        0 |            |
| 297 |  0.027027    |  1.13614   | 10.4775  | -0.049049   |        0 |            |
| 298 |  0.123423    | -0.115115  |  8.94995 |  0.00900901 |        0 |            |
| 299 | -0.116216    | -0.895896  | 10.4494  |  0.001001   |        0 |            |
```
