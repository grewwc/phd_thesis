from utils import *

from interfaces import *
from python_tools.NotEfficient.functions import Override
from astropy.stats import sigma_clip


class V1(MathOpsBase):
    @Override
    def mask_outliers(self, data, n_sigma=3, *args):
        return sigma_clip(data, n_sigma).mask
       
