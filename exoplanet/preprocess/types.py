

class TCEMeta(type):
    def __init__(cls, cls_name, bases, class_attrs):
        super().__init__(cls, cls_name, bases)
        _attrs = class_attrs['_attrs']
        for column in _attrs:
            setattr(cls, column, None)


class Item:
    def __init__(self):
        self._values = {}

    def __get__(self, instance, instance_type):
        return self._values.get(instance, None)

    def __set__(self, instance, value):
        self._values[instance] = value


class TCE(metaclass=TCEMeta):
    _attrs = ['kepid', 'tce_plnt_num', 'tce_period',
              'tce_period_err', 'tce_time0bk', 'tce_time0bk_err', 'tce_impact',
              'tce_impact_err', 'tce_duration', 'tce_duration_err', 'tce_depth',
              'tce_depth_err', 'tce_model_snr', 'tce_prad', 'tce_prad_err', 'tce_eqt',
              'tce_eqt_err', 'tce_steff',
              'tce_steff_err', 'tce_slogg', 'tce_slogg_err', 'tce_sradius',
              'tce_sradius_err', 'av_training_set']
