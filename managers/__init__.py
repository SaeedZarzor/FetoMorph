# Intentionally left without eager re-exports.
#
# Importing manager classes here (e.g. ``from .measurement_dispatcher import
# MeasurementDispatcher``) creates a circular import: ``measurements_image``
# imports ``managers.visualization_settings``, which runs this ``__init__`` and
# pulls in ``measurement_dispatcher``, whose ``from functions.measurements_image
# import *`` then runs against a half-initialised ``measurements_image`` and
# binds none of its ``compute_image_*`` functions.
#
# All call sites import managers fully-qualified
# (``from managers.measurement_dispatcher import MeasurementDispatcher``), so no
# package-level re-exports are needed.
