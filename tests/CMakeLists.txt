# Petrophysics-only test/exec targets.
#
# This branch contains the three petrophysical protocols:
#   k   -- absolute permeability       (lbpm_permeability_simulator)
#   pc  -- capillary pressure          (lbpm_color_simulator + lbpm_morphopen_pp)
#   kr  -- relative permeability       (lbpm_relperm_simulator)
#
# Plus the pre-processors needed to feed them.

# Pre-processors (run in order: mirror -> decomp -> [morphopen for kr/pc])
ADD_LBPM_EXECUTABLE( lbpm_mirror_pp )             # interporous mirror -> smooth periodic seam
ADD_LBPM_EXECUTABLE( lbpm_serial_decomp )         # raw -> per-rank ID.xxxxx
ADD_LBPM_EXECUTABLE( lbpm_serial_uCT_decomp )     # uCT input variant
ADD_LBPM_EXECUTABLE( lbpm_uCT_pp )                # uCT post-segmentation pp
ADD_LBPM_EXECUTABLE( lbpm_morphopen_pp )          # morph drainage -> Sw (kr init / pc curve)

# k -- single-phase absolute permeability (MRT)
ADD_LBPM_EXECUTABLE( lbpm_permeability_simulator )

# pc -- color-model drainage / displacement (pcProtocolFlag, Sw vs pc curve)
ADD_LBPM_EXECUTABLE( lbpm_color_simulator )
ADD_LBPM_EXECUTABLE( lbpm_color_simulator_SI )    # SI-unit interface variant

# kr -- color + automorph with per-event single-phase BGK on connected components
ADD_LBPM_EXECUTABLE( lbpm_relperm_simulator )

# GPU sanity check (useful before launching a real run on a multi-GPU box)
ADD_LBPM_EXECUTABLE( checkGPUs )
