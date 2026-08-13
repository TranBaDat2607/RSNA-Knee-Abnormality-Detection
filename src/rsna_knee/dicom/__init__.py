"""DICOM ingestion: header probing, laterality resolution, slot selection, physical-scale
slice sampling, and the in-memory training cache.

Pipeline order: :func:`header.walk` + :func:`header.annotate` recover sequence typing from
DICOM headers -> :func:`laterality.laterality_maps` resolves left/right per study ->
:func:`slots.pick_slots` chooses one series per slot per study -> :func:`cache.build_cache`
decodes every chosen series once into a shared ``uint8`` array via
:func:`sampling.read_slot`.
"""
