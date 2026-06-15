-- Bounding box (plus source frame dimensions) of the detection that produced
-- the thumbnail frame, so the gallery can zoom the cover image into the animal.
-- JSON: {"box":[x,y,w,h], "fw":<int>, "fh":<int>} with box normalised to [0,1].
-- NULL for sightings recorded before this migration (they fall back to the
-- full-frame thumbnail).
ALTER TABLE sightings ADD COLUMN thumb_bbox TEXT;
