"""
Algorithm 5 - Merging coarse lung regions using 3D connectivity
(manuscript Section 4.5).

Starting from the centre slice of the lung (assumed correct), slices are
processed outwards. For slice n with past best pB (previous slice), pB2 (two
slices back), current best cB and current options cP (= BW4reg, four largest
regions from Algorithm 2):

    LD = XOR(pB, cB) AND p2B                    abrupt change, confirmed 2 slices back
    if |LD| > 10 % |pB|:
        LC = LD AND cP                          region available among the 4 largest
        if |LC| > 95 % |LD|: cB = cB OR LC      restore the lost region

The manuscript skips slice nMid+1 in the front half; here both neighbours of
the centre slice are handled symmetrically.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .config import Config


def _correct(pB, p2B, cB, cP, cfg: Config):
    ld = (pB ^ cB) & p2B
    if ld.sum() > cfg.abrupt_change_fraction * max(pB.sum(), 1):
        lc = ld & cP
        if lc.sum() > cfg.correction_overlap_fraction * ld.sum():
            return cB | lc, lc
    return cB, None


def connectivity_3d(right_vol, left_vol, bw4reg_vol, cfg: Config):
    bw_lung = right_vol | left_vol
    nz = bw_lung.shape[0]
    occupied = np.where(bw_lung.reshape(nz, -1).any(1))[0]
    final = bw_lung.copy()
    restored = np.zeros_like(bw_lung)
    if occupied.size < 3:
        return right_vol, left_vol, restored
    n_mid = int((occupied[0] + occupied[-1]) // 2)

    for direction in (-1, +1):
        n = n_mid + direction
        while 0 <= n < nz:
            pB = final[n - direction]
            back2 = n - 2 * direction
            p2B = final[back2] if 0 <= back2 < nz else pB
            new, lc = _correct(pB, p2B, final[n], bw4reg_vol[n], cfg)
            final[n] = new
            if lc is not None:
                restored[n] |= lc
            n += direction

    # assign restored pixels to the side they are closest to
    right = right_vol.copy()
    left = left_vol.copy()
    for z in np.where(restored.reshape(nz, -1).any(1))[0]:
        dr = ndi.distance_transform_edt(~right[z]) if right[z].any() else np.full(right[z].shape, np.inf)
        dl = ndi.distance_transform_edt(~left[z]) if left[z].any() else np.full(left[z].shape, np.inf)
        add = restored[z] & ~right[z] & ~left[z]
        right[z] |= add & (dr <= dl)
        left[z] |= add & (dl < dr)

    if cfg.remove_small_3d_fraction > 0:
        # Remove isolated 3D fragments (< x % of the largest component of that
        # side). Keeping ONLY the largest component is unsafe: one failed slice
        # can split a lung into two large 3D pieces.
        struct = ndi.generate_binary_structure(3, 3)
        for side in (right, left):
            lab, n = ndi.label(side, structure=struct)
            if n > 1:
                sizes = np.bincount(lab.ravel())
                sizes[0] = 0
                keep = sizes >= cfg.remove_small_3d_fraction * sizes.max()
                keep[0] = False
                side[:] = keep[lab]
    return right, left, restored
