from math import ceil
from astropy.io import fits
from astropy.table import Table
import scipy
import numpy as np
import os

def SearchLineChunked(args, sigma_z, sigma_xy, EPS):
    truncate = 4.0

    print(100 * '#')
    print('Starting search of lines with sigmas =', (sigma_z, sigma_xy))
    hz, hy, hx = gaussian_halo(sigma_z, 
                               sigma_xy * EPS *args.FractionEPS, 
                               sigma_xy * EPS *args.FractionEPS, 
                               truncate = truncate)
    print('Halo sizes:', (hz, hy, hx))

    positive_tables = []
    negative_tables = []

    chunk_id = 0
    hdulist, cube = get_cube_data(args.Cube)

    for z0, z1, y0, y1, x0, x1 in plan_chunks(cube.shape, args.Chunk):
        print(
            f'Processing chunk {chunk_id}: '
            f'Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}]'
        )

        block, core_in_block, halo_slices = extract_block_with_halo(
            cube, z0, z1, y0, y1, x0, x1, hz, hy, hx
        )

        filtered = scipy.ndimage.gaussian_filter(
            block,
            sigma= [
                sigma_z, 
                sigma_xy * EPS *args.FractionEPS, 
                sigma_xy * EPS *args.FractionEPS
                ], 
            mode="constant",
            cval=0.0,
            truncate=truncate,
        )

        core = filtered[core_in_block]

        mask = None
        if args.UseMask:
            mask = get_mask(args.ContinuumImage, args.MaskSN)
            print('Using continuum mask with shape:', mask.shape)

        if args.UseMask:
            mask_core = mask[y0:y1, x0:x1]
            pass
        else:
            mask_core = None

        core_norm = normalize_block_channelwise(
            core,
            use_mask=args.UseMask,
            mask_block=mask_core,
        )

        pos_table, neg_table = detect_candidates(
            core_norm,
            args.MinSN,
            z0=z0,
            y0=y0,
            x0=x0,
        )

        if len(pos_table) > 0:
            positive_tables.append(pos_table)

        if len(neg_table) > 0:
            negative_tables.append(neg_table)

        print(
            f'Chunk {chunk_id} -> positive: {len(pos_table)} | negative: {len(neg_table)}'
        )
        chunk_id += 1

    hdulist.close()
    
    tpos = concatenate_tables(positive_tables)
    tneg = concatenate_tables(negative_tables)

    pos_name = os.path.join(
        args.OutputPath,
        f'line_candidates_{sigma_z}_{sigma_xy}_pos.fits'
    )
    neg_name = os.path.join(
        args.OutputPath,
        f'line_candidates_{sigma_z}_{sigma_xy}_neg.fits'
    )

    tpos.write(pos_name, format='fits', overwrite=True)
    tneg.write(neg_name, format='fits', overwrite=True)

    print('Positive pixels in search for sigmas:', (sigma_z,sigma_xy), 'N:', len(tpos))
    print('Negative pixels in search for sigmas:', (sigma_z, sigma_xy), 'N:', len(tneg))

def gaussian_halo(sigmaz, sigmay, sigmax, truncate=4.0):
    return (
        int(ceil(truncate * sigmaz)),
        int(ceil(truncate * sigmay)),
        int(ceil(truncate * sigmax)),
    )


def plan_chunks(shape, chunk_size = None):
    zsize, ysize, xsize = shape
    x0 = y0 = 0
    x1 = xsize
    y1 = ysize
    for z0 in range(0, zsize, chunk_size):
        z1 = min(z0 + chunk_size, zsize)

    #for y0 in range(0, ysize, chunk_size):
    #    y1 = min(y0 + chunk_size, ysize)

    #    for x0 in range(0, xsize, chunk_size):
    #        x1 = min(x0 + chunk_size, xsize)

        yield (z0, z1, y0, y1, x0, x1)


def get_cube_data(cube_path):
    hdul = fits.open(cube_path, memmap=True)
    data = hdul[0].data

    if len(np.shape(data)) == 4:
        cube = data[0]
    else:
        cube = data

    if cube.ndim != 3:
        hdul.close()
        raise ValueError("Cube must be 3D array (Spectral, Spatial, Spatial).")

    return hdul, cube

def extract_block_with_halo(cube, z0, z1, y0, y1, x0, x1, hz, hy, hx):
    z0h = max(0, z0 - hz)
    z1h = min(cube.shape[0], z1 + hz)

    y0h = max(0, y0 - hy)
    y1h = min(cube.shape[1], y1 + hy)

    x0h = max(0, x0 - hx)
    x1h = min(cube.shape[2], x1 + hx)

    block = np.asarray(cube[z0h:z1h, y0h:y1h, x0h:x1h], dtype=np.float32)

    core_in_block = (
        slice(z0 - z0h, z0 - z0h + (z1 - z0)),
        slice(y0 - y0h, y0 - y0h + (y1 - y0)),
        slice(x0 - x0h, x0 - x0h + (x1 - x0)),
    )

    halo_slices = (z0h, z1h, y0h, y1h, x0h, x1h)
    return block, core_in_block, halo_slices

def normalize_block_channelwise(core_block, use_mask=False, mask_block=None):
    """
    - Máscara
    - InitialRMS y FinalRMS 
    - Normaliza  por FinalRMS
    """
    core_block = core_block.copy()

    for i in range(len(core_block)):
        if use_mask and mask_block is not None:
            core_block[i][mask_block] = np.nan

        initial_rms = np.nanstd(core_block[i])

        valid = core_block[i][core_block[i] < 5.0 * initial_rms]
        final_rms = np.nanstd(valid)

        if not np.isfinite(final_rms) or final_rms == 0:
            core_block[i] = np.nan
        else:
            core_block[i] = core_block[i] / final_rms

    return core_block


def detect_candidates(core_norm_block, min_sn, z0, y0, x0):
    pix1, pix2, pix3 = np.where(core_norm_block >= min_sn)
    if len(pix1) == 0:
        pos_table = Table(
            names=("Channel", "Xpix", "Ypix", "SN")
        )
    else:
        pos_table = Table(
            [
                pix1 + z0,
                pix3 + x0,
                pix2 + y0,
                core_norm_block[pix1, pix2, pix3],
            ],
            names=("Channel", "Xpix", "Ypix", "SN"),
        )

    neg_block = -1.0 * core_norm_block
    pix1n, pix2n, pix3n = np.where(neg_block >= min_sn)

    if len(pix1n) == 0:
        neg_table = Table(
            names=("Channel", "Xpix", "Ypix", "SN")
        )
    else:
        neg_table = Table(
            [
                pix1n + z0,
                pix3n + x0,
                pix2n + y0,
                neg_block[pix1n, pix2n, pix3n],
            ],
            names=("Channel", "Xpix", "Ypix", "SN"),
        )

    return pos_table, neg_table

def concatenate_tables(tables):
    if len(tables) == 0:
        return Table(
            names=("Channel", "Xpix", "Ypix", "SN"),
        )

    if len(tables) == 1:
        return tables[0]

    from astropy.table import vstack
    return vstack(tables, metadata_conflicts="silent")


def get_mask(continuum_image, mask_sn):
    hdu = fits.open(continuum_image, memmap=True)
    data = hdu[0].data

    if data.ndim == 4:
        data_mask = data[0][0]
    elif data.ndim == 3:
        data_mask = data[0]
    else:
        data_mask = data

    initial_rms = np.nanstd(data_mask)
    final_rms = np.nanstd(data_mask[data_mask < mask_sn * initial_rms])
    mask = np.where(data_mask >= mask_sn * final_rms, True, False)

    hdu.close()
    return mask