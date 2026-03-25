
import tifffile
import numpy as np
import os
import re
from glob import glob
import argparse

def args():
    parser = argparse.ArgumentParser(description='Convert LBPM rawVis Part files to 3D TIFF stacks')
    parser.add_argument('--test_dir', dest='test_dir', default='./', help='directory containing rawVis* folders and inputFile.db')
    return parser.parse_args()

args = args()

def read_si_conversion(test_dir):
    """Read si_conversion.db if present. Returns dict of conversion factors or None."""
    path = os.path.join(test_dir, 'si_conversion.db')
    if not os.path.isfile(path):
        return None
    params = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=', 1)
                params[key.strip()] = float(val.strip())
    return params

def extract_values(file_path):
    """Parse the global domain size N from an LBPM input database file."""
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('N =') or line.startswith('N='):
                parts = line.split('=', 1)[1]
                nums = [int(x.strip().rstrip(',')) for x in parts.split(',')]
                return nums[0], nums[1], nums[2]
    raise RuntimeError(f"Could not find 'N = ...' in {file_path}")

def extract_filename(file_path):
    """Parse the geometry Filename from an LBPM input database file."""
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Filename'):
                val = line.split('=', 1)[1].strip().strip('"').strip("'")
                return val
    return "geopack.raw"

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# ---------- Setup ----------
input_db = os.path.join(args.test_dir, 'inputFile.db')
# Fall back to Color.db if inputFile.db doesn't exist
if not os.path.isfile(input_db):
    input_db = os.path.join(args.test_dir, 'Color.db')
nx, ny, nz = extract_values(input_db)
geo_file = extract_filename(input_db)
print(f"nx: {nx}, ny: {ny}, nz: {nz}, geo: {geo_file}")

geo_path = os.path.join(args.test_dir, geo_file)
with open(geo_path, 'rb') as fileID:
    solid = np.fromfile(fileID, dtype=np.int8, count=nx*ny*nz)
solid = solid.reshape((nz, ny, nx))

# Read SI conversion metadata if present
si_params = read_si_conversion(args.test_dir)
si_mode = si_params is not None
if si_mode:
    vel_scale  = si_params['dx_si'] / si_params['dt_si']
    pres_scale = si_params['rho_ref'] * vel_scale * vel_scale
    print(f"SI mode: dx={si_params['dx_si']:.4e} m, dt={si_params['dt_si']:.4e} s, rho_ref={si_params['rho_ref']:.4f} kg/m^3")
    print(f"  vel_scale = {vel_scale:.6e} m/s per LB unit, pres_scale = {pres_scale:.6e} Pa per LB unit")
else:
    vel_scale  = 1.0
    pres_scale = 1.0
    print("Lattice-unit mode (no si_conversion.db found)")

steadyTime = sorted(glob(os.path.join(args.test_dir, 'rawVis*')), key=natural_keys)
if not steadyTime:
    print("No rawVis* directories found. If using in-code TIFF output, this script is not needed.")
    exit(0)
print(f"Found {len(steadyTime)} rawVis directories")

def loadLBPMPhaseMap(time_str):

    files = [file for file in os.listdir(time_str) if file.startswith("Part")]
    
    numParts = len(files)
    file = files[0]
    fileMetaData = file[:-4].split('_')
    partLoc = int(fileMetaData[1])
    Nx, Ny, Nz = int(fileMetaData[2]), int(fileMetaData[3]), int(fileMetaData[4])
    NX, NY, NZ = int(fileMetaData[5]), int(fileMetaData[6]), int(fileMetaData[7])

    domain = np.zeros((NZ * (Nz-2), NY * (Ny-2), NX * (Nx-2)), dtype=np.uint8)

    for file in files:
        print(f"{time_str}: {file}")
        fileMetaData = file[:-4].split('_')
        partLoc = int(fileMetaData[1])
        Nx, Ny, Nz = int(fileMetaData[2]), int(fileMetaData[3]), int(fileMetaData[4])
        NX, NY, NZ = int(fileMetaData[5]), int(fileMetaData[6]), int(fileMetaData[7])

        with open(f"{time_str}/{file}", 'rb') as fileID:
            data = np.fromfile(fileID, dtype=np.uint8, count=Nx*Ny*Nz)
            data = np.reshape(data, (Nz, Ny, Nx))

        z, y, x = np.unravel_index(partLoc, (NZ, NY, NX))
        localOrigin = np.array([z, y, x]) * np.array([Nz-2, Ny-2, Nx-2])
        domain[localOrigin[0]:localOrigin[0]+(Nz-2),
               localOrigin[1]:localOrigin[1]+(Ny-2),
               localOrigin[2]:localOrigin[2]+(Nx-2)] = data[1:-1, 1:-1, 1:-1]

    return domain
    
    
    
def loadLBPMVelPMap(time_str):

    files = [file for file in os.listdir(time_str) if file.startswith("Part")]
    
    numParts = len(files)
    file = files[0]
    fileMetaData = file[:-4].split('_')
    partLoc = int(fileMetaData[1])
    Nx, Ny, Nz = int(fileMetaData[2]), int(fileMetaData[3]), int(fileMetaData[4])
    NX, NY, NZ = int(fileMetaData[5]), int(fileMetaData[6]), int(fileMetaData[7])

    domainx = np.zeros((NZ * (Nz-2), NY * (Ny-2), NX * (Nx-2)), dtype=np.double)
    domainy = np.zeros((NZ * (Nz-2), NY * (Ny-2), NX * (Nx-2)), dtype=np.double)
    domainz = np.zeros((NZ * (Nz-2), NY * (Ny-2), NX * (Nx-2)), dtype=np.double)
    domainp = np.zeros((NZ * (Nz-2), NY * (Ny-2), NX * (Nx-2)), dtype=np.double)
    
    for file in files:
        print(f"{time_str}: {file}")
        fileMetaData = file[:-4].split('_')
        partLoc = int(fileMetaData[1])
        Nx, Ny, Nz = int(fileMetaData[2]), int(fileMetaData[3]), int(fileMetaData[4])
        NX, NY, NZ = int(fileMetaData[5]), int(fileMetaData[6]), int(fileMetaData[7])

        with open(f"{time_str}/../ID.{partLoc:05}", 'rb') as fileID:
            subdomain = np.fromfile(fileID, dtype=np.int8, count=Nx*Ny*Nz)
            subdomain = np.reshape(subdomain, (Nz, Ny, Nx))

        with open(f"{time_str}/{file}", 'rb') as fileID2:
            data = np.fromfile(fileID2, dtype=np.float64, count=np.sum(subdomain > 0)*4)
            data = np.reshape(data, (4,-1))
        
        interior = subdomain[1:-1, 1:-1, 1:-1]
        fluid_mask = interior > 0
        fields = np.zeros((subdomain.shape[0], subdomain.shape[1], subdomain.shape[2], 4))
        for k in range(4):
            component = np.zeros_like(interior, dtype=np.float64)
            component[fluid_mask] = data[k, :]
            fields[1:-1, 1:-1, 1:-1, k] = component
        data = fields

        z, y, x = np.unravel_index(partLoc, (NZ, NY, NX))
        localOrigin = np.array([z, y, x]) * np.array([Nz-2, Ny-2, Nx-2])
        domainx[localOrigin[0]:localOrigin[0]+(Nz-2),
               localOrigin[1]:localOrigin[1]+(Ny-2),
               localOrigin[2]:localOrigin[2]+(Nx-2)] = data[1:-1, 1:-1, 1:-1,0]
               
        domainy[localOrigin[0]:localOrigin[0]+(Nz-2),
               localOrigin[1]:localOrigin[1]+(Ny-2),
               localOrigin[2]:localOrigin[2]+(Nx-2)] = data[1:-1, 1:-1, 1:-1,1]
               
        domainz[localOrigin[0]:localOrigin[0]+(Nz-2),
               localOrigin[1]:localOrigin[1]+(Ny-2),
               localOrigin[2]:localOrigin[2]+(Nx-2)] = data[1:-1, 1:-1, 1:-1,2]
               
        domainp[localOrigin[0]:localOrigin[0]+(Nz-2),
               localOrigin[1]:localOrigin[1]+(Ny-2),
               localOrigin[2]:localOrigin[2]+(Nx-2)] = data[1:-1, 1:-1, 1:-1,3]

    return domainx, domainy, domainz, domainp

# ---------- Main processing loop ----------
for time_str in steadyTime:
    time_string = time_str.split('/')[-1]
    if 'Vel' not in time_string:
        time_step = int(time_string[6:])
        fname = f"{args.test_dir}/colortest_{time_step:09}.tif"

        if not os.path.isfile(fname):
            try:
                SRSubSeg = loadLBPMPhaseMap(time_str)
                SRSubSeg = SRSubSeg[:nz, :ny, :nx]
                SRSubSeg[SRSubSeg < 127.5] = 0
                SRSubSeg[solid <= 0] = 1
                SRSubSeg[SRSubSeg > 127.5] = 2
                ijmeta = {'axes': 'ZYX'}
                tifffile.imwrite(fname, SRSubSeg, compression='zlib', imagej=True, metadata=ijmeta)
                print(f"  Phase:  {fname}")
            except Exception as e:
                print(f"An error occurred processing {time_str}: {e}")
    else:
        time_step = int(time_string[10:])

        if si_mode:
            fname = f"{args.test_dir}/colortestVelMag_SI_{time_step:09}.tif"
        else:
            fname = f"{args.test_dir}/colortestNormVelMag_{time_step:09}.tif"

        if not os.path.isfile(fname):
            try:
                domainx, domainy, domainz, domainp = loadLBPMVelPMap(time_str)

                domainx = domainx[:nz, :ny, :nx]
                domainx[solid <= 0] = 0
                domainy = domainy[:nz, :ny, :nx]
                domainy[solid <= 0] = 0
                domainz = domainz[:nz, :ny, :nx]
                domainz[solid <= 0] = 0
                domainp = domainp[:nz, :ny, :nx]
                domainp[solid <= 0] = 0

                velMag = np.sqrt(domainx*domainx + domainy*domainy + domainz*domainz)

                ijmeta = {'axes': 'ZYX'}
                if si_mode:
                    domainx *= vel_scale
                    domainy *= vel_scale
                    domainz *= vel_scale
                    domainp *= pres_scale
                    velMag  *= vel_scale
                    print(f"  VelP timestep {time_step}: |u|_max = {np.max(velMag):.4e} m/s, "
                          f"P range = [{np.min(domainp[solid>0]):.4e}, {np.max(domainp[solid>0]):.4e}] Pa")
                    tifffile.imwrite(fname, np.float32(velMag), compression='zlib', imagej=True, metadata=ijmeta)
                    tifffile.imwrite(f"{args.test_dir}/colortestVx_SI_{time_step:09}.tif", np.float32(domainx), compression='zlib', imagej=True, metadata=ijmeta)
                    tifffile.imwrite(f"{args.test_dir}/colortestVy_SI_{time_step:09}.tif", np.float32(domainy), compression='zlib', imagej=True, metadata=ijmeta)
                    tifffile.imwrite(f"{args.test_dir}/colortestVz_SI_{time_step:09}.tif", np.float32(domainz), compression='zlib', imagej=True, metadata=ijmeta)
                    tifffile.imwrite(f"{args.test_dir}/colortestP_SI_{time_step:09}.tif", np.float32(domainp), compression='zlib', imagej=True, metadata=ijmeta)
                else:
                    normVelMag = velMag / np.mean(velMag[velMag > 0]) if np.any(velMag > 0) else velMag
                    normVelMag = np.uint8(np.clip(normVelMag, 0, 255))
                    tifffile.imwrite(fname, normVelMag, compression='zlib', imagej=True, metadata=ijmeta)
                    tifffile.imwrite(f"{args.test_dir}/colortestVx_{time_step:09}.tif", domainx, compression='zlib', imagej=True, metadata=ijmeta)
                    tifffile.imwrite(f"{args.test_dir}/colortestVy_{time_step:09}.tif", domainy, compression='zlib', imagej=True, metadata=ijmeta)
                    tifffile.imwrite(f"{args.test_dir}/colortestVz_{time_step:09}.tif", domainz, compression='zlib', imagej=True, metadata=ijmeta)
                    tifffile.imwrite(f"{args.test_dir}/colortestP_{time_step:09}.tif", domainp, compression='zlib', imagej=True, metadata=ijmeta)
            except Exception as e:
                print(f"An error occurred processing {time_str}: {e}")
