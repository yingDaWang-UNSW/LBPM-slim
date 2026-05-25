/*
  lbpm_mirror_pp -- interporous mirroring preprocessor.

  Problem: when running under periodic body-force BCs (BC=0), the seam
  between the outlet face and the inlet face of the domain must be
  geometrically continuous, or the flow sees a discontinuity at the
  wrap-around (fluid streaming into a wall, body force not propagating
  cleanly along the flow axis).

  Fix: insert a transition slab between the outlet face and the wrap-
  around-to-inlet, whose porosity and pore structure interpolate smoothly
  between the two faces.  Concretely, for each requested axis:

    1. Take the first and last slices A1, A2 of the geometry along that
       axis.
    2. Compute the signed Euclidean distance map of each slice (deep in
       solid is positive, deep in fluid is negative), normalise to [0,1].
    3. For each padding index t in 1..pad:
         W(t)  = linspace(1/pad, 1 - 1/pad, pad)[t-1]
         B(t)  = W * B1 + (1 - W) * B2
         phi(t) = W * porosity(A1) + (1 - W) * porosity(A2)
         slice(t) = ( B(t) >= quantile(B(t), phi(t)) )    -- 1 = solid
    4. Append the stack of pad slices to the end of the axis.

  Output: the same raw on disk plus the new dimensions.  Run this BEFORE
  any other preprocessor (decomp / morphopen) so that all subsequent
  stages see the extended geometry.

  Input db fields (Domain section):
    Filename            -- input raw
    N = [Nx, Ny, Nz]    -- input raw dimensions
    ReadType            -- "8bit" (default) or "16bit"
    MirrorPad = [px,py,pz]    (new) -- transition slab thicknesses per axis;
                                       0 means skip that axis.  Just needs to
                                       be a handful of voxels (~16 is plenty)
                                       to smooth the porosity across the
                                       periodic seam; no need to double the
                                       domain.
    MirrorOutput        (optional) -- output filename, defaults to
                                       <Filename>_mirrored.raw

  Output convention: 0 = solid, 1 = fluid (binary).  Multi-label inputs
  (e.g. post-morphopen 0/1/2) are flattened to solid/fluid for the mirror
  -- run this stage BEFORE morphopen.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/Database.h"
#include "common/MPI_Helpers.h"

using namespace std;

// ============================================================================
// 1D squared-Euclidean distance transform (Felzenszwalb-Huttenlocher).
//
// Given f[i] (the "initial cost" at point i: 0 if i is an object point,
// +INF otherwise), produces d[i] = min over j of ( (i-j)^2 + f[j] ).
// Linear time in n.
// ============================================================================
static void edt1d(const vector<double> &f, vector<double> &d) {
    const double INF = 1e300;
    int n = (int)f.size();
    if (n <= 0) return;
    vector<int>    v(n);
    vector<double> z(n + 1);
    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] =  INF;
    for (int q = 1; q < n; q++) {
        double s;
        while (true) {
            double num = (f[q] + double(q) * q) - (f[v[k]] + double(v[k]) * v[k]);
            double den = 2.0 * (q - v[k]);
            s = num / den;
            if (s > z[k]) break;
            k--;
        }
        k++;
        v[k]     = q;
        z[k]     = s;
        z[k + 1] = INF;
    }
    k = 0;
    for (int q = 0; q < n; q++) {
        while (z[k + 1] < q) k++;
        double dq = double(q - v[k]);
        d[q] = dq * dq + f[v[k]];
    }
}

// 2D squared EDT: for each pixel, the squared distance to the nearest
// "object" pixel (obj[i] != 0).
static void edt2d_squared(int W, int H,
                          const vector<unsigned char> &obj,
                          vector<double> &dist2) {
    const double INF = 1e300;
    dist2.assign((size_t)W * H, INF);
    for (size_t i = 0; i < (size_t)W * H; i++)
        if (obj[i]) dist2[i] = 0.0;

    // Pass 1: rows
    vector<double> row(W), drow(W);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) row[x] = dist2[(size_t)y * W + x];
        edt1d(row, drow);
        for (int x = 0; x < W; x++) dist2[(size_t)y * W + x] = drow[x];
    }
    // Pass 2: columns
    vector<double> col(H), dcol(H);
    for (int x = 0; x < W; x++) {
        for (int y = 0; y < H; y++) col[y] = dist2[(size_t)y * W + x];
        edt1d(col, dcol);
        for (int y = 0; y < H; y++) dist2[(size_t)y * W + x] = dcol[y];
    }
}

// Signed normalised distance: positive deep in solid, negative deep in
// fluid, rescaled to [0,1].  solidMask[i] == 1 marks solid voxels.
static void signedNormalizedDistance(int W, int H,
                                     const vector<unsigned char> &solidMask,
                                     vector<double> &B) {
    vector<unsigned char> fluidMask((size_t)W * H);
    for (size_t i = 0; i < (size_t)W * H; i++)
        fluidMask[i] = solidMask[i] ? 0 : 1;
    vector<double> d2_to_solid, d2_to_fluid;
    edt2d_squared(W, H, solidMask, d2_to_solid);
    edt2d_squared(W, H, fluidMask, d2_to_fluid);
    B.resize((size_t)W * H);
    double Bmin =  1e300, Bmax = -1e300;
    for (size_t i = 0; i < (size_t)W * H; i++) {
        // bwdist(fluidMask) - bwdist(solidMask): positive in solid.
        double s = sqrt(d2_to_fluid[i]) - sqrt(d2_to_solid[i]);
        B[i] = s;
        if (s < Bmin) Bmin = s;
        if (s > Bmax) Bmax = s;
    }
    double range = Bmax - Bmin;
    if (range > 0.0) {
        for (size_t i = 0; i < (size_t)W * H; i++) B[i] = (B[i] - Bmin) / range;
    } else {
        for (size_t i = 0; i < (size_t)W * H; i++) B[i] = 0.5;
    }
}

// Linear-interpolated quantile of `v` at fraction q in [0,1].  Uses the
// MATLAB default (Hyndman-Fan type 5) so that the threshold matches
// generateInterpPorousMirroring.m bit-for-bit modulo EDT tie-breaks:
//   data points sit at probabilities (j - 0.5)/n for j = 1..n (1-indexed);
//   targets outside that range clamp to the extremes.
static double quantileOf(const vector<double> &v, double q) {
    if (v.empty()) return 0.0;
    vector<double> w(v);
    std::sort(w.begin(), w.end());
    size_t n = w.size();
    double pLow  = 0.5 / double(n);
    double pHigh = (double(n) - 0.5) / double(n);
    if (q <= pLow)  return w.front();
    if (q >= pHigh) return w.back();
    double pos = q * double(n) - 0.5;     // zero-indexed real position
    size_t lo = (size_t)floor(pos);
    size_t hi = lo + 1;
    if (hi >= n) return w[n - 1];
    double t = pos - double(lo);
    return w[lo] * (1.0 - t) + w[hi] * t;
}

// MATLAB linspace(1/pad, 1 - 1/pad, pad), 0-indexed.
static double weightAt(int t, int pad) {
    if (pad == 1) return 0.5;
    return 1.0 / pad + double(t) * (1.0 - 2.0 / pad) / double(pad - 1);
}

// Read the input db file as text and re-emit it with the Domain section
// fields Filename, N, n, and MirrorPad updated for the extended geometry.
// nproc is preserved exactly so the downstream decomposition is unchanged.
// Returns the path of the new db file.
static string writeUpdatedDb(const string &inputDb,
                              const string &newFilename,
                              int newNx, int newNy, int newNz,
                              const vector<int> &nproc) {
    ifstream in(inputDb);
    if (!in) {
        printf("[mirror_pp]: WARNING cannot read %s for sibling db; skipping\n",
               inputDb.c_str());
        return "";
    }
    stringstream raw;
    raw << in.rdbuf();
    in.close();
    string text = raw.str();

    // Hand-roll the per-line substitutions: std::regex lookbehind isn't
    // portable across libstdc++ versions, so split on '\n' and rewrite
    // each Domain-section field by prefix-match on the trimmed line.
    int newNxLocal = newNx / nproc[0];
    int newNyLocal = newNy / nproc[1];
    int newNzLocal = newNz / nproc[2];

    auto leadingWhitespace = [](const string &line) -> string {
        size_t i = 0;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;
        return line.substr(0, i);
    };
    auto strippedStartsWith = [](const string &line, const string &key) -> bool {
        size_t i = 0;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;
        if (line.compare(i, key.size(), key) != 0) return false;
        size_t j = i + key.size();
        // Next non-whitespace char must be '=' for this to be the key.
        while (j < line.size() && (line[j] == ' ' || line[j] == '\t')) j++;
        return j < line.size() && line[j] == '=';
    };

    stringstream rebuilt;
    stringstream src(text);
    string line;
    while (std::getline(src, line)) {
        string pad = leadingWhitespace(line);
        if (strippedStartsWith(line, "Filename")) {
            rebuilt << pad << "Filename = \"" << newFilename << "\"\n";
        } else if (strippedStartsWith(line, "N")) {
            rebuilt << pad << "N = " << newNx << ", " << newNy << ", " << newNz << "\n";
        } else if (strippedStartsWith(line, "n")) {
            rebuilt << pad << "n = " << newNxLocal << ", " << newNyLocal << ", " << newNzLocal << "\n";
        } else if (strippedStartsWith(line, "MirrorPad")) {
            rebuilt << pad << "MirrorPad = 0, 0, 0\n";
        } else {
            rebuilt << line << "\n";
        }
    }
    text = rebuilt.str();

    size_t dot = inputDb.find_last_of('.');
    string outDb = (dot != string::npos)
                       ? inputDb.substr(0, dot) + "_mirrored" + inputDb.substr(dot)
                       : inputDb + "_mirrored.db";
    ofstream out(outDb);
    if (!out) {
        printf("[mirror_pp]: WARNING cannot write sibling db %s\n", outDb.c_str());
        return "";
    }
    out << text;
    out.close();
    return outDb;
}

// ============================================================================
// Append `pad` interp slices along the requested axis.  Geometry is stored
// flat as solid-mask (1 = solid, 0 = fluid) in `geo`, in (i,j,k) order with
// k slowest.  Axis = 0 (x), 1 (y), 2 (z).
// ============================================================================
static void mirrorAxis(vector<unsigned char> &geo,
                       int &Nx, int &Ny, int &Nz,
                       int axis, int pad) {
    if (pad <= 0) return;
    auto idx = [&](int i, int j, int k, int LNx, int LNy) -> size_t {
        return (size_t)k * LNx * LNy + (size_t)j * LNx + (size_t)i;
    };

    // Pull boundary slices in the (a, b) plane perpendicular to `axis`.
    int W, H;            // 2D slice dimensions
    if (axis == 0)      { W = Ny; H = Nz; }
    else if (axis == 1) { W = Nx; H = Nz; }
    else                { W = Nx; H = Ny; }

    vector<unsigned char> A1((size_t)W * H), A2((size_t)W * H);
    auto extractSlice = [&](int faceIndex, vector<unsigned char> &slice) {
        for (int b = 0; b < H; b++) {
            for (int a = 0; a < W; a++) {
                size_t s;
                if (axis == 0)      s = idx(faceIndex, a, b, Nx, Ny);
                else if (axis == 1) s = idx(a, faceIndex, b, Nx, Ny);
                else                s = idx(a, b, faceIndex, Nx, Ny);
                slice[(size_t)b * W + a] = geo[s];
            }
        }
    };
    extractSlice(0, A1);
    int lastFace = (axis == 0 ? Nx : axis == 1 ? Ny : Nz) - 1;
    extractSlice(lastFace, A2);

    vector<double> B1, B2;
    signedNormalizedDistance(W, H, A1, B1);
    signedNormalizedDistance(W, H, A2, B2);

    double s1 = 0.0, s2 = 0.0;
    for (size_t i = 0; i < (size_t)W * H; i++) { s1 += A1[i]; s2 += A2[i]; }
    double porosity1 = 1.0 - s1 / double(W * H);
    double porosity2 = 1.0 - s2 / double(W * H);

    // New dimensions
    int newNx = Nx, newNy = Ny, newNz = Nz;
    if (axis == 0) newNx += pad;
    if (axis == 1) newNy += pad;
    if (axis == 2) newNz += pad;
    vector<unsigned char> newGeo((size_t)newNx * newNy * newNz, 0);

    // Copy original
    for (int k = 0; k < Nz; k++)
        for (int j = 0; j < Ny; j++)
            for (int i = 0; i < Nx; i++)
                newGeo[idx(i, j, k, newNx, newNy)] = geo[idx(i, j, k, Nx, Ny)];

    // Generate and append interp slices
    vector<double> slice((size_t)W * H);
    for (int t = 0; t < pad; t++) {
        double Wt = weightAt(t, pad);
        for (size_t i = 0; i < (size_t)W * H; i++)
            slice[i] = Wt * B1[i] + (1.0 - Wt) * B2[i];
        double phi = Wt * porosity1 + (1.0 - Wt) * porosity2;
        double thresh = quantileOf(slice, phi);
        int slicePos = (axis == 0 ? Nx : axis == 1 ? Ny : Nz) + t;
        for (int b = 0; b < H; b++) {
            for (int a = 0; a < W; a++) {
                unsigned char v = (slice[(size_t)b * W + a] >= thresh) ? 1 : 0;
                size_t s;
                if (axis == 0)      s = idx(slicePos, a, b, newNx, newNy);
                else if (axis == 1) s = idx(a, slicePos, b, newNx, newNy);
                else                s = idx(a, b, slicePos, newNx, newNy);
                newGeo[s] = v;
            }
        }
    }
    geo = std::move(newGeo);
    Nx = newNx; Ny = newNy; Nz = newNz;
    printf("[mirror_pp]: extended axis %d by %d voxels "
           "(porosity %.4f -> %.4f); new dims: %d x %d x %d\n",
           axis, pad, porosity1, porosity2, Nx, Ny, Nz);
}

// ============================================================================
int main(int argc, char **argv) {
    int rank = 0, nprocs = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank != 0) { MPI_Finalize(); return 0; }   // serial preproc

    if (argc < 2) {
        printf("Usage: lbpm_mirror_pp <inputFile.db>\n");
        MPI_Finalize();
        return 1;
    }

    auto db        = make_shared<Database>(argv[1]);
    auto domain_db = db->getDatabase("Domain");

    auto Nvec = domain_db->getVector<int>("N");
    int  Nx = Nvec[0], Ny = Nvec[1], Nz = Nvec[2];
    string filename = domain_db->getScalar<string>("Filename");
    string readType = domain_db->keyExists("ReadType")
                          ? domain_db->getScalar<string>("ReadType")
                          : "8bit";

    if (!domain_db->keyExists("MirrorPad")) {
        printf("[mirror_pp]: Domain.MirrorPad not set; nothing to do.\n");
        MPI_Finalize();
        return 0;
    }
    auto pad = domain_db->getVector<int>("MirrorPad");
    int px = pad[0], py = pad[1], pz = pad[2];
    if (px <= 0 && py <= 0 && pz <= 0) {
        printf("[mirror_pp]: Domain.MirrorPad = (%d, %d, %d); nothing to do.\n",
               px, py, pz);
        MPI_Finalize();
        return 0;
    }

    string outFilename;
    if (domain_db->keyExists("MirrorOutput")) {
        outFilename = domain_db->getScalar<string>("MirrorOutput");
    } else {
        size_t dot = filename.find_last_of('.');
        if (dot != string::npos)
            outFilename = filename.substr(0, dot) + "_mirrored" + filename.substr(dot);
        else
            outFilename = filename + "_mirrored.raw";
    }

    printf("================================================================\n");
    printf("  lbpm_mirror_pp: interporous mirror preprocessor\n");
    printf("================================================================\n");
    printf("  input    = %s  (%d x %d x %d, %s)\n",
           filename.c_str(), Nx, Ny, Nz, readType.c_str());
    printf("  pad      = (%d, %d, %d)\n", px, py, pz);
    printf("  output   = %s\n", outFilename.c_str());
    printf("================================================================\n");

    // Read input raw
    size_t total = (size_t)Nx * Ny * Nz;
    vector<unsigned char> data(total);
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        printf("[mirror_pp]: ERROR cannot open %s for reading\n", filename.c_str());
        MPI_Finalize();
        return 1;
    }
    if (readType == "16bit") {
        vector<short int> raw16(total);
        size_t nr = fread(raw16.data(), 2, total, fp);
        fclose(fp);
        if (nr != total) {
            printf("[mirror_pp]: ERROR short read on %s\n", filename.c_str());
            MPI_Finalize();
            return 1;
        }
        for (size_t i = 0; i < total; i++)
            data[i] = (unsigned char)raw16[i];
    } else {
        size_t nr = fread(data.data(), 1, total, fp);
        fclose(fp);
        if (nr != total) {
            printf("[mirror_pp]: ERROR short read on %s\n", filename.c_str());
            MPI_Finalize();
            return 1;
        }
    }

    // Diagnose: warn if input has labels other than 0/1 (suggests post-morphopen).
    bool sawMulti = false;
    for (size_t i = 0; i < total; i++) {
        if (data[i] > 1) { sawMulti = true; break; }
    }
    if (sawMulti)
        printf("[mirror_pp]: WARNING input contains labels > 1 -- flattening "
               "to solid (0) / fluid (1) for the mirror.  Run this stage BEFORE "
               "morphopen on raw segmented geometry.\n");

    // Convert to solid-mask convention (1 = solid).  In the LBPM petrophysics
    // convention the raw file stores 0 = solid, nonzero = fluid (NWP=1, WP=2).
    vector<unsigned char> geo(total);
    for (size_t i = 0; i < total; i++) geo[i] = (data[i] == 0) ? 1 : 0;
    data.clear();
    data.shrink_to_fit();

    // Apply mirror padding axis by axis.  Match the MATLAB ordering (z, y, x)
    // so successive extensions composed identically.
    if (pz > 0) mirrorAxis(geo, Nx, Ny, Nz, /*axis=*/2, pz);
    if (py > 0) mirrorAxis(geo, Nx, Ny, Nz, /*axis=*/1, py);
    if (px > 0) mirrorAxis(geo, Nx, Ny, Nz, /*axis=*/0, px);

    // Convert solid-mask back to LBPM convention (0 = solid, 1 = fluid).
    size_t newTotal = geo.size();
    vector<unsigned char> out(newTotal);
    for (size_t i = 0; i < newTotal; i++) out[i] = geo[i] ? 0 : 1;
    geo.clear();
    geo.shrink_to_fit();

    // Write extended raw
    FILE *fpo = fopen(outFilename.c_str(), "wb");
    if (!fpo) {
        printf("[mirror_pp]: ERROR cannot open %s for writing\n", outFilename.c_str());
        MPI_Finalize();
        return 1;
    }
    if (readType == "16bit") {
        vector<short int> raw16(newTotal);
        for (size_t i = 0; i < newTotal; i++) raw16[i] = out[i];
        fwrite(raw16.data(), 2, newTotal, fpo);
    } else {
        fwrite(out.data(), 1, newTotal, fpo);
    }
    fclose(fpo);

    printf("\n[mirror_pp]: wrote %s  (%d x %d x %d)\n",
           outFilename.c_str(), Nx, Ny, Nz);

    // Emit a sibling db with Filename + N + n updated and MirrorPad zeroed,
    // so chaining lbpm_serial_decomp / lbpm_morphopen_pp / simulators on
    // the new db just works.
    auto nproc = domain_db->getVector<int>("nproc");
    string outDb = writeUpdatedDb(argv[1], outFilename, Nx, Ny, Nz, nproc);
    if (!outDb.empty()) {
        printf("[mirror_pp]: wrote sibling input db %s -- point downstream stages at it\n",
               outDb.c_str());
    } else {
        printf("\nFallback: update your inputFile.db Domain section manually:\n");
        printf("    Filename = \"%s\"\n", outFilename.c_str());
        printf("    N = %d, %d, %d\n", Nx, Ny, Nz);
        printf("    n = %d, %d, %d   (assuming nproc = %d, %d, %d)\n",
               Nx / nproc[0], Ny / nproc[1], Nz / nproc[2],
               nproc[0], nproc[1], nproc[2]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}
