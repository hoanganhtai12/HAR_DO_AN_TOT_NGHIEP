// gcc -O3 -shared unpack.c -o unpack.dll
#include <stdint.h>

#define k_tof_unpack_sgn_mask (1u << 31)

void unpack_float_acphy(
    int nbits,
    int autoscale,
    int shft,
    int fmt,
    int nman,
    int nexp,
    int nfft,
    uint32_t *H,
    int32_t *Hout
)
{
    int e_p, maxbit, e, i, e_zero, sgn;
    int n_out, e_shift;

    int8_t He[256];
    int32_t vi, vq, *pOut;
    uint32_t x, iq_mask, e_mask;
    uint32_t sgnr_mask, sgni_mask;

    iq_mask = (1u << (nman - 1)) - 1u;
    e_mask = (1u << nexp) - 1u;
    e_p = (1 << (nexp - 1));
    sgnr_mask = (1u << (nexp + 2 * nman - 1));
    sgni_mask = (sgnr_mask >> nman);

    e_zero = -nman;
    pOut = Hout;
    n_out = (nfft << 1);
    e_shift = 1;
    maxbit = -e_p;

    for (i = 0; i < nfft; i++) {
        vi = (int32_t)((H[i] >> (nexp + nman)) & iq_mask);
        vq = (int32_t)((H[i] >> nexp) & iq_mask);
        e = (int)(H[i] & e_mask);

        if (e >= e_p)
            e -= (e_p << 1);

        He[i] = (int8_t)e;
        x = (uint32_t)vi | (uint32_t)vq;
        if (autoscale && x) {

            uint32_t m = 0xffff0000u;
            uint32_t b = 0xffffu;

            int s = 16;

            while (s > 0) {

                if (x & m) {
                    e += s;
                    x >>= s;
                }

                s >>= 1;

                m = (m >> s) & b;

                b >>= s;
            }

            if (e > maxbit)
                maxbit = e;
        }

        if (H[i] & sgnr_mask)
            vi |= k_tof_unpack_sgn_mask;

        if (H[i] & sgni_mask)
            vq |= k_tof_unpack_sgn_mask;

        Hout[i << 1] = vi;

        Hout[(i << 1) + 1] = vq;
    }

    if(autoscale)
        shft = nbits - maxbit;
    else
        shft = nbits;

    for (i = 0; i < n_out; i++) {
        e = He[(i >> e_shift)] + shft;
        vi = *pOut;
        sgn = 1;
        if (vi & k_tof_unpack_sgn_mask) {
            sgn = -1;
            vi &= ~k_tof_unpack_sgn_mask;
        }

        if (e < e_zero) {
            vi = 0;
        } else 
        if (e < 0) {
            e = -e;
            vi = (vi >> e);
        } else {
            vi = (vi << e);
        }

        *pOut++ = (int32_t)(sgn * vi);
    }
}