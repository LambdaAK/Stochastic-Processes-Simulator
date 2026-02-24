import type { ProcessDef } from '@/types/process'

/**
 * d-dimensional Bessel process: dR = (d−1)/(2R) dt + dW
 *
 * R_t is the Euclidean norm of a d-dimensional Brownian motion.
 * For d ≥ 2 the process is non-negative; for d ≥ 3 it never touches zero.
 * We clamp the denominator away from zero to keep the simulation stable.
 */
export const bessel: ProcessDef = {
  id: 'bessel',
  name: 'Bessel process',
  description: 'BM radial distance (d-dimensional)',
  equationLatex: 'dR = \\frac{d-1}{2R}\\,dt + dW',
  about:
    'The Bessel process of order d is the radial part (Euclidean norm) of a d-dimensional Brownian motion. ' +
    'It arises naturally in physics (3D diffusion, polymer models) and in the study of Brownian local times. ' +
    'For d ≥ 2 the process stays non-negative; for d ≥ 3 it never returns to zero. ' +
    'E[R_t²] = R₀² + d·t, but E[R_t] has no elementary closed form.',
  params: [
    { id: 'd', name: 'd (dimension)', default: 3, min: 1, max: 10, step: 1 },
  ],
  drift: (x, _t, p) => (p.d - 1) / (2 * Math.max(Math.abs(x), 1e-6)),
  diffusion: () => 1,
}
