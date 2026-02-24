import type { ProcessDef } from '@/types/process'

/**
 * Cox–Ingersoll–Ross (CIR) process: dX = κ(θ − X) dt + σ√X dW
 *
 * Stays non-negative when the Feller condition 2κθ ≥ σ² holds.
 * We clamp the argument of √ to 0 to keep the simulation stable even when
 * a discrete step transiently pushes X below zero.
 */
export const cir: ProcessDef = {
  id: 'cir',
  name: 'CIR (Cox–Ingersoll–Ross)',
  description: 'mean-reverting, non-negative',
  equationLatex: 'dX = \\kappa(\\theta - X)\\,dt + \\sigma\\sqrt{X}\\,dW',
  about:
    'Widely used in finance to model short-term interest rates (Cox, Ingersoll & Ross, 1985) ' +
    'and as the variance process in the Heston stochastic-volatility model. ' +
    'When the Feller condition 2κθ ≥ σ² holds, the process never reaches zero; ' +
    'the stationary distribution is a Gamma distribution with shape 2κθ/σ² and rate 2κ/σ².',
  params: [
    { id: 'kappa', name: 'κ (mean reversion speed)', default: 1, min: 0.01, max: 10, step: 0.01 },
    { id: 'theta', name: 'θ (long-run mean)', default: 1, min: 0.01, max: 10, step: 0.01 },
    { id: 'sigma', name: 'σ (volatility)', default: 0.5, min: 0.01, max: 5, step: 0.01 },
  ],
  drift: (x, _t, p) => p.kappa * (p.theta - x),
  diffusion: (x, _t, p) => p.sigma * Math.sqrt(Math.max(x, 0)),
}
