/**
 * Analytical solutions for built-in processes.
 * Returns theoretical E[X_t] and Std(X_t) on a given time grid.
 */

export type TheoreticalSolution = {
  t: number[]
  mean: number[]
  std: number[]
  /** LaTeX for the main formula (display mode). */
  formulaLatex: string
  /** LaTeX for stationary distribution, if applicable. */
  stationaryLatex?: string
}

function ouSolution(
  t: number[],
  x0: number,
  theta: number,
  mu: number,
  sigma: number
): TheoreticalSolution {
  const mean = t.map((s) => mu + (x0 - mu) * Math.exp(-theta * s))
  const variance = t.map((s) => (sigma * sigma / (2 * theta)) * (1 - Math.exp(-2 * theta * s)))
  const std = variance.map((v) => Math.sqrt(v))
  return {
    t,
    mean,
    std,
    formulaLatex:
      'E[X_t] = \\mu + (X_0 - \\mu)e^{-\\theta t}, \\quad \\mathrm{Var}(X_t) = \\frac{\\sigma^2}{2\\theta}(1 - e^{-2\\theta t})',
    stationaryLatex:
      '\\text{As } t \\to \\infty: \\quad X_\\infty \\sim N\\left(\\mu, \\frac{\\sigma^2}{2\\theta}\\right)',
  }
}

function brownianSolution(t: number[], x0: number): TheoreticalSolution {
  const mean = t.map(() => x0)
  const std = t.map((s) => Math.sqrt(s))
  return {
    t,
    mean,
    std,
    formulaLatex: 'E[X_t] = X_0, \\quad \\mathrm{Var}(X_t) = t',
    stationaryLatex: undefined,
  }
}

function gbmSolution(
  t: number[],
  x0: number,
  mu: number,
  sigma: number
): TheoreticalSolution {
  const mean = t.map((s) => x0 * Math.exp(mu * s))
  const variance = t.map((s) =>
    x0 * x0 * Math.exp(2 * mu * s) * (Math.exp(sigma * sigma * s) - 1)
  )
  const std = variance.map((v) => Math.sqrt(v))
  return {
    t,
    mean,
    std,
    formulaLatex:
      'E[X_t] = X_0 e^{\\mu t}, \\quad \\mathrm{Var}(X_t) = X_0^2 e^{2\\mu t}(e^{\\sigma^2 t} - 1)',
    stationaryLatex: undefined,
  }
}

function cirSolution(
  t: number[],
  x0: number,
  kappa: number,
  theta: number,
  sigma: number
): TheoreticalSolution {
  const mean = t.map((s) => theta + (x0 - theta) * Math.exp(-kappa * s))
  const variance = t.map((s) => {
    const ekt = Math.exp(-kappa * s)
    return (
      x0 * (sigma * sigma / kappa) * (ekt - ekt * ekt) +
      theta * (sigma * sigma / (2 * kappa)) * (1 - ekt) * (1 - ekt)
    )
  })
  const std = variance.map((v) => Math.sqrt(Math.max(v, 0)))
  return {
    t,
    mean,
    std,
    formulaLatex:
      'E[X_t] = \\theta + (X_0 - \\theta)e^{-\\kappa t}, \\quad ' +
      '\\mathrm{Var}(X_t) = \\frac{X_0 \\sigma^2}{\\kappa}\\bigl(e^{-\\kappa t} - e^{-2\\kappa t}\\bigr) ' +
      '+ \\frac{\\theta\\sigma^2}{2\\kappa}\\bigl(1 - e^{-\\kappa t}\\bigr)^2',
    stationaryLatex:
      '\\text{As } t \\to \\infty: \\quad X_\\infty \\sim \\mathrm{Gamma}\\!\\left(\\frac{2\\kappa\\theta}{\\sigma^2},\\, \\frac{2\\kappa}{\\sigma^2}\\right), \\quad ' +
      'E[X_\\infty] = \\theta, \\quad \\mathrm{Var}(X_\\infty) = \\frac{\\theta\\sigma^2}{2\\kappa}',
  }
}

/**
 * Get theoretical mean and std for a process on the given time grid, or null if no closed form.
 */
export function getTheoreticalSolution(
  processId: string,
  params: Record<string, number>,
  x0: number,
  t: number[]
): TheoreticalSolution | null {
  if (t.length === 0) return null
  switch (processId) {
    case 'brownian':
      return brownianSolution(t, x0)
    case 'ornstein-uhlenbeck':
      return ouSolution(t, x0, params.theta, params.mu, params.sigma)
    case 'geometric-brownian':
      return gbmSolution(t, x0, params.mu, params.sigma)
    case 'cir':
      return cirSolution(t, x0, params.kappa, params.theta, params.sigma)
    default:
      return null
  }
}

/**
 * Closed-form solution p(x,t) of the Fokker-Planck equation when available.
 * Returns LaTeX string(s) to display, or null if no closed form.
 */
export function getDensityFormulaLatex(processId: string): { main: string; where?: string } | null {
  switch (processId) {
    case 'brownian':
      return {
        main:
          'p(x, t) = \\frac{1}{\\sqrt{2\\pi t}} \\exp\\left(-\\frac{(x - X_0)^2}{2t}\\right) \\quad (t > 0)',
      }
    case 'ornstein-uhlenbeck':
      return {
        main:
          'p(x, t) = \\frac{1}{\\sqrt{2\\pi \\sigma_t^2}} \\exp\\left(-\\frac{(x - \\mu_t)^2}{2\\sigma_t^2}\\right)',
        where:
          '\\mu_t = \\mu + (X_0 - \\mu)e^{-\\theta t}, \\quad \\sigma_t^2 = \\frac{\\sigma^2}{2\\theta}(1 - e^{-2\\theta t})',
      }
    case 'geometric-brownian':
      return {
        main:
          'p(x, t) = \\frac{1}{x\\sigma\\sqrt{2\\pi t}} \\exp\\left(-\\frac{\\big(\\ln(x/X_0) - (\\mu - \\sigma^2/2)t\\big)^2}{2\\sigma^2 t}\\right) \\quad (x > 0,\\ t > 0)',
      }
    default:
      return null
  }
}
