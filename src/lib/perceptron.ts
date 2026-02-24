/**
 * Perceptron: online linear binary classifier (Rosenblatt, 1958).
 * Labels: +1 or -1.
 * Predict: ŷ = sign(w₁x + w₂y + b)
 * Update (on error): w ← w + η·label·x, b ← b + η·label
 * Converges iff data is linearly separable (Rosenblatt convergence theorem).
 */

export type PerceptronModel = {
  w1: number
  w2: number
  b: number
}

export type TrainHistory = {
  epoch: number
  errors: number
}

export type TrainResult = {
  model: PerceptronModel
  epochs: number
  history: TrainHistory[]
  converged: boolean
  /** Model weights after each epoch. snapshots[0] = initial (all zeros), snapshots[k] = after epoch k. */
  snapshots: PerceptronModel[]
}

export type DataPoint = {
  x: number
  y: number
  label: 1 | -1
}

/** Predict label (+1 or -1) for a 2D point. */
export function predict(x: number, y: number, model: PerceptronModel): 1 | -1 {
  return model.w1 * x + model.w2 * y + model.b >= 0 ? 1 : -1
}

/**
 * Train the perceptron on a binary dataset (labels +1 / -1).
 * Returns model, per-epoch error history, and convergence status.
 */
export function train(
  data: DataPoint[],
  options?: { learningRate?: number; maxEpochs?: number }
): TrainResult {
  const lr = options?.learningRate ?? 1
  const maxEpochs = options?.maxEpochs ?? 100

  let w1 = 0
  let w2 = 0
  let b = 0

  const history: TrainHistory[] = []
  const snapshots: PerceptronModel[] = [{ w1: 0, w2: 0, b: 0 }]

  for (let epoch = 1; epoch <= maxEpochs; epoch++) {
    let errors = 0
    for (const pt of data) {
      const pred: 1 | -1 = w1 * pt.x + w2 * pt.y + b >= 0 ? 1 : -1
      if (pred !== pt.label) {
        errors++
        w1 += lr * pt.label * pt.x
        w2 += lr * pt.label * pt.y
        b += lr * pt.label
      }
    }
    history.push({ epoch, errors })
    snapshots.push({ w1, w2, b })
    if (errors === 0) break
  }

  const converged = history[history.length - 1]?.errors === 0

  return { model: { w1, w2, b }, epochs: history.length, history, converged, snapshots }
}

/**
 * Compute the two endpoints of the decision boundary line clipped to the
 * chart bounding box [xMin,xMax] × [yMin,yMax].
 * Returns [] if weights are zero or the line doesn't cross the box.
 */
export function boundaryLine(
  model: PerceptronModel,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number
): { x: number; y: number }[] {
  const { w1, w2, b } = model
  if (Math.abs(w1) < 1e-10 && Math.abs(w2) < 1e-10) return []

  const pts: { x: number; y: number }[] = []

  // Intersect with left (x=xMin) and right (x=xMax) edges
  if (Math.abs(w2) > 1e-10) {
    const f = (x: number) => -(w1 * x + b) / w2
    const yL = f(xMin)
    const yR = f(xMax)
    if (yL >= yMin && yL <= yMax) pts.push({ x: xMin, y: yL })
    if (yR >= yMin && yR <= yMax) pts.push({ x: xMax, y: yR })
  }

  // Intersect with bottom (y=yMin) and top (y=yMax) edges
  if (Math.abs(w1) > 1e-10) {
    const g = (y: number) => -(w2 * y + b) / w1
    const xB = g(yMin)
    const xT = g(yMax)
    if (xB > xMin && xB < xMax) pts.push({ x: xB, y: yMin })
    if (xT > xMin && xT < xMax) pts.push({ x: xT, y: yMax })
  }

  if (pts.length < 2) return []

  // Sort by x and return first and last
  pts.sort((a, c) => a.x - c.x)
  return [pts[0], pts[pts.length - 1]]
}

/** Generate linearly separable 2D binary datasets for the perceptron demo. */
export function generateDataset(
  preset: 'blobs' | 'diagonal' | 'vertical' | 'wide-margin',
  nPerClass: number,
  rand: () => number
): DataPoint[] {
  const gaussian = () =>
    Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-15))) * Math.cos(2 * Math.PI * rand())

  const out: DataPoint[] = []

  if (preset === 'blobs') {
    // Two Gaussian blobs at diagonal positions
    for (let i = 0; i < nPerClass; i++) {
      out.push({ x: 1.5 + 0.8 * gaussian(), y: 1.5 + 0.8 * gaussian(), label: 1 })
      out.push({ x: -1.5 + 0.8 * gaussian(), y: -1.5 + 0.8 * gaussian(), label: -1 })
    }
  } else if (preset === 'diagonal') {
    // Separated by y = x: class +1 above the line, class -1 below
    for (let i = 0; i < nPerClass; i++) {
      const x1 = -3 + 6 * rand()
      out.push({ x: x1 + 0.2 * gaussian(), y: x1 + 0.6 + 0.4 * gaussian(), label: 1 })
      const x2 = -3 + 6 * rand()
      out.push({ x: x2 + 0.2 * gaussian(), y: x2 - 0.6 + 0.4 * gaussian(), label: -1 })
    }
  } else if (preset === 'vertical') {
    // Separated by a vertical boundary at x = 0
    for (let i = 0; i < nPerClass; i++) {
      out.push({ x: 0.6 + 0.7 * Math.abs(gaussian()), y: -3 + 6 * rand(), label: 1 })
      out.push({ x: -0.6 - 0.7 * Math.abs(gaussian()), y: -3 + 6 * rand(), label: -1 })
    }
  } else {
    // Wide margin: two blobs far apart with a large separating gap
    for (let i = 0; i < nPerClass; i++) {
      out.push({ x: 0.5 * gaussian(), y: 2.5 + 0.5 * gaussian(), label: 1 })
      out.push({ x: 0.5 * gaussian(), y: -2.5 + 0.5 * gaussian(), label: -1 })
    }
  }

  return out
}
