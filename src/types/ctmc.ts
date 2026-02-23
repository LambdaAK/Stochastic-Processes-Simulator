export type CtmcTransition = {
  from: string
  to: string
  rate: number
}

export type CtmcDef = {
  states: string[]
  /** Initial distribution over states; keys are state names, values are probabilities (sum to 1). */
  initialDistribution: Record<string, number>
  transitions: CtmcTransition[]
}

export type CtmcParseResult =
  | { ok: true; chain: CtmcDef }
  | { ok: false; error: string }
