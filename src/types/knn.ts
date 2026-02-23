import type { Point2D, TrainingPoint } from '@/lib/knn'

export type { Point2D, TrainingPoint }

export type KNNDatasetPreset = 'blobs' | 'xor' | 'circles'

export type KNNSectionState = {
  preset: KNNDatasetPreset
  training: TrainingPoint[]
  k: number
  query: Point2D | null
  predictedLabel: string | null
  showDecisionBoundary: boolean
  gridResolution: number
}
