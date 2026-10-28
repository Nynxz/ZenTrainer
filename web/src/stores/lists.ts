// The Datasets and Models pages' lists as last fetched: a page shows them at once when it's opened
// again and refreshes them in place, instead of starting empty and popping in.
import { ref } from 'vue'
import type { DatasetInfo, ModelFamily } from '@/api/engine'

export const datasetsSeen = ref<DatasetInfo[] | null>(null)
export const familiesSeen = ref<ModelFamily[]>([])
