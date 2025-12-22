import { NobelApiRepository } from './api'
import type { AjaxService } from '@/services/core/ajaxService'

export class NobelService {
  private repository: NobelApiRepository

  constructor({ ajaxService }: { ajaxService: AjaxService }) {
    this.repository = new NobelApiRepository({ ajaxService })
  }

  public async fetchLaureates(params?: Record<string, any>) {
    return this.repository.getLaureates(params)
  }

  public async fetchPrizes(params?: Record<string, any>) {
    return this.repository.getPrizes(params)
  }
}