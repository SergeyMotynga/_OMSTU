import type { AjaxService } from '@/services/core/ajaxService'
import type { AxiosRequestConfig } from 'axios'
import { decodeOrThrow } from '@/services/core/decodeOrThrow'
import { LaureatesResponseCodec, PrizesResponseCodec } from './codecs/ApiResponseCodecs'
import type { LaureatesResponse, PrizesResponse } from './codecs/ApiResponseCodecs'

export class NobelApiRepository {
  private ajaxService: AjaxService

  constructor({ ajaxService }: { ajaxService: AjaxService }) {
    this.ajaxService = ajaxService
  }

  public async getLaureates(params?: AxiosRequestConfig['params']): Promise<LaureatesResponse> {
    const json = await this.ajaxService.get({
      url: '/laureates',
      config: { params },
    })
    return decodeOrThrow(LaureatesResponseCodec, json)
  }

  public async getPrizes(params?: AxiosRequestConfig['params']): Promise<PrizesResponse> {
    const json = await this.ajaxService.get({
      url: '/nobelPrizes',
      config: { params },
    })
    return decodeOrThrow(PrizesResponseCodec, json)
  }
}