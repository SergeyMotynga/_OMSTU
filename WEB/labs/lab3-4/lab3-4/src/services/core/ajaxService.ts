import type { AxiosInstance, AxiosRequestConfig } from 'axios'
import axios from 'axios'

interface Request<T = any> {
  url: string
  data?: T
  config?: AxiosRequestConfig
}

export class AjaxService {
  private _axiosClient: AxiosInstance

  constructor(baseURL: string) {
    this._axiosClient = axios.create({
      baseURL,
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
    })
  }

  public async get({ url, config }: Request) {
    return (await this._axiosClient.get(url, config)).data
  }

  // public async post({ url, data, config }: Request) {
  //   return (await this._axiosClient.post(url, data, config)).data
  // }
}