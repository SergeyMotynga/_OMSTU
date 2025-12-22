import * as t from 'io-ts'
import { LaureateCodec } from './LaureateCodec'
import { PrizeCodec } from './PrizeCodec'

export const LaureatesResponseCodec = t.type({
  laureates: t.array(LaureateCodec),
})
export type LaureatesResponse = t.TypeOf<typeof LaureatesResponseCodec>

export const PrizesResponseCodec = t.type({
  nobelPrizes: t.array(PrizeCodec),
})
export type PrizesResponse = t.TypeOf<typeof PrizesResponseCodec>