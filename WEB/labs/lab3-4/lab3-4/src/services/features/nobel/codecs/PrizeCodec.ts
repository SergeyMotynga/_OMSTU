import * as t from 'io-ts'

const LaureateInPrizeCodec = t.type({
  id: t.string,
})

export const PrizeCodec = t.type({
  awardYear: t.union([t.string, t.undefined]),
  category: t.type({ en: t.string}),
  laureates: t.union([ t.undefined, t.array(LaureateInPrizeCodec)]),
})

export type Prize = t.TypeOf<typeof PrizeCodec>